"""Encoders and value-prediction head.

Implements:
  - ContextEncoder E_ctx(·; φ_ctx) producing h_ctx for a dialogue.
  - ValuePredictionHead f_val(·; ψ) producing p_pred^{(q,k)} (Eq. 2).
  - IntentEncoder E_intent(·; φ_int) producing h_intent^{(t)}.

For CODE_REPRO smoke tests we provide a `--mock` path with a tiny random-init
transformer (`MockTinyTransformer`) so the whole pipeline can run on CPU
without downloading real weights. The HF-backed path is also present and
wraps an `AutoModel` + mean-pool .
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mock backbone (used when --mock; no external downloads)
# ---------------------------------------------------------------------------
class MockTinyTransformer(nn.Module):
    """Random-init mini transformer (CPU-friendly) for dry-run pipelines.

    vocab=1000, hidden=64, 2 layers, 4 heads — matches the CODE_REPRO budget.
    """

    def __init__(self, vocab_size: int = 1000, hidden: int = 64, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.hidden_size = hidden
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids) # (B, L, H)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool() # True = pad
        else:
            key_padding_mask = None
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return h # (B, L, H)


def _mean_pool(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Context encoder (Eq. 2 input)
# ---------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    """E_ctx(·; φ_ctx): produces h_ctx = mean-pool(transformer(X^{(k)}))."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        pool: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.pool = pool

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, torch.Tensor):
            hidden = out
        else:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("Backbone did not return last_hidden_state.")
        if self.pool == "mean":
            return _mean_pool(hidden, attention_mask)
        if self.pool == "cls":
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pool: {self.pool!r}")


# ---------------------------------------------------------------------------
# Value prediction head f_val (Eq. 2 output)
# ---------------------------------------------------------------------------
class ValuePredictionHead(nn.Module):
    """f_val: h_ctx -> p_pred over the WVS answer choices for question q (Eq. 2)."""

    def __init__(self, hidden_size: int, num_choices: int, mlp_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_choices),
        )
        self.num_choices = num_choices

    def forward(self, h_ctx: torch.Tensor) -> torch.Tensor:
        logits = self.net(h_ctx)
        return F.softmax(logits, dim=-1) # p_pred^{(q,k)}


# ---------------------------------------------------------------------------
# Intent encoder E_intent
# ---------------------------------------------------------------------------
class IntentEncoder(nn.Module):
    """E_intent(·; φ_int).

    Encodes a dialogue context concatenated with the teacher-generated intent
    annotation y_t and returns h_intent^{(t)}. A linear projection maps the
    pooled hidden state into the cultural prototype space R^{d_c} so that
    cosine similarity in Eq. (4) is well-defined.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        prototype_dim: int,
        pool: str = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, prototype_dim)
        self.pool = pool

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, torch.Tensor):
            hidden = out
        else:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("Backbone did not return last_hidden_state.")
        if self.pool == "mean":
            pooled = _mean_pool(hidden, attention_mask)
        else:
            pooled = hidden[:, 0, :]
        return self.proj(pooled) # (B, d_c)


# ---------------------------------------------------------------------------
# Factory: build encoder stack from config (--mock or HF)
# ---------------------------------------------------------------------------
@dataclass
class EncoderBundle:
    context_encoder: ContextEncoder
    intent_encoder: IntentEncoder
    value_head: ValuePredictionHead
    hidden_size: int
    prototype_dim: int


def build_encoders(
    cfg: Dict[str, Any],
    prototype_dim: int,
    mock: bool = True,
    model_card: Optional[Dict[str, Any]] = None,
) -> EncoderBundle:
    """Construct E_ctx, E_intent, f_val based on config.

    `mock=True` builds CPU-friendly MockTinyTransformer backbones.
    `mock=False` would build HF AutoModel backbones; that path is intentionally
    left as a thin try/except so CODE_REPRO does not require model downloads.
    """
    ucca = cfg["ucca"]
    pool_ctx = ucca["context_encoder"]["pool"]
    pool_int = "mean"
    num_choices = ucca["value_head"]["num_choices"]
    mlp_hidden = ucca["value_head"]["mlp_hidden"]
    share_backbone = bool(ucca["intent_encoder"].get("share_backbone", False))

    if mock:
        hidden = int(ucca["context_encoder"].get("hidden_size", 64))
        ctx_backbone = MockTinyTransformer(hidden=hidden)
        if share_backbone:
            int_backbone = ctx_backbone
        else:
            int_backbone = MockTinyTransformer(hidden=hidden)
    else: # pragma: no cover — HF path (requires downloads, not exercised in tests)
        from transformers import AutoModel # type: ignore

        hf_id = (model_card or {}).get("hf_id") or ucca["context_encoder"]["backbone"]
        ctx_backbone_hf = AutoModel.from_pretrained(hf_id)
        hidden = ctx_backbone_hf.config.hidden_size

        class _HFWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask=None):
                return self.m(input_ids=input_ids, attention_mask=attention_mask)

        ctx_backbone = _HFWrapper(ctx_backbone_hf)
        if share_backbone:
            int_backbone = ctx_backbone
        else:
            int_backbone = _HFWrapper(AutoModel.from_pretrained(hf_id))

    ctx = ContextEncoder(ctx_backbone, hidden_size=hidden, pool=pool_ctx)
    inten = IntentEncoder(
        int_backbone, hidden_size=hidden, prototype_dim=prototype_dim, pool=pool_int
    )
    head = ValuePredictionHead(hidden_size=hidden, num_choices=num_choices, mlp_hidden=mlp_hidden)
    return EncoderBundle(
        context_encoder=ctx,
        intent_encoder=inten,
        value_head=head,
        hidden_size=hidden,
        prototype_dim=prototype_dim,
    )
