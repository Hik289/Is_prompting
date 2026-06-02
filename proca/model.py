"""ProCAModel — glues encoders + value head + losses + LoRA.

`forward(batch)` returns a dict with:
  - total: scalar tensor (Eq. 5)
  - kl: scalar tensor (Eq. 3)
  - cont: scalar tensor (Eq. 4)
  - p_pred:(B, C) for downstream confidence (Eq. 6)
  - h_intent_pooled: (B, d_c) aggregated intent representation (used for Eq. 6)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DialogueBatch
from .encoders import EncoderBundle
from .losses import unified_loss
from .prototypes import PrototypeBank


class ProCAModel(nn.Module):
    def __init__(
        self,
        encoders: EncoderBundle,
        prototypes: PrototypeBank,
        lam: float = 0.5,
        tau: float = 0.07,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.context_encoder = encoders.context_encoder
        self.intent_encoder = encoders.intent_encoder
        self.value_head = encoders.value_head
        self.prototype_dim = encoders.prototype_dim
        self.lam = float(lam)
        self.tau = float(tau)

        # Prototypes are FIXED anchors: register as buffer, not parameter.
        self.register_buffer("prototypes", prototypes.matrix.float(), persistent=True)
        self.cultures = list(prototypes.cultures)
        self.culture_to_id = {c: i for i, c in enumerate(self.cultures)}

        # LoRA installation hook. Wired through PEFT
        # only when caller explicitly requests it AND a real HF backbone is in
        # place. For --mock the LoRA layer is a no-op stub that simply records
        # the config so tests can assert it was attached.
        self.lora_cfg = dict(lora_cfg or {})
        self._lora_attached = False
        if self.lora_cfg.get("attach", False):
            self._attach_lora_best_effort()

    # ------------------------------------------------------------------
    def _attach_lora_best_effort(self) -> None:
        """Best-effort LoRA attach via PEFT; falls back to a tagged no-op."""
        try:
            from peft import LoraConfig, get_peft_model # type: ignore

            cfg = LoraConfig(
                r=int(self.lora_cfg.get("r", 64)),
                lora_alpha=int(self.lora_cfg.get("alpha", 128)),
                lora_dropout=float(self.lora_cfg.get("dropout", 0.05)),
                target_modules=list(self.lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            # PEFT expects HF model-style modules. For Mock backbones this will
            # not find target_modules; we just record the intent.
            try:
                self.context_encoder.backbone = get_peft_model(self.context_encoder.backbone, cfg)
                self.intent_encoder.backbone = get_peft_model(self.intent_encoder.backbone, cfg)
                self._lora_attached = True
            except Exception: # pragma: no cover
                self._lora_attached = False
        except ImportError: # pragma: no cover
            self._lora_attached = False

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ensure_proto_dim(self, h_intent: torch.Tensor) -> torch.Tensor:
        # The intent encoder already projects to d_c; this is a sanity check.
        if h_intent.shape[-1] != self.prototype_dim:
            raise RuntimeError(
                f"h_intent dim {h_intent.shape[-1]} != prototype_dim {self.prototype_dim}"
            )
        return h_intent

    # ------------------------------------------------------------------
    def encode_intent_per_turn(
        self,
        intent_ids: torch.Tensor, # (B, T', L)
        intent_mask: torch.Tensor, # (B, T', L)
    ) -> torch.Tensor:
        B, Tp, L = intent_ids.shape
        flat_ids = intent_ids.reshape(B * Tp, L)
        flat_mask = intent_mask.reshape(B * Tp, L)
        h = self.intent_encoder(flat_ids, flat_mask) # (B*T', d_c)
        return h.reshape(B, Tp, -1) # (B, T', d_c)

    # ------------------------------------------------------------------
    def forward(self, batch: DialogueBatch) -> Dict[str, torch.Tensor]:
        # 1) E_ctx → h_ctx, then f_val → p_pred (Eq. 2)
        h_ctx = self.context_encoder(batch.ctx_input_ids, batch.ctx_attention_mask)
        p_pred = self.value_head(h_ctx) # (B, C)

        # 2) E_intent per turn → (B, T', d_c)
        h_intent_per_turn = self.encode_intent_per_turn(
            batch.intent_input_ids, batch.intent_attention_mask
        )
        self._ensure_proto_dim(h_intent_per_turn)

        # 3) Unified loss (Eq. 3 + Eq. 4 → Eq. 5)
        losses = unified_loss(
            p_pred=p_pred,
            p_wvs=batch.p_wvs,
            h_intent_per_turn=h_intent_per_turn,
            intent_mask=batch.intent_turn_mask,
            prototypes=self.prototypes,
            culture_ids=batch.culture_ids,
            lam=self.lam,
            tau=self.tau,
        )

        # 4) Pooled intent over valid turns (for Eq. 6 downstream).
        turn_mask = batch.intent_turn_mask.unsqueeze(-1).to(h_intent_per_turn.dtype)
        denom = turn_mask.sum(dim=1).clamp(min=1.0)
        h_intent_pooled = (h_intent_per_turn * turn_mask).sum(dim=1) / denom

        out = dict(losses)
        out["p_pred"] = p_pred
        out["h_intent_pooled"] = h_intent_pooled
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_distribution(
        self, ctx_ids: torch.Tensor, ctx_mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.context_encoder(ctx_ids, ctx_mask)
        return self.value_head(h)
