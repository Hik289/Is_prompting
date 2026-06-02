"""Loss functions for ProCA.

Equations:
  Eq. 3 L_KL = D_KL( p_wvs || p_pred )
  Eq. 4 L_cont^(i) = -log( exp(s(h_i, c_{k_i}) / τ) / Σ_{j≠i} exp(s(h_i, c_{k_j}) / τ) )
  Eq. 5 L_total = (1/|B|) Σ_{(X,q) ∈ B} [ L_KL + λ · (1/T') Σ_t L_cont^(t) ]
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Eq. 3 — KL divergence between WVS ground-truth distribution and prediction.
# ---------------------------------------------------------------------------
def kl_divergence_loss(
    p_pred: torch.Tensor,
    p_wvs: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """L_KL (Eq. 3).

    Parameters
    ----------
    p_pred : (B, C) predicted probability distribution from f_val.
    p_wvs : (B, C) ground-truth WVS distribution.
    """
    if p_pred.shape != p_wvs.shape:
        raise ValueError(f"p_pred {tuple(p_pred.shape)} != p_wvs {tuple(p_wvs.shape)}")
    p_pred = p_pred.clamp(min=eps)
    p_wvs = p_wvs.clamp(min=eps)
    # D_KL(p_wvs || p_pred) = Σ p_wvs * (log p_wvs - log p_pred)
    per_sample = (p_wvs * (p_wvs.log() - p_pred.log())).sum(dim=-1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample


# ---------------------------------------------------------------------------
# Eq. 4 — contrastive intent-prototype alignment.
# ---------------------------------------------------------------------------
def contrastive_prototype_loss(
    h_intent: torch.Tensor,
    prototypes: torch.Tensor,
    culture_ids: torch.Tensor,
    tau: float = 0.07,
    reduction: str = "mean",
) -> torch.Tensor:
    """L_cont (Eq. 4).

    Parameters
    ----------
    h_intent : (B, d_c) pooled intent representations.
    prototypes : (|K|, d_c) fixed cultural prototype matrix C.
    culture_ids : (B,) long tensor indexing the prototype row for each sample.
    tau : temperature τ.

    Returns the InfoNCE-style loss as defined in Eq. 4: positive is the
    sample's own prototype c_{k_i}; the denominator sums over all OTHER
    samples' prototypes c_{k_j}, j≠i, in the batch.
    """
    if h_intent.dim() != 2:
        raise ValueError("h_intent must be (B, d_c).")
    if prototypes.dim() != 2:
        raise ValueError("prototypes must be (|K|, d_c).")
    if culture_ids.dim() != 1 or culture_ids.shape[0] != h_intent.shape[0]:
        raise ValueError("culture_ids must be (B,) and align with h_intent batch.")

    B = h_intent.shape[0]
    if B < 2:
        # Eq. 4 needs at least one negative (some other sample j≠i in the batch).
        return h_intent.new_zeros(())

    h_norm = F.normalize(h_intent, dim=-1)
    c_norm = F.normalize(prototypes, dim=-1)

    # Per-sample prototype (positive) — c_{k_i}
    pos_proto = c_norm[culture_ids] # (B, d_c)
    pos_sim = (h_norm * pos_proto).sum(dim=-1) / tau # (B,)

    # Negatives: every OTHER sample's prototype c_{k_j}, j≠i in the batch.
    # Build (B, B) similarity to the prototypes of each batch member.
    neg_sim = h_norm @ pos_proto.T / tau # (B, B)
    diag_mask = torch.eye(B, dtype=torch.bool, device=h_intent.device)
    neg_sim = neg_sim.masked_fill(diag_mask, float("-inf"))

    log_denom = torch.logsumexp(neg_sim, dim=-1) # (B,)
    per_sample = -(pos_sim - log_denom)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample


# ---------------------------------------------------------------------------
# Eq. 5 — unified objective.
# ---------------------------------------------------------------------------
def unified_loss(
    p_pred: torch.Tensor,
    p_wvs: torch.Tensor,
    h_intent_per_turn: torch.Tensor, # (B, T', d_c)
    intent_mask: Optional[torch.Tensor], # (B, T') or None
    prototypes: torch.Tensor, # (|K|, d_c)
    culture_ids: torch.Tensor, # (B,)
    lam: float = 0.5,
    tau: float = 0.07,
) -> dict:
    """L_total (Eq. 5).

    Returns a dict with the total loss and component breakdowns so callers can
    log per-component values during training.
    """
    l_kl = kl_divergence_loss(p_pred, p_wvs)

    B, Tp, d = h_intent_per_turn.shape
    flat_h = h_intent_per_turn.reshape(B * Tp, d)
    flat_culture = culture_ids.unsqueeze(1).expand(B, Tp).reshape(B * Tp)
    if intent_mask is not None:
        flat_mask = intent_mask.reshape(B * Tp).bool()
        flat_h = flat_h[flat_mask]
        flat_culture = flat_culture[flat_mask]
    l_cont = contrastive_prototype_loss(
        flat_h, prototypes, flat_culture, tau=tau, reduction="mean"
    )
    total = l_kl + lam * l_cont
    # Note: keep kl and cont as live (grad-tracking) tensors so callers that
    # use them for ablation back-prop
    # still get a working gradient. Detach is the caller's responsibility for
    # logging (use `.detach().item()`).
    return {"total": total, "kl": l_kl, "cont": l_cont}
