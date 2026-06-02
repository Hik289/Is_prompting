"""Tests for L_KL (Eq. 3), L_cont (Eq. 4), L_total (Eq. 5)."""
from __future__ import annotations

import math

import torch
import pytest

from proca.losses import (
    contrastive_prototype_loss,
    kl_divergence_loss,
    unified_loss,
)


def test_kl_zero_when_identical():
    p = torch.tensor([[0.2, 0.3, 0.5]])
    loss = kl_divergence_loss(p, p)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_when_different():
    p_pred = torch.tensor([[0.1, 0.1, 0.8]])
    p_wvs = torch.tensor([[0.5, 0.3, 0.2]])
    assert kl_divergence_loss(p_pred, p_wvs).item() > 0


def test_kl_shape_mismatch_raises():
    with pytest.raises(ValueError):
        kl_divergence_loss(torch.zeros(2, 3), torch.zeros(2, 4))


def test_contrastive_loss_finite_and_low_for_aligned():
    d_c = 16
    n_k = 4
    g = torch.Generator().manual_seed(0)
    protos = torch.randn(n_k, d_c, generator=g)
    # Aligned case: h_i directly equals its prototype.
    culture_ids = torch.tensor([0, 1, 2, 3])
    h = protos.clone()
    loss = contrastive_prototype_loss(h, protos, culture_ids, tau=0.1)
    assert torch.isfinite(loss)
    # Misaligned case: h is random.
    h_bad = torch.randn(n_k, d_c, generator=g)
    loss_bad = contrastive_prototype_loss(h_bad, protos, culture_ids, tau=0.1)
    assert loss.item() < loss_bad.item()


def test_contrastive_loss_single_sample_safe():
    protos = torch.randn(3, 8)
    h = torch.randn(1, 8)
    ids = torch.tensor([0])
    loss = contrastive_prototype_loss(h, protos, ids)
    assert torch.isfinite(loss)


def test_unified_loss_combines_components():
    g = torch.Generator().manual_seed(0)
    B, C, Tp, d_c = 4, 5, 3, 8
    # Make leaf tensors require grad so the loss is differentiable end-to-end.
    p_pred_logits = torch.randn(B, C, generator=g, requires_grad=True)
    p_pred = torch.softmax(p_pred_logits, dim=-1)
    p_wvs = torch.softmax(torch.randn(B, C, generator=g), dim=-1)
    h_int = torch.randn(B, Tp, d_c, generator=g, requires_grad=True)
    mask = torch.ones(B, Tp, dtype=torch.long)
    protos = torch.randn(3, d_c, generator=g)
    cul = torch.tensor([0, 1, 2, 0])
    out = unified_loss(p_pred, p_wvs, h_int, mask, protos, cul, lam=0.5, tau=0.07)
    assert set(out) == {"total", "kl", "cont"}
    assert torch.isfinite(out["total"])
    assert out["total"].grad_fn is not None
    out["total"].backward()
    assert p_pred_logits.grad is not None and h_int.grad is not None
