"""Tests for the encoder stack and value head."""
from __future__ import annotations

import torch

from proca.encoders import build_encoders


def test_encoders_forward_shapes(cfg, prototypes):
    enc = build_encoders(cfg, prototype_dim=prototypes.d_c, mock=True)
    B, L = 4, 16
    ids = torch.randint(3, 999, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    h_ctx = enc.context_encoder(ids, mask)
    assert h_ctx.shape == (B, enc.hidden_size)
    p = enc.value_head(h_ctx)
    assert p.shape == (B, enc.value_head.num_choices)
    assert torch.allclose(p.sum(dim=-1), torch.ones(B), atol=1e-5)
    h_int = enc.intent_encoder(ids, mask)
    assert h_int.shape == (B, enc.prototype_dim)
