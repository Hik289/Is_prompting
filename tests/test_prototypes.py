"""Tests for the cultural prototype space."""
from __future__ import annotations

import torch

from proca.prototypes import PrototypeBank, build_prototypes, random_prototypes


def test_prototype_shape(prototypes, cfg):
    assert prototypes.matrix.shape == (
        len(cfg["cultures"]),
        int(cfg["synthesis"]["prototype_dim"]),
    )


def test_prototype_cultures_align(prototypes, cfg):
    assert prototypes.cultures == cfg["cultures"]


def test_prototype_save_load(tmp_path, prototypes):
    p = tmp_path / "proto.pt"
    prototypes.save(p)
    loaded = PrototypeBank.load(p)
    assert loaded.cultures == prototypes.cultures
    assert torch.allclose(loaded.matrix, prototypes.matrix)


def test_random_prototypes_fallback():
    bank = random_prototypes(["A", "B", "C"], d_c=16, seed=0)
    assert bank.matrix.shape == (3, 16)
