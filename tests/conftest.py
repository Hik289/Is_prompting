"""Pytest fixtures: tiny mocked WVS + scenarios usable by all tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from proca import utils
from proca.data import MockTokenizer, load_mock_wvs, write_mock_sotopia
from proca.encoders import build_encoders
from proca.model import ProCAModel
from proca.prototypes import build_prototypes
from proca.synthesis import load_sotopia_scenarios, make_teacher, synthesize_dataset


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "proca_default.yaml"


@pytest.fixture(scope="session")
def cfg():
    return utils.load_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def wvs_df():
    return load_mock_wvs(seed=0)


@pytest.fixture(scope="session")
def prototypes(cfg, wvs_df):
    return build_prototypes(
        wvs_df,
        cultures=cfg["cultures"],
        n_components=int(cfg["synthesis"]["prototype_dim"]),
    )


@pytest.fixture(scope="session")
def scenarios(tmp_path_factory):
    p = tmp_path_factory.mktemp("scen") / "mock_sotopia.json"
    write_mock_sotopia(p, n=6)
    return load_sotopia_scenarios(p)


@pytest.fixture(scope="session")
def synthetic_dialogues(scenarios, prototypes):
    teacher = make_teacher("mock", seed=0)
    return synthesize_dataset(
        scenarios, prototypes, teacher,
        n_per_culture=3, min_turns=4, max_turns=6, seed=0,
    )


@pytest.fixture(scope="session")
def tokenizer():
    return MockTokenizer(vocab_size=1000, max_length=64)


@pytest.fixture(scope="session")
def proca_model(cfg, prototypes):
    enc = build_encoders(cfg, prototype_dim=prototypes.d_c, mock=True)
    model = ProCAModel(
        encoders=enc,
        prototypes=prototypes,
        lam=float(cfg["ucca"]["contrastive"]["lambda"]),
        tau=float(cfg["ucca"]["contrastive"]["tau"]),
    )
    return model
