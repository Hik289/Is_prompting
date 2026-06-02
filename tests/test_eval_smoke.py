"""Smoke tests for the WVS + cross-lingual evaluators."""
from __future__ import annotations

from proca.data import load_mock_wvs, MockTokenizer
from proca.eval.baselines import (
    CulturalPromptingBaseline, PersonaStandardBaseline,
)
from proca.eval.wvs_eval import evaluate, kl_divergence, jensen_shannon_distance
from proca.eval.xling_eval import evaluate_xling, MockTranslator
import numpy as np


def test_metrics_sanity():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) == 0.0
    assert jensen_shannon_distance(p, q) == 0.0
    p2 = np.array([0.9, 0.1])
    q2 = np.array([0.1, 0.9])
    assert kl_divergence(p2, q2) > 0
    assert jensen_shannon_distance(p2, q2) > 0


def test_evaluate_standard_baseline(proca_model, cfg):
    tok = MockTokenizer(vocab_size=1000)
    wvs_df = load_mock_wvs()
    bl = PersonaStandardBaseline(proca_model, tok)
    res = evaluate(
        bl, wvs_df, culture="CN", n_personas=4,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
    )
    assert res.culture == "CN"
    assert res.n_personas == 4
    assert res.kl_divergence == res.kl_divergence # not nan


def test_evaluate_cultural_baseline(proca_model, cfg):
    tok = MockTokenizer(vocab_size=1000)
    wvs_df = load_mock_wvs()
    bl = CulturalPromptingBaseline(proca_model, tok)
    res = evaluate(bl, wvs_df, culture="DE", n_personas=4,
                   n_choices=int(cfg["ucca"]["value_head"]["num_choices"]))
    assert res.model_name == "CulturalPrompting"


def test_xling_evaluation_runs(proca_model, cfg):
    tok = MockTokenizer(vocab_size=1000)
    wvs_df = load_mock_wvs()
    bl = PersonaStandardBaseline(proca_model, tok)
    res = evaluate_xling(
        bl, wvs_df, culture="JP", translator=MockTranslator(),
        n_personas=3, n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
    )
    assert res.language == "ja"
