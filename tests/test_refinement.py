"""Tests for iterative refinement (Eq. 6)."""
from __future__ import annotations

from proca.refinement import (
    importance_weighted_sample, score_dataset, top_k_filter,
)


def test_score_dataset_yields_scores(proca_model, synthetic_dialogues, wvs_df, tokenizer, cfg):
    scored = score_dataset(
        proca_model, synthetic_dialogues, wvs_df, tokenizer,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
        n_intent_turns=3,
    )
    assert len(scored) == len(synthetic_dialogues)
    for s in scored:
        assert s.score == s.score # not NaN


def test_top_k_filter_ratio(proca_model, synthetic_dialogues, wvs_df, tokenizer, cfg):
    scored = score_dataset(
        proca_model, synthetic_dialogues, wvs_df, tokenizer,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
        n_intent_turns=3,
    )
    kept = top_k_filter(scored, top_k_ratio=0.5)
    assert len(kept) == max(1, round(len(scored) * 0.5))


def test_importance_sample_runs(proca_model, synthetic_dialogues, wvs_df, tokenizer, cfg):
    scored = score_dataset(
        proca_model, synthetic_dialogues, wvs_df, tokenizer,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
        n_intent_turns=3,
    )
    out = importance_weighted_sample(scored, n=4, seed=0)
    assert len(out) == 4
