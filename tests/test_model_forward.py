"""End-to-end forward + backward smoke test for ProCAModel."""
from __future__ import annotations

import torch

from proca.data import make_batch


def test_proca_forward_backward(proca_model, synthetic_dialogues, wvs_df, tokenizer, cfg):
    batch = make_batch(
        synthetic_dialogues,
        wvs_df=wvs_df,
        culture_to_id=proca_model.culture_to_id,
        tokenizer=tokenizer,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
        n_intent_turns=3,
        seed=0,
    )
    out = proca_model(batch)
    assert "total" in out and "kl" in out and "cont" in out
    assert torch.isfinite(out["total"])
    assert not torch.isnan(out["total"])
    # Confirm a gradient flows through every learnable parameter group
    out["total"].backward()
    has_grad = [p.grad is not None for p in proca_model.parameters() if p.requires_grad]
    assert any(has_grad)


def test_proca_p_pred_is_distribution(proca_model, synthetic_dialogues, wvs_df, tokenizer, cfg):
    batch = make_batch(
        synthetic_dialogues, wvs_df=wvs_df,
        culture_to_id=proca_model.culture_to_id,
        tokenizer=tokenizer,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
        n_intent_turns=3, seed=1,
    )
    out = proca_model(batch)
    p = out["p_pred"]
    assert torch.allclose(p.sum(dim=-1), torch.ones(p.shape[0]), atol=1e-4)
    assert (p >= 0).all()
