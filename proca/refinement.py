"""Iterative refinement with adaptive filtering.

Eq. 6:
    w_i = conf(p_pred,i) · sim( E_intent(X_i^{(k)}), c_k )

where `conf` is taken as *negative entropy* (lower entropy ⇒ higher confidence)
following the paper's note; we expose this via the `confidence_metric` flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import DialogueBatch, MockTokenizer, make_batch
from .model import ProCAModel
from .synthesis import SyntheticDialogue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _confidence(p_pred: torch.Tensor, metric: str = "neg_entropy") -> torch.Tensor:
    """Map predicted distribution to a confidence scalar (Eq. 6 first factor).

    metrics:
      - 'neg_entropy': -H(p) / log(C) (in [-1, 0]); higher = more confident.
                        We re-scale to [0, 1] by adding 1 so all weights ≥ 0.
      - 'max_prob' : max_c p_c
    """
    if metric == "max_prob":
        return p_pred.max(dim=-1).values
    if metric == "neg_entropy":
        C = p_pred.shape[-1]
        ent = -(p_pred.clamp(min=1e-8) * p_pred.clamp(min=1e-8).log()).sum(dim=-1)
        norm = ent / max(np.log(C), 1e-8) # in [0,1]
        conf = (1.0 - norm) # in [0,1], 1=peaked
        return conf
    raise ValueError(f"Unknown confidence_metric: {metric!r}")


def _intent_prototype_sim(
    h_intent: torch.Tensor, # (B, d_c)
    prototypes: torch.Tensor, # (|K|, d_c)
    culture_ids: torch.Tensor, # (B,)
) -> torch.Tensor:
    h = F.normalize(h_intent, dim=-1)
    c = F.normalize(prototypes, dim=-1)
    proto = c[culture_ids]
    return (h * proto).sum(dim=-1) # cosine similarity in [-1, 1]


# ---------------------------------------------------------------------------
# Scoring + selection
# ---------------------------------------------------------------------------
@dataclass
class ScoredDialogue:
    dialogue: SyntheticDialogue
    score: float


@torch.no_grad()
def score_dataset(
    model: ProCAModel,
    dialogues: Sequence[SyntheticDialogue],
    wvs_df,
    tokenizer: MockTokenizer,
    n_choices: int,
    batch_size: int = 8,
    n_intent_turns: int = 4,
    confidence_metric: str = "neg_entropy",
) -> List[ScoredDialogue]:
    """Compute Eq. 6 score for every dialogue in `dialogues`."""
    model.eval()
    out: List[ScoredDialogue] = []
    cul2id = model.culture_to_id
    for start in range(0, len(dialogues), batch_size):
        chunk = list(dialogues[start : start + batch_size])
        batch = make_batch(
            chunk,
            wvs_df=wvs_df,
            culture_to_id=cul2id,
            tokenizer=tokenizer,
            n_choices=n_choices,
            n_intent_turns=n_intent_turns,
            seed=start,
        )
        fwd = model(batch)
        conf = _confidence(fwd["p_pred"], metric=confidence_metric) # (B,)
        sim = _intent_prototype_sim(
            fwd["h_intent_pooled"], model.prototypes, batch.culture_ids
        )
        # Clamp similarity to [0,1] (it is cosine; we want non-negative weight).
        sim_clamped = (sim + 1.0) * 0.5
        w = (conf * sim_clamped).cpu().tolist()
        for d, s in zip(chunk, w):
            out.append(ScoredDialogue(dialogue=d, score=float(s)))
    return out


def top_k_filter(
    scored: Sequence[ScoredDialogue], top_k_ratio: float = 0.70
) -> List[SyntheticDialogue]:
    """Keep top-K by score."""
    if not scored:
        return []
    sorted_s = sorted(scored, key=lambda x: x.score, reverse=True)
    k = max(1, int(round(len(sorted_s) * top_k_ratio)))
    return [s.dialogue for s in sorted_s[:k]]


def importance_weighted_sample(
    scored: Sequence[ScoredDialogue],
    n: int,
    seed: int = 0,
) -> List[SyntheticDialogue]:
    """Sample `n` dialogues with replacement using w_i as probabilities."""
    if not scored:
        return []
    rng = np.random.default_rng(seed)
    weights = np.array([max(s.score, 0.0) for s in scored], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    idx = rng.choice(len(scored), size=n, replace=True, p=probs)
    return [scored[int(i)].dialogue for i in idx]
