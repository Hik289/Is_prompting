"""WVS evaluation.

Computes:
  - Culture-Level Alignment via KL-D (primary, lower is better).
  - Jensen-Shannon distance (secondary,
  - Persona-Level Accuracy.

Invalid / safeguarded outputs are mapped to a dedicated `invalid` slot so that
refusals don't artificially inflate alignment.

Usage:
    python -m proca.eval.wvs_eval --culture China --mock
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import utils
from ..data import MockTokenizer, load_mock_wvs, load_wvs_csv
from ..encoders import build_encoders
from ..model import ProCAModel
from ..prototypes import build_prototypes
from .baselines import (
    BaselineModel,
    CulturalPromptingBaseline,
    PersonaStandardBaseline,
    Persona,
    sample_personas,
)


CULTURE_ALIASES = {
    "China": "CN", "CN": "CN", "Germany": "DE", "DE": "DE",
    "UK": "UK", "United Kingdom": "UK",
    "Mexico": "MX", "MX": "MX",
    "Japan": "JP", "JP": "JP",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """D_KL(p || q)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """JS distance = sqrt(JS divergence), in [0, log2(2)/2]."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (np.log(p) - np.log(m))).sum() + 0.5 * (q * (np.log(q) - np.log(m))).sum()
    return float(np.sqrt(max(js, 0.0)))


def safeguard_distribution(
    p: torch.Tensor | np.ndarray,
    invalid_prob: float = 0.0,
) -> np.ndarray:
    """Append an `invalid` slot to a length-C distribution."""
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    p = np.array(p, dtype=np.float64)
    # Re-scale valid mass to (1 - invalid_prob), append invalid slot.
    if p.sum() <= 0:
        p = np.ones_like(p) / len(p)
    p = p / p.sum() * (1.0 - invalid_prob)
    return np.concatenate([p, [invalid_prob]])


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------
@dataclass
class WVSEvalResult:
    culture: str
    model_name: str
    kl_divergence: float
    jensen_shannon: float
    persona_accuracy: float
    n_personas: int
    n_questions: int
    per_question_kl: Dict[str, float]


def _empirical_distribution_from_predictions(
    per_persona_preds: List[np.ndarray],
    n_choices: int,
) -> np.ndarray:
    """Aggregate persona-level argmax predictions into an empirical distribution."""
    counts = np.zeros(n_choices + 1, dtype=np.float64) # +1 = invalid slot
    for p in per_persona_preds:
        if p is None or len(p) == 0 or not np.isfinite(p).all() or p.sum() <= 0:
            counts[-1] += 1.0
            continue
        choice = int(np.argmax(p[:n_choices]))
        counts[choice] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


def _wvs_distribution_for_question(
    wvs_df, culture: str, q_col: str, n_choices: int
) -> np.ndarray:
    sub = wvs_df[wvs_df["culture"] == culture]
    counts = np.zeros(n_choices + 1, dtype=np.float64)
    if len(sub) == 0:
        counts[:n_choices] = 1.0 / n_choices
        return counts / counts.sum()
    for c in range(1, n_choices + 1):
        counts[c - 1] = (sub[q_col] == c).sum()
    counts[-1] = 0.0 # WVS gold rarely contains explicit invalid
    return counts / counts.sum() if counts.sum() > 0 else counts


def evaluate(
    baseline: BaselineModel,
    wvs_df,
    culture: str,
    n_personas: int = 1000,
    n_choices: int = 5,
    q_cols: Optional[Sequence[str]] = None,
    seed: int = 0,
    invalid_safeguard: bool = True,
) -> WVSEvalResult:
    culture = CULTURE_ALIASES.get(culture, culture)
    if q_cols is None:
        q_cols = [c for c in wvs_df.columns if c.startswith("Q")]

    personas = sample_personas(culture=culture, n=n_personas, seed=seed)
    sub = wvs_df[wvs_df["culture"] == culture].reset_index(drop=True)

    kls: Dict[str, float] = {}
    jss: List[float] = []
    correct = 0
    total = 0
    for q in q_cols:
        per_persona = []
        for p_idx, persona in enumerate(personas):
            try:
                pred = baseline.predict(
                    question_text=f"WVS question {q}",
                    n_choices=n_choices,
                    persona=persona,
                    culture=culture,
                )
                per_persona.append(pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred))
            except Exception:
                per_persona.append(np.zeros(n_choices)) # invalid

        pred_dist = _empirical_distribution_from_predictions(per_persona, n_choices)
        gold_dist = _wvs_distribution_for_question(wvs_df, culture, q, n_choices)
        if not invalid_safeguard:
            pred_dist = pred_dist[:n_choices] / max(pred_dist[:n_choices].sum(), 1e-8)
            gold_dist = gold_dist[:n_choices] / max(gold_dist[:n_choices].sum(), 1e-8)
        kl = kl_divergence(gold_dist, pred_dist)
        js = jensen_shannon_distance(gold_dist, pred_dist)
        kls[q] = kl
        jss.append(js)

        # Persona-level accuracy: per-persona argmax vs. matched WVS respondent.
        for i, persona in enumerate(personas):
            if i >= len(sub):
                break
            pred_choice = int(np.argmax(per_persona[i][:n_choices])) + 1
            gold_choice = int(sub.iloc[i % len(sub)][q])
            correct += int(pred_choice == gold_choice)
            total += 1

    avg_kl = float(np.mean(list(kls.values()))) if kls else float("nan")
    avg_js = float(np.mean(jss)) if jss else float("nan")
    acc = correct / total if total else float("nan")
    return WVSEvalResult(
        culture=culture,
        model_name=baseline.name,
        kl_divergence=avg_kl,
        jensen_shannon=avg_js,
        persona_accuracy=acc,
        n_personas=n_personas,
        n_questions=len(q_cols),
        per_question_kl=kls,
    )


# ---------------------------------------------------------------------------
def _build_eval_model(cfg, mock: bool, prototypes):
    enc = build_encoders(cfg, prototype_dim=prototypes.d_c, mock=mock)
    return ProCAModel(
        encoders=enc,
        prototypes=prototypes,
        lam=float(cfg["ucca"]["contrastive"]["lambda"]),
        tau=float(cfg["ucca"]["contrastive"]["tau"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/proca_default.yaml")
    p.add_argument("--culture", default="China")
    p.add_argument("--n-personas", type=int, default=50,
                   help="Override paper's 1000 for smoke-test; paper default 1000.")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--ckpt", default=None, help="Path to a ProCA checkpoint.")
    p.add_argument("--baseline", default="standard",
                   choices=["standard", "cultural"])
    args = p.parse_args(argv)

    cfg = utils.load_yaml(args.config)
    utils.set_seed(int(cfg["training"]["seed"]))

    wvs_df = load_mock_wvs(cultures=cfg["cultures"]) if args.mock else load_wvs_csv(cfg["paths"]["wvs_csv"])
    prototypes = build_prototypes(wvs_df, cultures=cfg["cultures"], n_components=int(cfg["synthesis"]["prototype_dim"]))
    model = _build_eval_model(cfg, mock=args.mock, prototypes=prototypes)
    if args.ckpt and Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    tokenizer = MockTokenizer(vocab_size=1000)

    baseline: BaselineModel
    if args.baseline == "standard":
        baseline = PersonaStandardBaseline(model, tokenizer)
    else:
        baseline = CulturalPromptingBaseline(model, tokenizer)

    res = evaluate(
        baseline=baseline,
        wvs_df=wvs_df,
        culture=args.culture,
        n_personas=args.n_personas,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
        invalid_safeguard=bool(cfg["evaluation"]["invalid_option_safeguard"]),
    )
    print(f"== WVS Evaluation: {res.model_name} on {res.culture} ==")
    print(f" KL-D : {res.kl_divergence:.4f}")
    print(f" Jensen-Shannon : {res.jensen_shannon:.4f}")
    print(f" Persona accuracy: {res.persona_accuracy:.4f}")
    print(f" n_personas={res.n_personas} n_questions={res.n_questions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
