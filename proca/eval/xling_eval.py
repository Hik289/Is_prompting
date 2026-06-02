"""Cross-lingual evaluation.

We translate WVS questions and persona prompts into the primary local
language of each culture (e.g., zh for China). Translation is abstracted as a
:class:`TranslationProvider`; the default :class:`MockTranslator` simply
prepends a language tag so the pipeline can run without GPT-4. A real
:class:`GPT4Translator` stub is provided as a plug-in point.

Usage:
    python -m proca.eval.xling_eval --culture China --mock
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .. import utils
from ..data import MockTokenizer, load_mock_wvs, load_wvs_csv
from ..prototypes import build_prototypes
from .baselines import (
    BaselineModel, CulturalPromptingBaseline, PersonaStandardBaseline,
    sample_personas,
)
from .wvs_eval import (
    _build_eval_model, _wvs_distribution_for_question,
    _empirical_distribution_from_predictions,
    kl_divergence, jensen_shannon_distance,
    CULTURE_ALIASES,
)


LANGUAGE_OF: Dict[str, str] = {
    "CN": "zh", "DE": "de", "UK": "en", "MX": "es", "JP": "ja",
}


# ---------------------------------------------------------------------------
class TranslationProvider:
    name: str = "abstract"

    def translate(self, text: str, target_lang: str) -> str:
        raise NotImplementedError


class MockTranslator(TranslationProvider):
    """Identity-with-tag translator: tags the text with `[lang=xx]`."""
    name = "mock"

    def translate(self, text: str, target_lang: str) -> str:
        return f"[lang={target_lang}] {text}"


class GPT4Translator(TranslationProvider):
    """Stub for the paper's GPT-4 translation (network bound)."""
    name = "gpt4"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def translate(self, text: str, target_lang: str) -> str: # pragma: no cover
        raise NotImplementedError(
            "GPT4Translator.translate(): wire up the OpenAI client here."
        )


def make_translator(backend: str = "mock", **kw) -> TranslationProvider:
    return {"mock": MockTranslator, "gpt4": GPT4Translator}[backend](**kw) if backend == "gpt4" else MockTranslator()


# ---------------------------------------------------------------------------
@dataclass
class XLingResult:
    culture: str
    language: str
    kl_divergence: float
    jensen_shannon: float
    persona_accuracy: float


def evaluate_xling(
    baseline: BaselineModel,
    wvs_df,
    culture: str,
    translator: TranslationProvider,
    n_personas: int = 200,
    n_choices: int = 5,
    q_cols: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> XLingResult:
    culture = CULTURE_ALIASES.get(culture, culture)
    target_lang = LANGUAGE_OF.get(culture, "en")
    if q_cols is None:
        q_cols = [c for c in wvs_df.columns if c.startswith("Q")]

    personas = sample_personas(culture, n_personas, seed=seed)
    sub = wvs_df[wvs_df["culture"] == culture].reset_index(drop=True)

    kls: List[float] = []
    jss: List[float] = []
    correct, total = 0, 0
    for q in q_cols:
        q_text_native = translator.translate(f"WVS question {q}", target_lang)
        per_persona = []
        for persona in personas:
            try:
                pred = baseline.predict(
                    question_text=q_text_native,
                    n_choices=n_choices,
                    persona=persona,
                    culture=culture,
                )
                per_persona.append(
                    pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
                )
            except Exception:
                per_persona.append(np.zeros(n_choices))
        pred_dist = _empirical_distribution_from_predictions(per_persona, n_choices)
        gold_dist = _wvs_distribution_for_question(wvs_df, culture, q, n_choices)
        kls.append(kl_divergence(gold_dist, pred_dist))
        jss.append(jensen_shannon_distance(gold_dist, pred_dist))

        for i, persona in enumerate(personas):
            if i >= len(sub):
                break
            pred_choice = int(np.argmax(per_persona[i][:n_choices])) + 1
            gold_choice = int(sub.iloc[i % len(sub)][q])
            correct += int(pred_choice == gold_choice)
            total += 1

    return XLingResult(
        culture=culture,
        language=target_lang,
        kl_divergence=float(np.mean(kls)) if kls else float("nan"),
        jensen_shannon=float(np.mean(jss)) if jss else float("nan"),
        persona_accuracy=correct / total if total else float("nan"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/proca_default.yaml")
    p.add_argument("--culture", default="China")
    p.add_argument("--n-personas", type=int, default=30)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--translator", default="mock", choices=["mock", "gpt4"])
    args = p.parse_args(argv)

    cfg = utils.load_yaml(args.config)
    utils.set_seed(int(cfg["training"]["seed"]))

    wvs_df = load_mock_wvs(cultures=cfg["cultures"]) if args.mock else load_wvs_csv(cfg["paths"]["wvs_csv"])
    prototypes = build_prototypes(wvs_df, cultures=cfg["cultures"], n_components=int(cfg["synthesis"]["prototype_dim"]))
    model = _build_eval_model(cfg, mock=args.mock, prototypes=prototypes)
    if args.ckpt and Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    tokenizer = MockTokenizer(vocab_size=1000)
    baseline = PersonaStandardBaseline(model, tokenizer)
    translator = make_translator(args.translator)

    res = evaluate_xling(
        baseline=baseline,
        wvs_df=wvs_df,
        culture=args.culture,
        translator=translator,
        n_personas=args.n_personas,
        n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
    )
    print(f"== X-Lingual Eval: {res.culture} ({res.language}) ==")
    print(f" KL-D : {res.kl_divergence:.4f}")
    print(f" Jensen-Shannon : {res.jensen_shannon:.4f}")
    print(f" Persona accuracy: {res.persona_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
