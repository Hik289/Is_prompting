"""reasoning_only ablation Table 2 (GSM8K / MathChat).

Replaces the synthetic dialogue corpus with a reasoning-only dataset
(GSM8K or MathChat) to test whether cultural alignment comes from the
*social interaction* signal or just any additional fine-tuning.

For CODE_REPRO we keep this CPU-friendly: we wrap GSM8K-like prompts into
the same SyntheticDialogue schema with neutral (non-cultural) intent
annotations. A real loader can later swap in the actual datasets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import torch
from torch.optim import AdamW

from .. import utils
from ..data import MockTokenizer, load_mock_wvs, load_wvs_csv, make_batch
from ..encoders import build_encoders
from ..model import ProCAModel
from ..prototypes import build_prototypes
from ..synthesis import DialogueTurn, SyntheticDialogue
from ..train import _train_one_round


def _build_reasoning_corpus(
    cultures: Sequence[str],
    dataset: str = "gsm8k_mock",
    n_per_culture: int = 4,
) -> List[SyntheticDialogue]:
    """Tiny reasoning-style 'dialogues' that share schema with TGSIS dialogues."""
    out: List[SyntheticDialogue] = []
    next_id = 0
    for k in cultures:
        for i in range(n_per_culture):
            turns = [
                DialogueTurn(speaker="User", text=f"({dataset}) If 7 * (3+5) = ?", intent="solve a math problem"),
                DialogueTurn(speaker="Assistant", text="Compute 3+5=8, then 7*8=56. Answer: 56.", intent="explain reasoning"),
            ]
            out.append(SyntheticDialogue(
                dialogue_id=f"reason_{dataset}_{next_id:06d}",
                culture=k, scenario_id=f"{dataset}_{i}", turns=turns,
            ))
            next_id += 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/proca_default.yaml")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset", default="gsm8k_mock", choices=["gsm8k_mock", "mathchat_mock"])
    args = p.parse_args(argv)

    cfg = utils.load_yaml(args.config)
    utils.set_seed(int(cfg["training"]["seed"]))

    wvs_df = load_mock_wvs(cultures=cfg["cultures"])
    prototypes = build_prototypes(wvs_df, cultures=cfg["cultures"], n_components=int(cfg["synthesis"]["prototype_dim"]))
    enc = build_encoders(cfg, prototype_dim=prototypes.d_c, mock=True)
    model = ProCAModel(
        encoders=enc,
        prototypes=prototypes,
        lam=float(cfg["ucca"]["contrastive"]["lambda"]),
        tau=float(cfg["ucca"]["contrastive"]["tau"]),
    )
    tokenizer = MockTokenizer(vocab_size=1000, max_length=64 if args.dry_run else 128)

    corpus = _build_reasoning_corpus(
        cultures=cfg["cultures"],
        dataset=args.dataset,
        n_per_culture=2 if args.dry_run else 4,
    )
    max_steps = int(cfg["dryrun"]["n_steps"]) if args.dry_run else None
    log = _train_one_round(
        model, corpus, wvs_df, tokenizer, cfg,
        max_steps=max_steps,
        ablation="none", # uses full L_total but on reasoning data instead of dialogues
    )
    print(f"[reasoning_only] dataset={args.dataset} n_steps={len(log)}")
    if log:
        print(f"[reasoning_only] last loss: {log[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
