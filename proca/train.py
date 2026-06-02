"""ProCA training loop — Algorithm 1.

Implements the iterative loop:
  Init prototypes C from WVS
  for r in [0..R-1]:
    if r == 0: generate D_syn^(0) via Eq.(1)
    else: generate candidate D_cand^(r) via Eq.(1), score with Eq.(6),
               take top-K → D_syn^(r)
    train M^(r) on D_syn^(r) with L_total (Eq. 5)
  return M^(R-1)

Run dry-run (CPU, mocked weights & data):
    python -m proca.train --config configs/proca_default.yaml --mock --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from . import utils
from .data import MockTokenizer, load_mock_wvs, load_wvs_csv, make_batch, write_mock_sotopia
from .encoders import build_encoders
from .model import ProCAModel
from .prototypes import PrototypeBank, build_prototypes, random_prototypes
from .refinement import score_dataset, top_k_filter, importance_weighted_sample
from .synthesis import (
    SyntheticDialogue,
    load_sotopia_scenarios,
    make_teacher,
    save_dialogues,
    synthesize_dataset,
)


# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ProCA training driver (Algorithm 1).")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--mock", action="store_true", help="Use MockTinyTransformer + mock data.")
    p.add_argument("--dry-run", action="store_true", help="Run 2 steps only, print losses, exit.")
    p.add_argument("--ablation", default=None,
                   help="One of: none|dialogue_only|intent_only|reasoning_only|teacher_swap")
    p.add_argument("--teacher", default=None,
                   help="Override teacher model (e.g. gpt_oss_120b, qwen3_32b).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
def _build_prototypes(cfg: Dict[str, Any], mock: bool) -> PrototypeBank:
    cultures = list(cfg["cultures"])
    d_c = int(cfg["synthesis"]["prototype_dim"])
    if mock or not Path(cfg["paths"]["wvs_csv"]).exists():
        # mock_wvs has known cultures; produce PCA-based prototypes from it.
        wvs_df = load_mock_wvs(cultures=cultures)
        return build_prototypes(wvs_df, cultures=cultures, n_components=d_c)
    wvs_df = load_wvs_csv(cfg["paths"]["wvs_csv"])
    return build_prototypes(wvs_df, cultures=cultures, n_components=d_c)


def _build_wvs(cfg: Dict[str, Any], mock: bool):
    if mock or not Path(cfg["paths"]["wvs_csv"]).exists():
        return load_mock_wvs(cultures=cfg["cultures"])
    return load_wvs_csv(cfg["paths"]["wvs_csv"])


def _build_scenarios(cfg: Dict[str, Any], mock: bool):
    p = Path(cfg["paths"]["sotopia_json"])
    if mock or not p.exists():
        write_mock_sotopia(p, n=12)
    return load_sotopia_scenarios(p)


# ---------------------------------------------------------------------------
def _train_one_round(
    model: ProCAModel,
    dialogues: List[SyntheticDialogue],
    wvs_df,
    tokenizer: MockTokenizer,
    cfg: Dict[str, Any],
    *,
    max_steps: int | None = None,
    log_every: int = 1,
    ablation: str = "none",
) -> List[Dict[str, float]]:
    """Train `model` on `dialogues` for one round; returns per-step log."""
    bs = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    lr = float(cfg["training"]["lr"])
    n_choices = int(cfg["ucca"]["value_head"]["num_choices"])
    n_intent_turns = 4 # matches T' in unified_loss default

    opt = AdamW(model.parameters(), lr=lr)
    model.train()
    log: List[Dict[str, float]] = []
    step = 0
    for epoch in range(epochs):
        # naive shuffling by simple slice rotation (deterministic w/ seed).
        for start in range(0, len(dialogues), bs):
            batch_items = dialogues[start : start + bs]
            if len(batch_items) < 2:
                # contrastive needs ≥2 samples in a batch
                continue
            batch = make_batch(
                batch_items,
                wvs_df=wvs_df,
                culture_to_id=model.culture_to_id,
                tokenizer=tokenizer,
                n_choices=n_choices,
                n_intent_turns=n_intent_turns,
                seed=step,
            )
            out = model(batch)

            # --- ablation gating ---
            if ablation == "dialogue_only":
                # turn off L_cont
                loss = out["kl"]
            elif ablation == "intent_only":
                # turn off L_KL + dialogue grounding
                loss = out["cont"]
            else:
                loss = out["total"]

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            entry = {
                "step": float(step),
                "epoch": float(epoch),
                "loss_total": float(out["total"].detach().item()),
                "loss_kl": float(out["kl"].detach().item()),
                "loss_cont": float(out["cont"].detach().item()),
                "lr": lr,
            }
            log.append(entry)
            if log_every and step % log_every == 0:
                print(
                    f"[train] epoch={epoch} step={step} "
                    f"total={entry['loss_total']:.4f} "
                    f"kl={entry['loss_kl']:.4f} "
                    f"cont={entry['loss_cont']:.4f}"
                )
            if max_steps is not None and step >= max_steps:
                return log
    return log


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = utils.load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg["training"]["seed"])
    utils.set_seed(seed)

    out_dir = Path(args.output_dir or cfg["paths"]["results_dir"])
    utils.ensure_dir(out_dir)
    utils.ensure_dir(cfg["paths"]["syn_data_dir"])
    utils.ensure_dir(cfg["paths"]["ckpt_dir"])

    # ------------- Prototypes (fixed anchors, -----------------
    prototypes = _build_prototypes(cfg, mock=args.mock)
    prototypes.save(cfg["paths"]["prototypes_out"])

    wvs_df = _build_wvs(cfg, mock=args.mock)
    scenarios = _build_scenarios(cfg, mock=args.mock)
    tokenizer = MockTokenizer(vocab_size=1000, max_length=64 if args.dry_run else 128)

    # ------------- Teacher G (Eq. 1) --------------------------------------
    teacher_backend = (cfg["synthesis"].get("teacher_backend") or "mock").lower()
    teacher_name = args.teacher or cfg["synthesis"]["teacher_model"]
    if args.mock:
        teacher_backend = "mock"
    teacher = make_teacher(teacher_backend, seed=seed)

    # ------------- Encoders + model (LoRA hooked) -------------------------
    enc = build_encoders(cfg, prototype_dim=prototypes.d_c, mock=args.mock)
    lora_cfg = dict(cfg.get("lora", {}))
    lora_cfg["attach"] = not args.mock # only attach PEFT on real backbones
    model = ProCAModel(
        encoders=enc,
        prototypes=prototypes,
        lam=float(cfg["ucca"]["contrastive"]["lambda"]),
        tau=float(cfg["ucca"]["contrastive"]["tau"]),
        lora_cfg=lora_cfg,
    )

    # ------------- Algorithm 1 main loop ----------------------------------
    R = int(cfg["refinement"]["R"])
    top_k_ratio = float(cfg["refinement"]["top_k_ratio"])
    use_iw = bool(cfg["refinement"]["use_importance_sampling"])
    conf_metric = cfg["refinement"]["confidence_metric"]

    n_per_culture = (
        2 if args.dry_run else int(cfg["synthesis"]["n_dialogues_per_culture"])
    )
    min_turns = int(cfg["synthesis"]["dialogue_min_turns"])
    max_turns = int(cfg["synthesis"]["dialogue_max_turns"])

    history: Dict[str, Any] = {"rounds": []}
    syn_data: List[SyntheticDialogue] = []
    for r in range(R):
        t0 = time.time()
        if r == 0:
            syn_data = synthesize_dataset(
                scenarios, prototypes, teacher,
                n_per_culture=n_per_culture,
                min_turns=min_turns, max_turns=max_turns, seed=seed + r,
            )
        else:
            cand = synthesize_dataset(
                scenarios, prototypes, teacher,
                n_per_culture=n_per_culture,
                min_turns=min_turns, max_turns=max_turns, seed=seed + r,
            )
            scored = score_dataset(
                model, cand, wvs_df, tokenizer,
                n_choices=int(cfg["ucca"]["value_head"]["num_choices"]),
                confidence_metric=conf_metric,
            )
            if use_iw:
                syn_data = importance_weighted_sample(scored, n=len(cand), seed=seed + r)
            else:
                syn_data = top_k_filter(scored, top_k_ratio=top_k_ratio)
        save_dialogues(syn_data, Path(cfg["paths"]["syn_data_dir"]) / f"round_{r}.json")

        # ------ Train M^(r) ------
        max_steps = int(cfg["dryrun"]["n_steps"]) if args.dry_run else None
        log = _train_one_round(
            model, syn_data, wvs_df, tokenizer, cfg,
            max_steps=max_steps,
            ablation=(args.ablation or "none"),
        )

        history["rounds"].append({
            "round": r,
            "n_dialogues": len(syn_data),
            "n_steps": len(log),
            "last_loss": log[-1] if log else None,
            "round_time_sec": time.time() - t0,
        })

    # ------------- Persist ------------------------------------------------
    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / "final.pt"
    torch.save(model.state_dict(), ckpt_path)
    utils.dump_json(history, out_dir / "train_history.json")

    print("\n[train] ===== summary =====")
    for r_info in history["rounds"]:
        print(f" round {r_info['round']}: n_dialogues={r_info['n_dialogues']} "
              f"steps={r_info['n_steps']} last={r_info['last_loss']}")
    print(f"[train] checkpoint: {ckpt_path}")
    print(f"[train] history: {out_dir / 'train_history.json'}")
    return history


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
