# ProCA: Prototype-driven Contrastive Cultural Alignment

Official implementation of

> **Is Prompting Enough for Cultural Alignment? Learning Prototype-Aware Values with Contrastive Adaptation in Large Language Models**

ProCA reformulates cultural value alignment as contrastive representation
learning in a *cultural prototype space* derived from human survey data.
It combines (i) theory-guided social interaction synthesis, (ii) a unified
KL + contrastive training objective, and (iii) an iterative refinement loop
that re-curates training data using the model's own cultural understanding.

---

## Overview                  

| Component                              | Module                                |
|----------------------------------------|---------------------------------------|
| Cultural prototypes  `C`               | `proca/prototypes.py`                 |
| Dialogue synthesis  `G(s, c_k)`        | `proca/synthesis.py`                  |
| Context encoder + value head           | `proca/encoders.py`                   |
| Intent encoder                         | `proca/encoders.py`                   |
| KL loss / contrastive loss / unified   | `proca/losses.py`                     |
| Refinement scoring + filtering         | `proca/refinement.py`                 |
| Training loop                          | `proca/train.py`                      |
| WVS / cross-lingual evaluation         | `proca/eval/`                         |
| Persona & Cultural-prompt baselines    | `proca/eval/baselines.py`             |
| Ablations                              | `proca/ablations/`                    |

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or, editable install with dev extras:
pip install -e .[dev]
```

GPU training requires PyTorch with CUDA, `transformers`, and `peft`.
CPU is sufficient for the lightweight test mode described below.

---

## Quickstart

### 1. Run the test suite

```bash
pytest tests/ -v
```

### 2. Train on a tiny in-memory dataset (no downloads)

```bash
python -m proca.train --config configs/proca_default.yaml --mock --dry-run
```

The `--mock` flag swaps the HuggingFace backbones for a small random-init
transformer (`vocab=1000, hidden=64, 2 layers`) and replaces the WVS /
Sotopia loaders with bundled fixtures, so the entire `train → score → refine
→ retrain` loop runs in seconds on a laptop. `--dry-run` further caps the
number of optimization steps. These flags are intended for CI and for
contributors without GPUs; production runs simply omit them.

### 3. Evaluate on the WVS

```bash
python -m proca.eval.wvs_eval --culture China
```

For full benchmarking across all five regions and both inference-time
baselines:

```bash
bash scripts/eval_all.sh --ckpt artifacts/checkpoints/final.pt
```

---

## Training on real data

1. **WVS Wave 7.** Place the per-respondent CSV at `data/wvs_wave7.csv`.
   The loader (`proca.data.load_wvs_csv`) expects a `culture` column and
   integer `Q*` answer columns.
2. **Sotopia scenarios.** Place the scenario JSON at
   `data/sotopia_scenarios.json` (schema matches `data/mock_sotopia.json`).
3. **Teacher model.** Implement `generate()` in
   `proca.synthesis.OpenAITeacher` or `proca.synthesis.VLLMTeacher`
   (a no-op `MockTeacher` is provided for testing).
4. **Backbone.** Choose a model card from `configs/models/` — currently
   `gpt_oss_{20b,120b}`, `qwen3_{8b,14b,32b}`, `gemma3_{4b,12b,27b}` —
   and pass it via `--model-config`.
5. Launch:

```bash
python -m proca.train \
    --config configs/proca_default.yaml \
    --model-config configs/models/gpt_oss_20b.yaml
```

---

## Configuration

All hyperparameters live in `configs/proca_default.yaml`:

| Hyperparameter      | Default  |
|---------------------|----------|
| `d_c` (prototype)   | 128      |
| `λ` (contrastive)   | 0.5      |
| `τ` (temperature)   | 0.07     |
| LoRA `r` / `α`      | 64 / 128 |
| Learning rate       | 2e-5     |
| Batch size          | 32       |
| Epochs              | 3        |
| Refinement rounds R | 2        |
| Top-K ratio         | 0.70     |
| Dialogue turns      | 6 – 12   |
| Personas / culture  | 1000     |
| WVS questions       | 44       |

The contrastive loss uses in-batch negatives (each dialogue is contrasted
against the prototypes of all other cultures in the batch).

---

## Reproducing the experiments

| Result                                                  | Entry point                                                                  |
|---------------------------------------------------------|------------------------------------------------------------------------------|
| KL-D across five cultures and eight backbones           | `bash scripts/eval_all.sh --ckpt <path>`                                     |
| Reasoning-only fine-tuning baseline (GSM8K / MathChat)  | `python -m proca.ablations.reasoning_only --dataset gsm8k`                   |
| `dialogue_only` ablation (drops `L_cont`)               | `python -m proca.ablations.dialogue_only`                                    |
| `intent_only` ablation (drops `L_KL`)                   | `python -m proca.ablations.intent_only`                                      |
| Cross-lingual transfer                                  | `python -m proca.eval.xling_eval --culture <name>`                           |
| Teacher-model robustness (Qwen3 32B vs. GPT-OSS 120B)   | `python -m proca.ablations.teacher_swap --teacher qwen3_32b`                 |

End-to-end smoke test (for CI):

```bash
bash scripts/train_proca.sh   --mock --dry-run
bash scripts/eval_all.sh      --mock
bash scripts/run_ablations.sh --mock --dry-run
```

---

## Repository layout

```
configs/                YAML configuration files
  models/               Per-backbone model cards
data/                   Bundled lightweight fixtures
proca/                  Library source
  prototypes.py         Cultural prototype space
  synthesis.py          Teacher-guided dialogue generation
  encoders.py           Context + value head + intent encoder
  losses.py             KL, contrastive, unified objectives
  model.py              ProCAModel (orchestration + LoRA)
  refinement.py         Iterative scoring + filtering
  train.py              Training loop (Algorithm 1)
  data.py               WVS / dialogue loaders & batching
  eval/                 WVS, cross-lingual, baselines
  ablations/            Ablation entry points
scripts/                Convenience shell wrappers
tests/                  pytest suite (CPU only)
```

---

## Ethics

ProCA aligns models to *aggregate* World Values Survey response
distributions, not to prescriptive cultural norms. Aggregate trends should
not be assumed to apply to any individual, and cultural contextualization
must never be used to justify human-rights violations. We recommend
human-supervised deployment.

---

## Citation

If you use this code, please cite the original paper.

## License

Released under the MIT License — see [LICENSE](LICENSE).
