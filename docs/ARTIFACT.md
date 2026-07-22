# Artifact Guide

Operational notes for reproducing `ProCA` from the public `Is_prompting` repository.

## Review Path

- `proca/`: Project-specific implementation subtree.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `configs/`: Configuration files for model, benchmark, or experiment settings.
- `assets/`: README and paper-facing visual assets.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `pyproject.toml`: Package metadata and optional extras when available.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
python tests/test_eval_smoke.py
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `bash scripts/eval_all.sh`
- `bash scripts/run_ablations.sh`
- `bash scripts/train_proca.sh`

## Figure Assets

- `assets/fig_motivation.png`
- `assets/proca_architecture.pdf`
- `assets/proca_architecture.png`

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
