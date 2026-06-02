"""End-to-end dry-run for Algorithm 1."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from proca.train import main as train_main


def test_dryrun_runs_two_steps(tmp_path):
    cfg_path = Path(__file__).parents[1] / "configs" / "proca_default.yaml"
    out_dir = tmp_path / "results"
    rc = train_main([
        "--config", str(cfg_path),
        "--mock", "--dry-run",
        "--output-dir", str(out_dir),
    ])
    assert rc == 0
    history = out_dir / "train_history.json"
    assert history.exists()
