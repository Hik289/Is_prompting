"""Shared utilities (config loading, seeding, logging)."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def dump_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Cultural metadata
# ---------------------------------------------------------------------------
CULTURE_INFO: Dict[str, Dict[str, str]] = {
    "CN": {"name": "China", "language": "Chinese", "wvs_code": "156"},
    "DE": {"name": "Germany", "language": "German", "wvs_code": "276"},
    "UK": {"name": "United Kingdom", "language": "English", "wvs_code": "826"},
    "MX": {"name": "Mexico", "language": "Spanish", "wvs_code": "484"},
    "JP": {"name": "Japan", "language": "Japanese", "wvs_code": "392"},
}
