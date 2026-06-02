"""teacher_swap ablation "Robustness to teacher model".

Re-runs synthesis with a different teacher G (e.g., Qwen3 32B instead of
GPT-OSS 120B). Confirms ProCA generalises across teachers but performance
ceiling depends on teacher quality.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..train import main as train_main


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3_32b",
                   help="Teacher model to use (e.g., gpt_oss_120b, qwen3_32b).")
    p.add_argument("--config", default="configs/proca_default.yaml")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args(argv)

    forwarded = ["--config", args.config, "--teacher", args.teacher]
    if args.mock:
        forwarded.append("--mock")
    if args.dry_run:
        forwarded.append("--dry-run")
    forwarded += extra
    return train_main(forwarded)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
