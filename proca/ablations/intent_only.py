"""intent_only ablation Table 3.

Turns off L_KL and dialogue value-distribution supervision. Only the
contrastive intent-prototype loss (L_cont) is active. Paper reports this
under-performs slightly because the contrastive loss can drive unanchored
representations toward prototypes (representational drift).
"""
from __future__ import annotations

import sys

from ..train import main as train_main


def main(argv=None) -> int:
    extra = ["--ablation", "intent_only"]
    return train_main(list(argv) + extra if argv else extra)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
