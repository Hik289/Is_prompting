"""dialogue_only ablation Table 3.

Turns off the contrastive intent-prototype loss (L_cont). The model is trained
only with L_KL on the dialogue data, isolating the contribution of L_cont.
"""
from __future__ import annotations

import sys

from ..train import main as train_main


def main(argv=None) -> int:
    extra = ["--ablation", "dialogue_only"]
    return train_main(list(argv) + extra if argv else extra)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
