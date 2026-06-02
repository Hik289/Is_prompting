"""Cultural prototype space C ∈ R^{|K|×d_c}.

Eq. (paper): we initialize prototypes c_k ∈ R^{d_c} via PCA on aggregated
WVS response distributions for culture k. Prototypes serve as *fixed anchors*
for downstream contrastive alignment (Eq. 4) and refinement scoring (Eq. 6).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


@dataclass
class PrototypeBank:
    """Container for fixed cultural prototype anchors.

    Attributes
    ----------
    cultures: ordered list of culture codes (e.g. ['CN','DE','UK','MX','JP'])
    matrix: tensor of shape (|K|, d_c) – the prototype matrix C in
    explained_variance: optional PCA explained-variance ratios (informational).
    """

    cultures: List[str]
    matrix: torch.Tensor # shape: (|K|, d_c)
    explained_variance: np.ndarray | None = None

    @property
    def d_c(self) -> int:
        return int(self.matrix.shape[1])

    def index(self, culture: str) -> int:
        return self.cultures.index(culture)

    def get(self, culture: str) -> torch.Tensor:
        return self.matrix[self.index(culture)]

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cultures": self.cultures,
                "matrix": self.matrix.cpu(),
                "explained_variance": self.explained_variance,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PrototypeBank":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            cultures=obj["cultures"],
            matrix=obj["matrix"],
            explained_variance=obj.get("explained_variance"),
        )


def _aggregate_wvs(wvs_df: pd.DataFrame, cultures: Sequence[str]) -> np.ndarray:
    """Aggregate WVS responses per culture into a fixed-length feature vector.

    Strategy:
      - per culture, for each WVS question (column starting with 'Q'),
        compute the empirical answer distribution over choices 1..K_q,
        then concatenate across questions.
    Returns
    -------
    feats: array (|cultures|, F) with F = sum_q K_q.
    """
    q_cols = [c for c in wvs_df.columns if c.startswith("Q")]
    if not q_cols:
        raise ValueError("No WVS Q-columns found (expected columns starting with 'Q').")

    # Determine choices per question (use observed support).
    choices_per_q: Dict[str, List[int]] = {}
    for q in q_cols:
        vals = sorted(int(v) for v in wvs_df[q].dropna().unique() if int(v) >= 0)
        if not vals:
            vals = [1]
        choices_per_q[q] = vals

    feats = []
    for k in cultures:
        sub = wvs_df[wvs_df["culture"] == k]
        if len(sub) == 0:
            # Underspecified: zero vector if culture missing.
            feats.append(np.zeros(sum(len(v) for v in choices_per_q.values())))
            continue
        per_q_dists = []
        for q in q_cols:
            choices = choices_per_q[q]
            counts = np.array([(sub[q] == c).sum() for c in choices], dtype=np.float64)
            total = counts.sum()
            dist = counts / total if total > 0 else np.ones_like(counts) / len(counts)
            per_q_dists.append(dist)
        feats.append(np.concatenate(per_q_dists))
    return np.stack(feats, axis=0)


def build_prototypes(
    wvs_df: pd.DataFrame,
    cultures: Sequence[str],
    n_components: int = 128,
    random_state: int = 0,
) -> PrototypeBank:
    """Build cultural prototypes via PCA over aggregated WVS responses (§3.1).

    Parameters
    ----------
    wvs_df : DataFrame with columns including 'culture' + 'Q*' answer columns.
    cultures : ordered iterable of culture codes.
    n_components : d_c, the prototype dimensionality (default 128,
    """
    cultures = list(cultures)
    feats = _aggregate_wvs(wvs_df, cultures) # (|K|, F)

    n_samples, n_features = feats.shape
    # PCA cannot extract more components than min(n_samples, n_features); pad with
    # zeros if the request exceeds what is mathematically extractable. This is
    # consistent with the paper's intent: produce a d_c-dim representation.
    max_components = min(n_samples, n_features)
    used = min(n_components, max_components)
    if used < 2:
        # Fall back to raw feature truncation if PCA degenerate (e.g. tiny mock).
        proj = feats[:, :n_components] if feats.shape[1] >= n_components else np.pad(
            feats, ((0, 0), (0, n_components - feats.shape[1]))
        )
        ev = None
    else:
        pca = PCA(n_components=used, random_state=random_state)
        proj = pca.fit_transform(feats) # (|K|, used)
        if used < n_components:
            proj = np.pad(proj, ((0, 0), (0, n_components - used)))
        ev = pca.explained_variance_ratio_

    matrix = torch.tensor(proj, dtype=torch.float32)
    return PrototypeBank(cultures=cultures, matrix=matrix, explained_variance=ev)


def random_prototypes(cultures: Sequence[str], d_c: int = 128, seed: int = 0) -> PrototypeBank:
    """Random orthogonal-ish prototypes for --mock / dry-run smoke tests."""
    g = torch.Generator().manual_seed(seed)
    cultures = list(cultures)
    mat = torch.randn(len(cultures), d_c, generator=g)
    mat = torch.nn.functional.normalize(mat, dim=-1) * np.sqrt(d_c)
    return PrototypeBank(cultures=cultures, matrix=mat)
