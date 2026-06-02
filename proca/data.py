"""Data loading + batching utilities for ProCA.

Provides:
  - `MockTokenizer`: deterministic hash-based tokenizer for --mock paths.
  - `DialogueBatch` dataclass.
  - `make_batch()` collator producing tensors required by `ProCAModel.forward`.
  - `load_wvs_csv()` + `load_mock_wvs()` for the WVS table backing prototypes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .synthesis import SyntheticDialogue


# ---------------------------------------------------------------------------
# Mock tokenizer (no external download required)
# ---------------------------------------------------------------------------
class MockTokenizer:
    """Deterministic whitespace-split hash tokenizer.

    Maps tokens to ints in [3, vocab_size). Special ids: pad=0, bos=1, eos=2.
    Good enough to exercise the pipeline; not semantically meaningful.
    """

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2

    def __init__(self, vocab_size: int = 1000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _tok(self, w: str) -> int:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        return 3 + (h % (self.vocab_size - 3))

    def encode(self, text: str) -> List[int]:
        ids = [self.BOS_ID]
        for w in text.split():
            ids.append(self._tok(w))
            if len(ids) >= self.max_length - 1:
                break
        ids.append(self.EOS_ID)
        return ids

    def encode_batch(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode(t) for t in texts]
        L = max((len(s) for s in seqs), default=1)
        L = min(L, self.max_length)
        ids = torch.full((len(seqs), L), self.PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:L]
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids, mask


# ---------------------------------------------------------------------------
# WVS table loading
# ---------------------------------------------------------------------------
def load_wvs_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "culture" not in df.columns:
        raise ValueError("WVS CSV must include a 'culture' column.")
    return df


def load_mock_wvs(
    cultures: Sequence[str] = ("CN", "DE", "UK", "MX", "JP"),
    n_questions: int = 8,
    n_respondents_per_culture: int = 30,
    n_choices: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a small mock WVS table for tests/dry-runs."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    q_cols = [f"Q{i+1}" for i in range(n_questions)]
    for k_idx, k in enumerate(cultures):
        # Slightly culture-specific bias so prototypes differ.
        bias = rng.dirichlet(np.ones(n_choices) * (1.0 + 0.5 * k_idx))
        for r in range(n_respondents_per_culture):
            row: Dict[str, Any] = {"culture": k, "respondent_id": f"{k}-{r:04d}"}
            for q in q_cols:
                row[q] = int(rng.choice(np.arange(1, n_choices + 1), p=bias))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dialogue → batch tensors
# ---------------------------------------------------------------------------
@dataclass
class DialogueBatch:
    ctx_input_ids: torch.Tensor # (B, L_ctx)
    ctx_attention_mask: torch.Tensor
    intent_input_ids: torch.Tensor # (B, T', L_int)
    intent_attention_mask: torch.Tensor # (B, T', L_int)
    intent_turn_mask: torch.Tensor # (B, T') — 1 where turn exists
    culture_ids: torch.Tensor # (B,)
    p_wvs: torch.Tensor # (B, C) ground-truth distribution
    question_ids: torch.Tensor # (B,) which WVS question q
    meta: List[Dict[str, Any]] = field(default_factory=list)


def _flatten_dialogue_text(d: SyntheticDialogue) -> str:
    return " ".join(f"<{t.speaker}> {t.text}" for t in d.turns)


def _ground_truth_distribution(
    wvs_df: pd.DataFrame, culture: str, q_col: str, n_choices: int
) -> torch.Tensor:
    sub = wvs_df[wvs_df["culture"] == culture]
    if len(sub) == 0:
        # Uniform fallback .
        return torch.full((n_choices,), 1.0 / n_choices)
    choices = np.arange(1, n_choices + 1)
    counts = np.array([(sub[q_col] == c).sum() for c in choices], dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return torch.full((n_choices,), 1.0 / n_choices)
    return torch.tensor(counts / total, dtype=torch.float32)


def make_batch(
    dialogues: Sequence[SyntheticDialogue],
    wvs_df: pd.DataFrame,
    culture_to_id: Dict[str, int],
    tokenizer: MockTokenizer,
    n_choices: int = 5,
    n_intent_turns: int = 4,
    q_cols: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> DialogueBatch:
    """Collate a list of synthetic dialogues into a `DialogueBatch`.

    For each dialogue, randomly sample one WVS question q and build:
      - context tokens (whole dialogue concatenated),
      - intent tokens (up to T' turn intents),
      - culture id,
      - p_wvs(q,k) ground-truth distribution.
    """
    if q_cols is None:
        q_cols = [c for c in wvs_df.columns if c.startswith("Q")]
    if not q_cols:
        raise ValueError("No Q* columns in WVS table.")
    rng = np.random.default_rng(seed)

    ctx_texts: List[str] = []
    intent_text_grid: List[List[str]] = []
    turn_mask_rows: List[List[int]] = []
    culture_ids: List[int] = []
    p_wvs_list: List[torch.Tensor] = []
    question_ids: List[int] = []
    meta: List[Dict[str, Any]] = []

    for d in dialogues:
        ctx_texts.append(_flatten_dialogue_text(d))
        # Collect first n_intent_turns intents (or pad with "")
        intents: List[str] = []
        mask_row: List[int] = []
        for t_idx in range(n_intent_turns):
            if t_idx < len(d.turns):
                turn = d.turns[t_idx]
                intent_text = f"<{turn.speaker}> {turn.text} || INTENT: {turn.intent}"
                intents.append(intent_text)
                mask_row.append(1)
            else:
                intents.append("")
                mask_row.append(0)
        intent_text_grid.append(intents)
        turn_mask_rows.append(mask_row)

        culture_ids.append(culture_to_id[d.culture])
        q_idx = int(rng.integers(0, len(q_cols)))
        q_col = q_cols[q_idx]
        p_wvs_list.append(_ground_truth_distribution(wvs_df, d.culture, q_col, n_choices))
        question_ids.append(q_idx)
        meta.append({"dialogue_id": d.dialogue_id, "culture": d.culture, "question": q_col})

    ctx_ids, ctx_mask = tokenizer.encode_batch(ctx_texts)

    # Tokenize each turn slot independently then stack to (B, T', L_int).
    B = len(dialogues)
    Tp = n_intent_turns
    per_slot_ids: List[torch.Tensor] = []
    per_slot_masks: List[torch.Tensor] = []
    for t_idx in range(Tp):
        texts_t = [intent_text_grid[b][t_idx] for b in range(B)]
        ids_t, mask_t = tokenizer.encode_batch(texts_t)
        per_slot_ids.append(ids_t)
        per_slot_masks.append(mask_t)
    # Right-pad along L_int to a common length.
    L_int = max(x.shape[1] for x in per_slot_ids)
    intent_ids = torch.zeros(B, Tp, L_int, dtype=torch.long)
    intent_mask = torch.zeros(B, Tp, L_int, dtype=torch.long)
    for t_idx in range(Tp):
        L_t = per_slot_ids[t_idx].shape[1]
        intent_ids[:, t_idx, :L_t] = per_slot_ids[t_idx]
        intent_mask[:, t_idx, :L_t] = per_slot_masks[t_idx]

    turn_mask = torch.tensor(turn_mask_rows, dtype=torch.long)

    return DialogueBatch(
        ctx_input_ids=ctx_ids,
        ctx_attention_mask=ctx_mask,
        intent_input_ids=intent_ids,
        intent_attention_mask=intent_mask,
        intent_turn_mask=turn_mask,
        culture_ids=torch.tensor(culture_ids, dtype=torch.long),
        p_wvs=torch.stack(p_wvs_list, dim=0),
        question_ids=torch.tensor(question_ids, dtype=torch.long),
        meta=meta,
    )


def write_mock_sotopia(path: str | Path, n: int = 10) -> None:
    """Emit a tiny Sotopia-style JSON file used by tests."""
    items = []
    for i in range(n):
        items.append(
            {
                "scenario_id": f"sotopia_mock_{i:03d}",
                "template": (
                    f"Two colleagues are meeting to discuss a delayed shipment "
                    f"in scenario {i}. They must reach an agreement on next steps."
                ),
                "roles": ["Manager", "Vendor"],
                "metadata": {"source": "mock"},
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(items, indent=2), encoding="utf-8")
