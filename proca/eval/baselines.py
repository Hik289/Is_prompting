"""Inference-time baselines from

  - PersonaStandardBaseline: zero-shot persona prompting (`Standard`).
  - CulturalPromptingBaseline: culture-only role prompt
                               (Tao et al. 2024 style, `Cultural`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data import MockTokenizer
from ..model import ProCAModel


@dataclass
class Persona:
    persona_id: str
    culture: str
    age: int
    gender: str
    education: str
    occupation: str

    def to_prompt(self) -> str:
        return (
            f"You are a {self.age}-year-old {self.gender} from {self.culture}, "
            f"with education level '{self.education}' and occupation '{self.occupation}'. "
            f"Please answer the following question reflecting your personal cultural values."
        )


class BaselineModel:
    """Common interface for an inference-time baseline."""
    name: str = "abstract"

    def predict(
        self,
        question_text: str,
        n_choices: int,
        persona: Optional[Persona] = None,
        culture: Optional[str] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class PersonaStandardBaseline(BaselineModel):
    """Standard persona prompting.

    For CODE_REPRO this wraps a ProCAModel-like callable that maps tokenized
    prompts to a distribution over WVS answer choices. With --mock the model
    is the same ProCAModel running on a tiny random transformer.
    """

    name = "PersonaStandard"

    def __init__(
        self,
        model: ProCAModel,
        tokenizer: MockTokenizer,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._rng = np.random.default_rng(seed)

    def _encode(self, text: str):
        ids, mask = self.tokenizer.encode_batch([text])
        return ids, mask

    @torch.no_grad()
    def predict(
        self,
        question_text: str,
        n_choices: int,
        persona: Optional[Persona] = None,
        culture: Optional[str] = None,
    ) -> torch.Tensor:
        sys_prompt = persona.to_prompt() if persona else "You are a survey respondent."
        full = f"{sys_prompt}\n\nQuestion: {question_text}"
        ids, mask = self._encode(full)
        p = self.model.predict_distribution(ids, mask)[0]
        # Note: f_val output dim is fixed at config time; if mismatched, fall back
        # to uniform/trimmed distribution .
        if p.shape[-1] != n_choices:
            if p.shape[-1] > n_choices:
                p = p[:n_choices]
                p = p / p.sum().clamp(min=1e-8)
            else:
                pad = torch.zeros(n_choices - p.shape[-1])
                p = torch.cat([p, pad], dim=0)
                p = p / p.sum().clamp(min=1e-8)
        return p


# ---------------------------------------------------------------------------
class CulturalPromptingBaseline(BaselineModel):
    """Tao et al. 2024 style — culture-only role prompt."""

    name = "CulturalPrompting"

    def __init__(
        self,
        model: ProCAModel,
        tokenizer: MockTokenizer,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def predict(
        self,
        question_text: str,
        n_choices: int,
        persona: Optional[Persona] = None,
        culture: Optional[str] = None,
    ) -> torch.Tensor:
        cul = culture or (persona.culture if persona else "unspecified")
        sys_prompt = (
            f"You are a typical member of {cul}. Answer the following "
            f"survey question reflecting the cultural norms and values of {cul}."
        )
        full = f"{sys_prompt}\n\nQuestion: {question_text}"
        ids, mask = self.tokenizer.encode_batch([full])
        p = self.model.predict_distribution(ids, mask)[0]
        if p.shape[-1] != n_choices:
            if p.shape[-1] > n_choices:
                p = p[:n_choices]
            else:
                p = torch.cat([p, torch.zeros(n_choices - p.shape[-1])], dim=0)
            p = p / p.sum().clamp(min=1e-8)
        return p


# ---------------------------------------------------------------------------
def sample_personas(
    culture: str, n: int = 1000, seed: int = 0
) -> List[Persona]:
    """Sample synthetic personas.

    Without the real WVS we synthesize plausible demographics.
    """
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 80, size=n)
    genders = rng.choice(["male", "female", "non-binary"], size=n, p=[0.49, 0.49, 0.02])
    educs = rng.choice(
        ["no_formal", "primary", "secondary", "bachelor", "graduate"],
        size=n,
        p=[0.05, 0.10, 0.35, 0.35, 0.15],
    )
    occs = rng.choice(
        ["student", "professional", "service", "manual_labor", "retired", "unemployed"],
        size=n,
    )
    out: List[Persona] = []
    for i in range(n):
        out.append(
            Persona(
                persona_id=f"{culture}-{i:05d}",
                culture=culture,
                age=int(ages[i]),
                gender=str(genders[i]),
                education=str(educs[i]),
                occupation=str(occs[i]),
            )
        )
    return out
