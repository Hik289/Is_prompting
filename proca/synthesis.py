"""Theory-Guided Social Interaction Synthesis (TGSIS)

Implements the Eq. (1) generation step:

    X^{(k)} ~ G(s, c_k; theta_g)

where G is a powerful teacher language model conditioned on a base scenario
template `s` (Sotopia-style) and the cultural prototype `c_k`.

Also produces per-turn intent annotations ( — the teacher model
generates free-text intent y_t conditioned on dialogue context and culture).

This is a wrapper layer: the teacher model is abstracted as
:class:`TeacherModel`, with a default :class:`MockTeacher` for smoke testing
and pluggable OpenAI / vLLM stubs.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .prototypes import PrototypeBank


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
@dataclass
class DialogueTurn:
    speaker: str
    text: str
    intent: str = "" # — teacher-generated intent annotation y_t


@dataclass
class Scenario:
    scenario_id: str
    template: str
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticDialogue:
    dialogue_id: str
    culture: str
    scenario_id: str
    turns: List[DialogueTurn]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "culture": self.culture,
            "scenario_id": self.scenario_id,
            "turns": [asdict(t) for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SyntheticDialogue":
        return cls(
            dialogue_id=d["dialogue_id"],
            culture=d["culture"],
            scenario_id=d["scenario_id"],
            turns=[DialogueTurn(**t) for t in d["turns"]],
        )


# ---------------------------------------------------------------------------
# Teacher model abstraction
# ---------------------------------------------------------------------------
class TeacherModel:
    """Interface for the generator G in Eq. (1).

    A concrete implementation must provide :meth:`generate`, returning a list
    of (speaker, utterance, intent) tuples for one synthetic dialogue.
    """

    name: str = "abstract"

    def generate(
        self,
        scenario: Scenario,
        prototype: torch.Tensor,
        culture: str,
        n_turns: int,
    ) -> List[DialogueTurn]:
        raise NotImplementedError


class MockTeacher(TeacherModel):
    """Deterministic-ish mock teacher for dry-run / smoke tests."""

    name = "mock"

    _CULTURE_FLAVOR = {
        "CN": "respectful and harmony-oriented",
        "DE": "direct and structured",
        "UK": "polite and indirect",
        "MX": "warm and family-oriented",
        "JP": "considerate and hierarchical",
    }
    _INTENT_FLAVOR = {
        "CN": "maintain face and group harmony",
        "DE": "make objective progress on the task",
        "UK": "preserve politeness while resolving the issue",
        "MX": "express warmth and personal regard",
        "JP": "show deference and read the room",
    }

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def generate(
        self,
        scenario: Scenario,
        prototype: torch.Tensor,
        culture: str,
        n_turns: int,
    ) -> List[DialogueTurn]:
        flavor = self._CULTURE_FLAVOR.get(culture, "culturally adapted")
        intent_flavor = self._INTENT_FLAVOR.get(culture, "achieve the social objective")
        roles = scenario.roles or ["AgentA", "AgentB"]
        turns: List[DialogueTurn] = []
        for t in range(n_turns):
            spk = roles[t % len(roles)]
            text = (
                f"[{spk}] {scenario.template} (turn {t+1}, {flavor})."
            )
            intent = (
                f"To {intent_flavor} while engaging with the scenario "
                f"'{scenario.scenario_id}' from a {culture} cultural perspective."
            )
            turns.append(DialogueTurn(speaker=spk, text=text, intent=intent))
        return turns


class OpenAITeacher(TeacherModel):
    """Stub for OpenAI-backed teacher. Concrete API call left as a hook."""

    name = "openai"

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(
        self,
        scenario: Scenario,
        prototype: torch.Tensor,
        culture: str,
        n_turns: int,
    ) -> List[DialogueTurn]: # pragma: no cover — network-bound
        raise NotImplementedError(
            "OpenAITeacher.generate(): wire up the OpenAI client here."
        )


class VLLMTeacher(TeacherModel):
    """Stub for self-hosted vLLM-backed teacher (e.g. GPT-OSS 120B, Qwen3 32B)."""

    name = "vllm"

    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint
        self.model_name = model_name

    def generate(
        self,
        scenario: Scenario,
        prototype: torch.Tensor,
        culture: str,
        n_turns: int,
    ) -> List[DialogueTurn]: # pragma: no cover — network-bound
        raise NotImplementedError(
            "VLLMTeacher.generate(): wire up the vLLM HTTP/grpc client here."
        )


def make_teacher(backend: str = "mock", **kwargs: Any) -> TeacherModel:
    backend = (backend or "mock").lower()
    if backend == "mock":
        return MockTeacher(**kwargs)
    if backend == "openai":
        return OpenAITeacher(**kwargs)
    if backend == "vllm":
        return VLLMTeacher(**kwargs)
    raise ValueError(f"Unknown teacher backend: {backend!r}")


# ---------------------------------------------------------------------------
# Sotopia scenario loader (paper cites Sotopia for `s`)
# ---------------------------------------------------------------------------
def load_sotopia_scenarios(path: str | Path) -> List[Scenario]:
    """Load scenarios from a Sotopia-style JSON file.

    Expected schema (per item):
      {"scenario_id": "...", "template": "...", "roles": ["A","B"], "metadata": {...}}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sotopia scenarios not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Sotopia JSON must be a list of scenario dicts.")
    return [
        Scenario(
            scenario_id=str(d["scenario_id"]),
            template=str(d["template"]),
            roles=list(d.get("roles", ["AgentA", "AgentB"])),
            metadata=dict(d.get("metadata", {})),
        )
        for d in data
    ]


# ---------------------------------------------------------------------------
# Synthesis driver (Eq. 1, looped per culture)
# ---------------------------------------------------------------------------
def synthesize_dataset(
    scenarios: Sequence[Scenario],
    prototypes: PrototypeBank,
    teacher: TeacherModel,
    n_per_culture: int = 10,
    min_turns: int = 6,
    max_turns: int = 12,
    seed: int = 0,
) -> List[SyntheticDialogue]:
    """Generate a synthetic dialogue corpus across cultures.

    Implements Eq. (1) — for each culture k and each sampled scenario s,
    invokes the teacher G(s, c_k) and collects per-turn intent annotations.
    """
    rng = random.Random(seed)
    out: List[SyntheticDialogue] = []
    n_scenarios = len(scenarios)
    if n_scenarios == 0:
        raise ValueError("No scenarios provided to synthesize_dataset.")
    next_id = 0
    for k in prototypes.cultures:
        proto = prototypes.get(k)
        for i in range(n_per_culture):
            scen = scenarios[rng.randrange(n_scenarios)]
            n_turns = rng.randint(min_turns, max_turns)
            turns = teacher.generate(scen, proto, k, n_turns)
            out.append(
                SyntheticDialogue(
                    dialogue_id=f"d{next_id:07d}",
                    culture=k,
                    scenario_id=scen.scenario_id,
                    turns=turns,
                )
            )
            next_id += 1
    return out


def save_dialogues(dialogues: Sequence[SyntheticDialogue], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([d.to_dict() for d in dialogues], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_dialogues(path: str | Path) -> List[SyntheticDialogue]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SyntheticDialogue.from_dict(d) for d in data]
