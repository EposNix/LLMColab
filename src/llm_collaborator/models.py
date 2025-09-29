"""Model handling helpers for LLM Collaborator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ModelConfig:
    """Represents a single model identifier used by LiteLLM."""

    name: str

    def __post_init__(self) -> None:  # type: ignore[override]
        if not self.name or not self.name.strip():
            raise ValueError("Model name must be a non-empty string")


def parse_model_cycle(raw: str | Sequence[str]) -> List[ModelConfig]:
    """Parse a comma separated list of models into :class:`ModelConfig` objects."""

    if isinstance(raw, str):
        entries = [piece.strip() for piece in raw.split(",")]
    else:
        entries = [piece.strip() for piece in raw]

    entries = [entry for entry in entries if entry]
    if not entries:
        raise ValueError("At least one model must be provided")

    seen = set()
    model_cycle: List[ModelConfig] = []
    for entry in entries:
        if entry.lower() not in seen:
            seen.add(entry.lower())
        model_cycle.append(ModelConfig(name=entry))
    return model_cycle


def cycle_models(models: Sequence[ModelConfig]) -> Iterable[ModelConfig]:
    """Yield models in a round-robin fashion indefinitely."""

    if not models:
        raise ValueError("Model list cannot be empty")

    index = 0
    length = len(models)
    while True:
        yield models[index]
        index = (index + 1) % length


__all__ = ["ModelConfig", "parse_model_cycle", "cycle_models"]
