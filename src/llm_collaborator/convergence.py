"""Convergence detection helpers."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional


@dataclass
class ConvergenceResult:
    """Represents the similarity between two drafts."""

    similarity: float
    threshold: float

    @property
    def converged(self) -> bool:
        return self.similarity >= self.threshold


def similarity_ratio(previous: Optional[str], current: Optional[str]) -> float:
    """Compute a similarity score between two drafts."""

    if not previous or not current:
        return 0.0

    matcher = SequenceMatcher(None, previous, current)
    return matcher.quick_ratio()


def has_converged(previous: Optional[str], current: Optional[str], threshold: float) -> ConvergenceResult:
    """Check whether the drafts are similar enough to stop iterating."""

    score = similarity_ratio(previous, current)
    return ConvergenceResult(similarity=score, threshold=threshold)


__all__ = ["ConvergenceResult", "has_converged", "similarity_ratio"]
