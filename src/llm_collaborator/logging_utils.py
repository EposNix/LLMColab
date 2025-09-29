"""Logging helpers for the LLM Collaborator."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure and return the module logger."""

    logger = logging.getLogger("llm_collaborator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def log_iteration(
    logger: logging.Logger,
    *,
    iteration: int,
    model_name: str,
    critique: str,
    draft_preview: str,
    usage: Optional[dict] = None,
) -> None:
    """Log an iteration in a structured format."""

    payload: dict[str, Any] = {
        "iteration": iteration,
        "model": model_name,
        "critique": critique.strip(),
        "draft_preview": draft_preview.strip()[:200],
    }
    if usage:
        payload["usage"] = usage

    logger.info("iteration_summary: %s", json.dumps(payload, ensure_ascii=False))


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


__all__ = ["setup_logger", "log_iteration", "write_output"]
