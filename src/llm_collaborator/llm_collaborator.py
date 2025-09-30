"""CLI entry point and orchestration for the LLM Collaborator."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
from collections.abc import Mapping, Sequence as SequenceABC

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .convergence import ConvergenceResult, has_converged
from .logging_utils import log_iteration, setup_logger, write_output
from .models import ModelConfig, parse_model_cycle
from .prompts import build_messages

try:  # pragma: no cover - import guard for optional dependency discovery
    from litellm import completion
except ImportError as exc:  # pragma: no cover
    completion = None  # type: ignore[assignment]

CRITIQUE_TAG = "critique"
DRAFT_TAG = "draft"


class LiteLLMNotInstalledError(RuntimeError):
    """Raised when LiteLLM is not installed but required."""


class ModelInvocationError(RuntimeError):
    """Raised when a model invocation fails after retries."""


@dataclass
class CollaborationResult:
    """Container for the result of a collaboration run."""

    final_draft: str
    history: List[Tuple[str, str, str]]
    convergence: Optional[ConvergenceResult]


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, max=20),
)
def _completion_with_retry(*, model: str, messages: List[dict], temperature: float) -> Any:
    if completion is None:
        raise LiteLLMNotInstalledError(
            "LiteLLM is required. Install it via `pip install litellm`."
        )

    try:
        return completion(model=model, messages=messages, temperature=temperature)
    except Exception as exc:  # pragma: no cover - depends on LiteLLM internals
        raise ModelInvocationError(str(exc)) from exc


def _response_to_dict(response: Any) -> dict:
    """Best-effort conversion of LiteLLM responses to plain dictionaries."""

    if isinstance(response, Mapping):
        return dict(response)
    for attr in ("model_dump", "dict"):
        if hasattr(response, attr):
            method = getattr(response, attr)
            try:
                data = method()
            except TypeError:
                continue
            if isinstance(data, Mapping):
                return dict(data)
    return {}


def _coerce_content(value: Any) -> Optional[str]:
    """Convert LiteLLM message content into a text string when possible."""

    if isinstance(value, str):
        return value
    if isinstance(value, SequenceABC):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                parts.append(getattr(item, "text"))
        if parts:
            return "".join(parts)
    return None


def _extract_text(response: Any) -> str:
    """Extract the message content from a LiteLLM completion response."""

    data = _response_to_dict(response)

    choices = data.get("choices") if isinstance(data, Mapping) else None
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message: Optional[Mapping[str, Any]] = None
        if isinstance(first_choice, Mapping):
            raw_message = first_choice.get("message")
            if isinstance(raw_message, Mapping):
                message = raw_message
        if message:
            content = _coerce_content(message.get("content"))
            if content is not None:
                return content

    for key in ("output", "output_text"):
        value = data.get(key) if isinstance(data, Mapping) else None
        text = _coerce_content(value)
        if text:
            return text

    choices_attr = getattr(response, "choices", None)
    if isinstance(choices_attr, SequenceABC) and choices_attr:
        first_choice = choices_attr[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            content = _coerce_content(getattr(message, "content", None))
            if content is not None:
                return content

    output_text_attr = getattr(response, "output_text", None)
    text = _coerce_content(output_text_attr)
    if text:
        return text

    raise ValueError("Unable to extract message content from model response")


def _extract_tagged_section(text: str, tag: str) -> Optional[str]:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag not in text or end_tag not in text:
        return None
    start = text.index(start_tag) + len(start_tag)
    end = text.index(end_tag, start)
    return text[start:end].strip()


def extract_sections(text: str) -> Tuple[str, str]:
    """Extract the <critique> and <draft> sections from model output."""

    critique = _extract_tagged_section(text, CRITIQUE_TAG)
    draft = _extract_tagged_section(text, DRAFT_TAG)
    if not critique or not draft:
        raise ValueError(
            "Model response must include <critique> and <draft> sections."
        )
    return critique, draft


def run_collaboration(
    *,
    task: str,
    model_cycle: Sequence[Union[str, ModelConfig]],
    iterations: int,
    temperature: float = 0.2,
    verbose: bool = False,
    stop_on_convergence: bool = True,
    convergence_threshold: float = 0.995,
) -> CollaborationResult:
    """Run the collaboration loop and return the final draft and history."""

    if not task or not task.strip():
        raise ValueError("Task must be a non-empty string")
    if iterations <= 0:
        raise ValueError("Iterations must be greater than zero")

    logger = setup_logger(verbose)

    models: List[ModelConfig] = []
    for entry in model_cycle:
        if isinstance(entry, ModelConfig):
            models.append(entry)
        else:
            models.append(ModelConfig(name=entry))

    if not models:
        raise ValueError("No models provided")

    history: List[Tuple[str, str, str]] = []
    critique_summaries: List[str] = []
    last_draft: Optional[str] = None
    convergence_info: Optional[ConvergenceResult] = None

    for turn in range(1, iterations + 1):
        model = models[(turn - 1) % len(models)]
        messages = build_messages(
            task=task,
            last_draft=last_draft,
            actor_name=model.name,
            critique_summaries=critique_summaries[-3:],
        )

        response = _completion_with_retry(model=model.name, messages=messages, temperature=temperature)
        content = _extract_text(response)
        critique, draft = extract_sections(content)
        history.append((model.name, critique, draft))
        critique_summaries.append(f"{model.name}: {critique.strip()}")

        response_dict = _response_to_dict(response)
        usage: Optional[Any]
        if response_dict:
            usage = response_dict.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            try:
                usage = usage.model_dump()
            except TypeError:
                usage = None
        log_iteration(
            logger,
            iteration=turn,
            model_name=model.name,
            critique=critique,
            draft_preview=draft[:500],
            usage=usage,
        )

        if last_draft is not None and stop_on_convergence:
            convergence_info = has_converged(last_draft, draft, convergence_threshold)
            if convergence_info.converged:
                logger.info(
                    "Converged after %s iterations (similarity=%.3f)",
                    turn,
                    convergence_info.similarity,
                )
                last_draft = draft
                break

        last_draft = draft

    if last_draft is None:
        raise RuntimeError("No draft was produced during the collaboration")

    return CollaborationResult(final_draft=last_draft, history=history, convergence=convergence_info)


def _load_task_from_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM collaboration loop")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", type=str, help="Task description to run")
    task_group.add_argument("--task-file", type=Path, help="Path to a file containing the task")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of LiteLLM model identifiers")
    parser.add_argument("--iters", type=int, default=3, help="Number of iterations to run")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the models")
    parser.add_argument("--out", type=Path, help="Optional path to save the final draft")
    parser.add_argument("--convergence-threshold", type=float, default=0.995, help="Similarity threshold for convergence")
    parser.add_argument("--no-convergence-stop", action="store_true", help="Disable early stopping on convergence")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    task_text = args.task
    if args.task_file is not None:
        task_text = _load_task_from_path(args.task_file)

    try:
        model_configs = parse_model_cycle(args.models)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        result = run_collaboration(
            task=task_text,
            model_cycle=model_configs,
            iterations=args.iters,
            temperature=args.temperature,
            verbose=args.verbose,
            stop_on_convergence=not args.no_convergence_stop,
            convergence_threshold=args.convergence_threshold,
        )
    except LiteLLMNotInstalledError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Error running collaboration: {exc}", file=sys.stderr)
        return 1

    if args.out:
        write_output(args.out, result.final_draft)

    print(result.final_draft)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
