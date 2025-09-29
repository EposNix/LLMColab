"""CLI entry point and orchestration for the LLM Collaborator."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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
def _completion_with_retry(*, model: str, messages: List[dict], temperature: float) -> dict:
    if completion is None:
        raise LiteLLMNotInstalledError(
            "LiteLLM is required. Install it via `pip install litellm`."
        )

    try:
        return completion(model=model, messages=messages, temperature=temperature)
    except Exception as exc:  # pragma: no cover - depends on LiteLLM internals
        raise ModelInvocationError(str(exc)) from exc


def _extract_text(response: dict) -> str:
    """Extract the message content from a LiteLLM completion response."""

    if "choices" in response and response["choices"]:
        choice = response["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if message and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # e.g., OpenAI responses with tool usage
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if "output" in response and isinstance(response["output"], str):
        return response["output"]
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

        usage = response.get("usage") if isinstance(response, dict) else None
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
