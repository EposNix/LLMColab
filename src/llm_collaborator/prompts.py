"""Prompt building utilities for the LLM Collaborator."""
from __future__ import annotations

from typing import List, Sequence

SYSTEM_PROMPT_TEMPLATE = (
    "You are {actor_name}, an expert collaborator in a multi-model writing loop. "
    "Work with the other models to iteratively improve the draft. Provide helpful "
    "feedback when needed and share a refined contribution that can stand on its "
    "own each turn."
)


def build_messages(
    task: str,
    last_draft: str | None,
    actor_name: str,
    response_summaries: Sequence[str] | None = None,
) -> List[dict]:
    """Build a message list to send to LiteLLM.

    Parameters
    ----------
    task:
        The original task description from the user.
    last_draft:
        The latest full draft produced by the loop. ``None`` on the first turn.
    actor_name:
        A human readable name for the model producing the next turn.
    response_summaries:
        Optional list of short response summaries to provide additional context.

    Returns
    -------
    list of dict
        Messages compatible with the OpenAI-style chat schema used by LiteLLM.
    """

    messages: List[dict] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(actor_name=actor_name or "The collaborator"),
        }
    ]

    user_sections: List[str] = [f"<task>\n{task.strip()}\n</task>"]
    if response_summaries:
        history = "\n\n".join(summary.strip() for summary in response_summaries if summary.strip())
        if history:
            user_sections.append(f"<recent_updates>\n{history}\n</recent_updates>")

    if last_draft:
        user_sections.append(f"<current_draft>\n{last_draft.strip()}\n</current_draft>")

    user_sections.append(
        "Offer constructive thoughts on the work so far if helpful, then provide "
        "your updated contribution in plain text or code as appropriate."
    )

    messages.append({"role": "user", "content": "\n\n".join(user_sections)})
    return messages


__all__ = ["build_messages", "SYSTEM_PROMPT_TEMPLATE"]
