from unittest.mock import patch

from llm_collaborator.llm_collaborator import (
    CollaborationResult,
    _extract_text,
    run_collaboration,
)

from litellm.utils import Choices, Message, ModelResponse, Usage


def test_run_collaboration_uses_round_robin_models():
    responses = [
        {
            "choices": [
                {
                    "message": {"content": "Model A first response"}
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": "Model B second response"}
                }
            ]
        },
    ]

    with patch("llm_collaborator.llm_collaborator._completion_with_retry", side_effect=responses):
        result = run_collaboration(
            task="Write a poem",
            model_cycle=["model-a", "model-b"],
            iterations=2,
            stop_on_convergence=False,
        )

    assert isinstance(result, CollaborationResult)
    assert result.final_draft == "Model B second response"
    assert [entry[0] for entry in result.history] == ["model-a", "model-b"]
    assert [entry[1] for entry in result.history] == [
        "Model A first response",
        "Model B second response",
    ]


def test_extract_text_supports_model_response_object():
    response = ModelResponse(
        id="resp-1",
        created=0,
        model="demo",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Here is a combined response body.",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    text = _extract_text(response)

    assert "combined response body" in text


def test_run_collaboration_accepts_model_response_objects():
    responses = [
        ModelResponse(
            id="resp-1",
            created=0,
            model="demo",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Alpha turn response",
                        role="assistant",
                    ),
                )
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        ),
        ModelResponse(
            id="resp-2",
            created=0,
            model="demo",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Beta follow-up response",
                        role="assistant",
                    ),
                )
            ],
            usage=Usage(prompt_tokens=4, completion_tokens=5, total_tokens=9),
        ),
    ]

    with patch("llm_collaborator.llm_collaborator._completion_with_retry", side_effect=responses):
        result = run_collaboration(
            task="Write a story",
            model_cycle=["model-a", "model-b"],
            iterations=2,
            stop_on_convergence=False,
        )

    assert result.final_draft == "Beta follow-up response"
    assert [entry[0] for entry in result.history] == ["model-a", "model-b"]
    assert [entry[1] for entry in result.history] == [
        "Alpha turn response",
        "Beta follow-up response",
    ]
