from unittest.mock import patch

from llm_collaborator.llm_collaborator import (
    CollaborationResult,
    _extract_text,
    extract_sections,
    run_collaboration,
)

from litellm.utils import Choices, Message, ModelResponse, Usage


def test_extract_sections_parses_xml():
    critique, draft = extract_sections("<critique>Note</critique><draft>Body</draft>")
    assert critique == "Note"
    assert draft == "Body"


def test_run_collaboration_uses_round_robin_models():
    responses = [
        {
            "choices": [
                {
                    "message": {"content": "<critique>First</critique><draft>Draft1</draft>"}
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": "<critique>Second</critique><draft>Draft2</draft>"}
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
    assert result.final_draft == "Draft2"
    assert [entry[0] for entry in result.history] == ["model-a", "model-b"]
    assert [entry[1] for entry in result.history] == ["First", "Second"]


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
                    content="<critique>Note</critique><draft>Body</draft>",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    text = _extract_text(response)

    assert "<critique>Note</critique>" in text
    assert "<draft>Body</draft>" in text


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
                        content="<critique>Alpha</critique><draft>Draft1</draft>",
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
                        content="<critique>Beta</critique><draft>Draft2</draft>",
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

    assert result.final_draft == "Draft2"
    assert [entry[0] for entry in result.history] == ["model-a", "model-b"]
    assert [entry[1] for entry in result.history] == ["Alpha", "Beta"]
