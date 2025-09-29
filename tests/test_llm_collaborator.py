from unittest.mock import patch

from llm_collaborator.llm_collaborator import (
    CollaborationResult,
    extract_sections,
    run_collaboration,
)


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
