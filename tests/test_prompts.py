from llm_collaborator.prompts import build_messages


def test_build_messages_includes_task_and_draft():
    messages = build_messages(
        task="Do something",
        last_draft="Draft text",
        actor_name="Model A",
        critique_summaries=["Model B: Looks good"],
    )

    assert messages[0]["role"] == "system"
    user_message = messages[1]["content"]
    assert "<task>" in user_message
    assert "Draft text" in user_message
    assert "Model B: Looks good" in user_message
    assert "<critique>" in user_message
    assert "<draft>" in user_message
