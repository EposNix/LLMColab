from llm_collaborator.prompts import build_messages


def test_build_messages_includes_task_and_draft():
    messages = build_messages(
        task="Do something",
        last_draft="Draft text",
        actor_name="Model A",
        response_summaries=["Model B: Looks good"],
    )

    assert messages[0]["role"] == "system"
    user_message = messages[1]["content"]
    assert "<task>" in user_message
    assert "Draft text" in user_message
    assert "Model B: Looks good" in user_message
    assert "<recent_updates>" in user_message
    assert "updated contribution" in user_message
