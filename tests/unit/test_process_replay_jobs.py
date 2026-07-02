from scripts.process_replay_jobs import canonical_messages, process_job


def base_job(messages):
    return {
        "prompt_id": "request-1",
        "messages": messages,
        "sampling": {"temperature": 0.2, "max_tokens": 9000},
        "prompt_tokens": 5000,
    }


def test_canonical_messages_discards_agent_trace_and_keeps_latest_user_task():
    messages = [
        {"role": "system", "content": "Follow the user request."},
        {"role": "user", "content": "Old task"},
        {"role": "assistant", "content": "Calling a tool"},
        {"role": "tool", "content": "not valid JSON"},
        {"role": "user", "content": "Current context"},
        {"role": "user", "content": [{"type": "text", "text": "Current task"}]},
    ]

    result, removed = canonical_messages(messages)

    assert result == [
        {"role": "system", "content": "Follow the user request."},
        {"role": "user", "content": "Current context\n\nCurrent task"},
    ]
    assert removed["tool_messages"] == 1
    assert removed["history_messages"] == 3


def test_process_job_rejects_selected_truncated_content():
    job = base_job(
        [{"role": "user", "content": "partial ... (litellm_truncated skipped 20 chars) ..."}]
    )

    result, reason, _ = process_job(job, 60_000, 16_384, 4096)

    assert result is None
    assert reason == "truncated_content"


def test_process_job_emits_bounded_text_only_job():
    job = base_job(
        [
            {"role": "system", "content": [{"type": "text", "text": "Be concise"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the equipment"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
    )

    result, reason, _ = process_job(job, 60_000, 16_384, 4096)

    assert reason is None
    assert result["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Describe the equipment"},
    ]
    assert result["sampling"]["max_tokens"] == 4096
    assert result["source_prompt_tokens"] == 5000
    assert result["processing"]["version"] == 2
