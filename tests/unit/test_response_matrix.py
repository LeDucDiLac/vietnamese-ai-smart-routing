from scripts.build_response_matrix import sanitize_messages


def test_sanitize_messages_drops_truncated_image_payload_without_mutating_input():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,abc... (litellm_truncated) ...xyz"
                    },
                },
            ],
        }
    ]

    result = sanitize_messages(messages)

    assert result == [
        {"role": "user", "content": [{"type": "text", "text": "Describe this image"}]}
    ]
    assert len(messages[0]["content"]) == 2


def test_sanitize_messages_preserves_plain_text_and_metadata():
    messages = [
        {"role": "system", "content": "Be concise", "name": "policy"},
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    assert sanitize_messages(messages) == messages
