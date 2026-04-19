"""
CustomJSONWithParts: like CustomJSON, but allows assistant message content to be
either a plain string OR a list of typed parts (for tool-use conversations).

Each part is a dict {"type": "text"|"python"|"python_output", "text": "..."}.
This matches the structure used by tasks/gsm8k.py:60-77 and consumed by
tokenizer.render_conversation() in nanochat/tokenizer.py:309-345.
"""

import os
import json
from tasks.common import Task

ALLOWED_PART_TYPES = {"text", "python", "python_output"}


class CustomJSONWithParts(Task):
    """
    Load conversations from a JSONL file. Like CustomJSON, but assistant content
    can be a list of {"type":..., "text":...} parts in addition to plain string.
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        if not os.path.exists(filepath):
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print(f"This Task is empty until {filepath} is created.")
            print("-" * 80)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    messages = json.loads(line)
                    assert isinstance(messages, list), \
                        f"Line {line_no}: expected list, got {type(messages)}"
                    assert len(messages) >= 2, \
                        f"Line {line_no}: must have >=2 messages, got {len(messages)}"
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Line {line_no} message {i} missing 'role'"
                        assert "content" in message, f"Line {line_no} message {i} missing 'content'"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, \
                            f"Line {line_no} message {i}: role={message['role']!r}, expected {expected_role!r}"
                        content = message["content"]
                        if message["role"] == "user":
                            assert isinstance(content, str), \
                                f"Line {line_no} message {i}: user content must be str"
                        else:  # assistant
                            if isinstance(content, str):
                                pass  # OK
                            elif isinstance(content, list):
                                for j, part in enumerate(content):
                                    assert isinstance(part, dict), \
                                        f"Line {line_no} message {i} part {j}: must be dict"
                                    assert "type" in part and "text" in part, \
                                        f"Line {line_no} message {i} part {j}: missing 'type' or 'text'"
                                    assert part["type"] in ALLOWED_PART_TYPES, \
                                        f"Line {line_no} message {i} part {j}: type {part['type']!r} not in {ALLOWED_PART_TYPES}"
                                    assert isinstance(part["text"], str), \
                                        f"Line {line_no} message {i} part {j}: text must be str"
                            else:
                                raise AssertionError(
                                    f"Line {line_no} message {i}: assistant content must be str or list, got {type(content)}"
                                )
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}
