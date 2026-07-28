"""A stand-in for the Claude Code CLI in stream-json mode, for tests.

Reads newline-delimited JSON user messages on stdin and emits canned stream-json
output (deltas + a tool use/result + a result) per turn. Ignores all CLI flags.
"""

import json
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    emit({"type": "system", "subtype": "init", "session_id": "fake-123"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "control_request":
            continue
        text = ""
        for b in msg.get("message", {}).get("content", []):
            if isinstance(b, dict) and b.get("type") == "text":
                text = b.get("text", "")
        emit({"type": "stream_event", "event": {"type": "content_block_delta",
              "delta": {"type": "text_delta", "text": "You said "}}})
        emit({"type": "stream_event", "event": {"type": "content_block_delta",
              "delta": {"type": "text_delta", "text": text + ". "}}})
        emit({"type": "assistant", "message": {"content": [
              {"type": "tool_use", "id": "t1", "name": "Bash",
               "input": {"description": "Run the tests"}}]}})
        emit({"type": "user", "message": {"content": [
              {"type": "tool_result", "tool_use_id": "t1",
               "content": "12 passed in 0.3s", "is_error": False}]}})
        emit({"type": "result", "subtype": "success", "session_id": "fake-123",
              "usage": {"input_tokens": 5, "output_tokens": 2},
              "total_cost_usd": 0.0, "is_error": False})


if __name__ == "__main__":
    main()
