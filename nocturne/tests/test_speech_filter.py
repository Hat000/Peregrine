"""Tests for the speech filter — the pure heart of what gets read aloud."""

from nocturne.speech_filter import (
    ProseFilter, describe_tool_use, summarize_tool_result, _clean_for_speech,
)


def _speak_all(chunks):
    pf = ProseFilter()
    items = []
    for c in chunks:
        items += pf.feed(c)
    items += pf.flush()
    return items


def test_sentences_stream_out_on_boundaries():
    items = _speak_all(["Hello there. ", "How are you? ", "Fine."])
    texts = [i.text for i in items if i.kind == "prose"]
    assert texts == ["Hello there.", "How are you?", "Fine."]


def test_abbreviation_does_not_split():
    items = _speak_all(["Use e.g. this pattern here. Done."])
    texts = [i.text for i in items if i.kind == "prose"]
    assert texts[0] == "Use e.g. this pattern here."
    assert texts[1] == "Done."


def test_decimal_does_not_split():
    items = _speak_all(["Pi is about 3.14 in value. Yes."])
    texts = [i.text for i in items if i.kind == "prose"]
    assert texts[0].startswith("Pi is about 3.14")


def test_code_block_is_announced_not_read():
    text = "Here is code:\n```python\nprint('hi')\nx = 1\n```\nThat's it."
    items = _speak_all([text])
    kinds = [i.kind for i in items]
    assert "code_announce" in kinds
    spoken = " ".join(i.text for i in items if i.kind == "prose")
    assert "print" not in spoken and "x = 1" not in spoken
    ann = next(i.text for i in items if i.kind == "code_announce")
    assert "2 lines" in ann


def test_partial_fence_across_chunks_suppressed():
    items = _speak_all(["Look:\n``", "`\ncode line\n``", "`\nOkay."])
    spoken = " ".join(i.text for i in items if i.kind == "prose")
    assert "code line" not in spoken
    assert any(i.kind == "code_announce" for i in items)


def test_markdown_is_stripped():
    assert _clean_for_speech("**bold** and `code` and [link](http://x)") == \
        "bold and code and link"
    assert _clean_for_speech("# Heading") == "Heading"
    assert _clean_for_speech("- a bullet") == "a bullet"


def test_paragraph_break_flushes():
    items = _speak_all(["First thought\n\nSecond thought"])
    texts = [i.text for i in items if i.kind == "prose"]
    assert texts == ["First thought", "Second thought"]


def test_tool_use_breadcrumbs():
    assert describe_tool_use("Edit", {"file_path": "/a/b/config.py"}) == "editing config.py"
    assert describe_tool_use("Write", {"file_path": "motor.py"}) == "writing motor.py"
    assert describe_tool_use("Read", {"file_path": "x/y.txt"}) == "reading y.txt"
    assert describe_tool_use("Bash", {"command": "pytest", "description": "Run the tests"}) == "run the tests"
    assert describe_tool_use("Bash", {"command": "ls"}) == "running a shell command"
    assert describe_tool_use("Grep", {"pattern": "x"}) == "searching the code"


def test_tool_result_summaries():
    assert summarize_tool_result("Bash", "... 12 passed in 0.3s") == "12 tests passed"
    assert summarize_tool_result("Bash", "3 passed, 2 failed") == "3 tests passed, 2 failed"
    assert "lines" in summarize_tool_result("Write", "wrote 42 lines to motor.py")
    long = "\n".join(f"line {i}" for i in range(50))
    assert "50 lines" in summarize_tool_result("Read", long)
    assert summarize_tool_result("Bash", "hello", is_error=True).startswith("that")


def test_short_plain_result_is_read():
    out = summarize_tool_result("Bash", "All good")
    assert out == "All good"
