from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import mock_open, patch

import pytest

from outomem.utils import (
    count_tokens,
    format_conversation,
    load_style_file,
    safe_json_parse,
    truncate_to_token_limit,
)


class TestFormatConversation:
    def test_list_of_dicts_valid(self) -> None:
        conv = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = format_conversation(conv)
        assert result == conv

    def test_list_of_dicts_invalid_role_raises(self) -> None:
        conv = [{"role": "invalid", "content": "hello"}]
        with pytest.raises(ValueError, match="Invalid role"):
            format_conversation(conv)

    def test_list_of_dicts_missing_keys_raises(self) -> None:
        conv = [{"role": "user"}]
        with pytest.raises(ValueError, match="must have 'role' and 'content'"):
            format_conversation(conv)

    def test_string_returns_raw_marker(self) -> None:
        text = "Hello\nHow are you"
        result = format_conversation(text)
        assert len(result) == 1
        assert result[0]["role"] == "raw"
        assert result[0]["content"] == "Hello\nHow are you"

    def test_string_empty_returns_empty_list(self) -> None:
        result = format_conversation("")
        assert result == []

    def test_string_single_line_returns_raw_marker(self) -> None:
        result = format_conversation("Hello")
        assert len(result) == 1
        assert result[0]["role"] == "raw"
        assert result[0]["content"] == "Hello"

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be list\\[dict\\] or str"):
            format_conversation(123)  # type: ignore[arg-type]


class TestCountTokens:
    def test_empty_string_returns_zero(self) -> None:
        assert count_tokens("") == 0

    def test_openai_model_uses_tiktoken(self) -> None:
        result = count_tokens("hello world", model="gpt-4")
        assert result > 0

    def test_non_openai_model_estimates(self) -> None:
        result = count_tokens("hello world", model="claude-3")
        assert result > 0

    def test_o1_model_uses_tiktoken(self) -> None:
        result = count_tokens("hello", model="o1-preview")
        assert result > 0


class TestTruncateToTokenLimit:
    def test_empty_text_returns_empty(self) -> None:
        result = truncate_to_token_limit("", max_tokens=10)
        assert result == ""

    def test_zero_max_tokens_returns_empty(self) -> None:
        result = truncate_to_token_limit("some text", max_tokens=0)
        assert result == ""

    def test_text_within_limit_unchanged(self) -> None:
        text = "short"
        result = truncate_to_token_limit(text, max_tokens=100, model="gpt-4")
        assert result == text

    def test_long_text_truncated(self) -> None:
        text = "a" * 1000
        result = truncate_to_token_limit(text, max_tokens=10, model="gpt-4")
        assert count_tokens(result, model="gpt-4") <= 10


class TestSafeJsonParse:
    def test_valid_json_dict(self) -> None:
        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_list(self) -> None:
        result = safe_json_parse('["a", "b"]')
        assert result == ["a", "b"]

    def test_json_with_code_fence(self) -> None:
        result = safe_json_parse('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_markdown_fence(self) -> None:
        result = safe_json_parse('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self) -> None:
        result = safe_json_parse("not json at all")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = safe_json_parse("")
        assert result is None

    def test_none_returns_none(self) -> None:
        result = safe_json_parse(None)  # type: ignore[arg-type]
        assert result is None

    def test_non_dict_list_returns_none(self) -> None:
        result = safe_json_parse('"just a string"')
        assert result is None


class TestLoadStyleFile:
    def test_load_existing_file(self) -> None:
        with patch("builtins.open", mock_open(read_data="style content")):
            result = load_style_file("/fake/path.md")
            assert result == "style content"

    def test_missing_file_returns_empty_when_not_required(self) -> None:
        result = load_style_file("/nonexistent/path.md")
        assert result == ""

    def test_empty_path_returns_empty(self) -> None:
        result = load_style_file("")
        assert result == ""

    def test_required_file_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_style_file("/nonexistent/path.md", required=True)
