from __future__ import annotations

import json
import os
import re
from typing import Any

import tiktoken

_VALID_ROLES = {"user", "assistant", "system"}

_cl100k_encoding: tiktoken.Encoding | None = None


def _get_cl100k() -> tiktoken.Encoding:
    global _cl100k_encoding
    if _cl100k_encoding is None:
        _cl100k_encoding = tiktoken.get_encoding("cl100k_base")
    return _cl100k_encoding


def format_conversation(
    conversation: list[dict[str, str]] | str,
) -> list[dict[str, str]]:
    """Convert various input formats into standardized list[dict] with role/content keys.

    Args:
        conversation: Either a list of message dicts or a string.
            - list[dict]: validated and returned (must have 'role' and 'content').
            - str: split by newlines, alternating user/assistant starting with user.

    Returns:
        list[dict] with 'role' and 'content' keys per message.
    """
    if isinstance(conversation, str):
        return _parse_string_conversation(conversation)

    if isinstance(conversation, list):
        return _validate_message_list(conversation)

    raise TypeError(
        f"conversation must be list[dict] or str, got {type(conversation).__name__}"
    )


def _parse_string_conversation(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    roles = ("user", "assistant")
    return [{"role": roles[i % 2], "content": line} for i, line in enumerate(lines)]


def _validate_message_list(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise TypeError(f"Each message must be a dict, got {type(msg).__name__}")
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' keys")
        role = msg["role"]
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid role '{role}', must be one of {_VALID_ROLES}")
        result.append({"role": role, "content": msg["content"]})
    return result


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text.

    Args:
        text: The text to count tokens for.
        model: Model name. OpenAI models use tiktoken; others estimate ~4 chars/token.

    Returns:
        Token count as int.
    """
    if not text:
        return 0

    if _is_openai_model(model):
        enc = _get_cl100k()
        return len(enc.encode(text))

    return max(1, len(text) // 4)


def _is_openai_model(model: str) -> bool:
    model_lower = model.lower()
    openai_prefixes = (
        "gpt-",
        "o1",
        "o3",
        "o4",
        "text-",
        "davinci",
        "curie",
        "babbage",
        "ada",
    )
    return any(model_lower.startswith(p) for p in openai_prefixes)


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to stay within a token limit.

    Tries to cut at sentence boundaries when possible.

    Args:
        text: Text to truncate.
        max_tokens: Maximum number of tokens allowed.
        model: Model name for token counting.

    Returns:
        Truncated text within the token limit.
    """
    if not text or max_tokens <= 0:
        return ""

    current_count = count_tokens(text, model)
    if current_count <= max_tokens:
        return text

    if _is_openai_model(model):
        return _truncate_tiktoken(text, max_tokens)

    char_limit = max_tokens * 4
    truncated = text[:char_limit]

    last_period = truncated.rfind(".")
    if last_period > len(truncated) // 2:
        truncated = truncated[: last_period + 1]

    return truncated


def _truncate_tiktoken(text: str, max_tokens: int) -> str:
    enc = _get_cl100k()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = enc.decode(tokens[:max_tokens])

    last_period = truncated.rfind(".")
    if last_period > len(truncated) // 2:
        truncated = truncated[: last_period + 1]

    return truncated


def safe_json_parse(text: str) -> dict[str, Any] | list[Any] | None:
    """Safely parse JSON from LLM responses, handling markdown code blocks.

    Args:
        text: Raw text that may contain JSON, possibly wrapped in ```json blocks.

    Returns:
        Parsed dict/list, or None on failure.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed: Any = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed  # type: ignore[no-any-return]
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def load_style_file(style_path: str, required: bool = False) -> str:
    """Load and return contents of a style.md file.

    Args:
        style_path: Path to the style file.
        required: If True, raises FileNotFoundError when file is missing.

    Returns:
        File contents as string, or empty string if not required and file doesn't exist.

    Raises:
        FileNotFoundError: If required=True and file is missing.
    """
    if not style_path:
        if required:
            raise FileNotFoundError("style_path is required but was empty")
        return ""

    path = os.path.expanduser(style_path)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, IOError) as e:
        if required:
            raise FileNotFoundError(f"style.md not found at {path}: {e}") from e
        return ""
