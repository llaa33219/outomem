"""Tests for prompts module."""

from __future__ import annotations

from outomem.prompts import (
    get_consolidation_prompt,
    get_context_synthesis_prompt,
    get_extraction_prompt,
    get_summarization_prompt,
)


class TestGetExtractionPrompt:
    def test_returns_system_and_user_prompts(self) -> None:
        sys_p, usr_p = get_extraction_prompt(
            conversation=[{"role": "user", "content": "I love coffee"}],
            style="friendly",
        )
        assert isinstance(sys_p, str)
        assert isinstance(usr_p, str)
        assert "I love coffee" in usr_p
        assert "friendly" in usr_p

    def test_handles_string_conversation(self) -> None:
        sys_p, usr_p = get_extraction_prompt(
            conversation="I love coffee",
            style="",
        )
        assert "I love coffee" in usr_p

    def test_empty_style_uses_placeholder(self) -> None:
        _, usr_p = get_extraction_prompt(
            conversation=[{"role": "user", "content": "test"}],
            style="",
        )
        assert "(no style context)" in usr_p


class TestGetConsolidationPrompt:
    def test_returns_prompts_with_memories(self) -> None:
        memories = [
            {"id": "1", "content": "likes coffee", "strength": 0.9},
            {"id": "2", "content": "loves tea", "strength": 0.8},
        ]
        sys_p, usr_p = get_consolidation_prompt(memories)
        assert isinstance(sys_p, str)
        assert isinstance(usr_p, str)
        assert "likes coffee" in usr_p
        assert "1" in usr_p

    def test_empty_memories_list(self) -> None:
        _, usr_p = get_consolidation_prompt([])
        assert "(none)" in usr_p


class TestGetSummarizationPrompt:
    def test_returns_prompts(self) -> None:
        sys_p, usr_p = get_summarization_prompt(
            style="friendly",
            memories=[{"id": "1", "content": "test", "strength": 0.5}],
            recent_history="user: hello",
        )
        assert isinstance(sys_p, str)
        assert isinstance(usr_p, str)
        assert "friendly" in usr_p
        assert "hello" in usr_p

    def test_empty_style_uses_placeholder(self) -> None:
        _, usr_p = get_summarization_prompt(
            style="",
            memories=[],
            recent_history="",
        )
        assert "(no style context)" in usr_p


class TestGetContextSynthesisPrompt:
    def test_returns_prompts(self) -> None:
        sys_p, usr_p = get_context_synthesis_prompt(
            conversation="user: I love coffee",
            style="friendly tone",
            personalization="likes coffee",
            long_term="developer",
            recent_events="had coffee yesterday",
        )
        assert isinstance(sys_p, str)
        assert isinstance(usr_p, str)
        assert "friendly tone" in usr_p
        assert "coffee" in usr_p

    def test_empty_personalization_uses_placeholder(self) -> None:
        _, usr_p = get_context_synthesis_prompt(
            conversation="",
            style="",
            personalization="",
            long_term="",
            recent_events="",
        )
        assert "(no preference memories)" in usr_p

    def test_empty_long_term_uses_placeholder(self) -> None:
        _, usr_p = get_context_synthesis_prompt(
            conversation="",
            style="",
            personalization="",
            long_term="",
            recent_events="",
        )
        assert "(no factual memories)" in usr_p

    def test_empty_recent_events_uses_placeholder(self) -> None:
        _, usr_p = get_context_synthesis_prompt(
            conversation="",
            style="",
            personalization="",
            long_term="",
            recent_events="",
        )
        assert "(no recent events)" in usr_p
