from __future__ import annotations

import pytest

from outomem.providers import create_provider


class TestCreateProvider:
    def test_creates_openai_provider(self) -> None:
        provider = create_provider(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4",
        )
        assert provider.__class__.__name__ == "OpenAIProvider"
        assert provider.model == "gpt-4"

    def test_creates_anthropic_provider(self) -> None:
        provider = create_provider(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="test-key",
            model="claude-3-opus",
        )
        assert provider.__class__.__name__ == "AnthropicProvider"
        assert provider.model == "claude-3-opus"

    def test_creates_google_provider(self) -> None:
        provider = create_provider(
            provider="google",
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            model="gemini-pro",
        )
        assert provider.__class__.__name__ == "GoogleProvider"
        assert "gemini-pro" in provider.model

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(
                provider="unknown",
                base_url="",
                api_key="",
                model="",
            )
