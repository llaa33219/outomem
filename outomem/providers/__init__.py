"""LLM provider implementations."""

from outomem.providers.base import LLMProvider


def create_provider(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
) -> LLMProvider:
    """Factory function to create LLM provider."""
    if provider == "openai":
        from outomem.providers.openai_client import OpenAIProvider

        return OpenAIProvider(base_url=base_url, api_key=api_key, model=model)
    elif provider == "anthropic":
        from outomem.providers.anthropic_client import AnthropicProvider

        return AnthropicProvider(base_url=base_url, api_key=api_key, model=model)
    elif provider == "google":
        from outomem.providers.google_client import GoogleProvider

        return GoogleProvider(base_url=base_url, api_key=api_key, model=model)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Must be one of: 'openai', 'anthropic', 'google'"
        )


__all__ = ["LLMProvider", "create_provider"]
