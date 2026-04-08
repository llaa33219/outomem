"""Anthropic LLM provider implementation."""

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client: Anthropic = Anthropic(base_url=base_url, api_key=api_key)
        self._model: str = model

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt to the LLM and return the text response."""
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        response = self._client.messages.create(
            model=self._model,
            system=system_prompt or "",
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
        )
        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise TypeError(f"Expected TextBlock, got {type(block).__name__}")
        return block.text

    @property
    def model(self) -> str:
        """Return the model identifier."""
        return self._model
