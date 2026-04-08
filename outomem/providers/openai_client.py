from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    _client: OpenAI
    _model: str

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        import time

        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.0,
                )
                if response.choices and response.choices[0].message is not None:
                    raw = response.choices[0].message.content
                    if raw is not None:
                        if raw.startswith("<think>"):
                            import re

                            stripped = re.sub(
                                r"<think>.*?</think>",
                                "",
                                raw,
                                flags=re.DOTALL,
                            ).strip()
                            return stripped if stripped else ""
                        return raw
                last_error = ValueError(f"Empty response (attempt {attempt + 1}/3)")
            except Exception as e:
                last_error = e
            if attempt < 2:
                time.sleep(2**attempt)
        raise last_error or RuntimeError("Provider call failed")

    @property
    def model(self) -> str:
        return self._model
