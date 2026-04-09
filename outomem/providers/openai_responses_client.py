"""OpenAI Responses API provider."""

from __future__ import annotations

import time
from openai import OpenAI

from .base import LLMProvider


class OpenAIResponsesProvider(LLMProvider):
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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.responses.create(
                    model=self._model,
                    input=messages,
                )
                raw = ""
                for output in response.output:
                    if output.type == "message":
                        raw = output.content[0].text
                        break
                if raw:
                    import re

                    pattern = r"<think>.*?</think>"
                    stripped = re.sub(
                        pattern,
                        "",
                        raw,
                        flags=re.DOTALL,
                    ).strip()
                    return stripped if stripped else ""
                last_error = ValueError(f"Empty response (attempt {attempt + 1}/3)")
            except Exception as e:
                last_error = e
            if attempt < 2:
                time.sleep(2**attempt)
        raise last_error or RuntimeError("Provider call failed")

    @property
    def model(self) -> str:
        return self._model
