from __future__ import annotations

from google import genai
from google.genai import types

from .base import LLMProvider


class GoogleProvider(LLMProvider):
    _client: genai.Client
    _model_name: str

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        contents: list[types.ContentDict] = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        config: types.GenerateContentConfigDict = {
            "temperature": 0.0,
            "max_output_tokens": 4096,
        }
        if system_prompt:
            config["system_instruction"] = {"parts": [{"text": system_prompt}]}

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        return response.text

    @property
    def model(self) -> str:
        return self._model_name
