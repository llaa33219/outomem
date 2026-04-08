# LLM Providers API Reference

This document details the LLM provider interface and its implementations in `outomem`.

## LLMProvider Abstract Base
The `LLMProvider` class in `outomem/providers/base.py` defines the interface for all language model clients.

- `complete(prompt: str, system_prompt: str | None) -> str`: Sends a prompt and returns the text response.
- `model`: Abstract property returning the model name string.

Implementation: [base.py:4-14](outomem/providers/base.py)

## Factory Function
The `create_provider` function in `outomem/providers/__init__.py` instantiates the correct client based on the provider name.

```python
def create_provider(provider: str, base_url: str, api_key: str, model: str) -> LLMProvider
```

Supported providers: `"openai"`, `"anthropic"`, `"google"`.

Implementation: [__init__.py:6-28](outomem/providers/__init__.py)

## OpenAIProvider
Located in `outomem/providers/openai_client.py`.

- **Configuration**: Uses `temperature=0.0`.
- **Retry Logic**: Implements 3 retries with exponential backoff (`2**attempt`).
- **Thinking Blocks**: Automatically strips `<think>...</think>` tags using `re.sub` if the response starts with a thinking block.
- **Error Handling**: Catches general exceptions and raises the last encountered error or a `RuntimeError`.

Implementation: [openai_client.py:9-61](outomem/providers/openai_client.py)

## AnthropicProvider
Located in `outomem/providers/anthropic_client.py`.

- **Configuration**: Uses `temperature=0.0` and `max_tokens=4096`.
- **System Prompt**: Passed via the `system` parameter in the messages API.
- **Response Handling**: Extracts text from the first `TextBlock` in the response.

Implementation: [anthropic_client.py:9-35](outomem/providers/anthropic_client.py)

## GoogleProvider
Located in `outomem/providers/google_client.py`.

- **Configuration**: Uses `temperature=0.0` and `max_output_tokens=4096`.
- **System Instruction**: Passed via `system_instruction` in the `GenerateContentConfigDict`.
- **API Structure**: Uses `google.genai` client with `models.generate_content`.

Implementation: [google_client.py:9-42](outomem/providers/google_client.py)

## Adding a New Provider
To add a new LLM provider, follow these steps:

1. **Create Client**: Add a new file in `outomem/providers/` (e.g., `myservice_client.py`).
2. **Implement Interface**: Inherit from `LLMProvider` and implement `complete` and `model`.
3. **Handle Errors**: Implement appropriate retry logic and error catching within `complete`.
4. **Register Factory**: Update `create_provider` in `outomem/providers/__init__.py` to include the new provider.
5. **Update Exports**: Add the new class to `__all__` if necessary.
