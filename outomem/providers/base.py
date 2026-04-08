from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt and return the text response."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name."""
        ...
