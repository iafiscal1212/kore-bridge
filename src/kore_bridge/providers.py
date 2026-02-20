"""LLM providers. Runtime-agnostic: wrap any LLM as a callable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send messages, get response text."""
        ...

    @abstractmethod
    def summarize(self, text: str, instruction: str) -> str:
        """Summarize text with instruction. Used by reflect()."""
        ...


class CallableLLM(LLMProvider):
    """Wrap any callable as an LLM provider.

    Usage:
        llm = CallableLLM(lambda msgs: my_api_call(msgs))
    """

    def __init__(self, fn, summarize_fn=None) -> None:
        self._fn = fn
        self._summarize_fn = summarize_fn or self._default_summarize

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self._fn(messages)

    def summarize(self, text: str, instruction: str) -> str:
        return self._summarize_fn(text, instruction)

    def _default_summarize(self, text: str, instruction: str) -> str:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ]
        return self._fn(messages)


class OpenAIProvider(LLMProvider):
    """OpenAI provider. Requires: pip install kore-bridge[openai]"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install kore-bridge[openai]")
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
        )
        return resp.choices[0].message.content

    def summarize(self, text: str, instruction: str) -> str:
        return self.complete([
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ])


class AnthropicProvider(LLMProvider):
    """Anthropic provider. Requires: pip install kore-bridge[anthropic]"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929",
                 api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install kore-bridge[anthropic]")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        # Anthropic separa system de messages
        system = ""
        user_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_msgs.append(msg)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=user_msgs or [{"role": "user", "content": "Hello"}],
        )
        return resp.content[0].text

    def summarize(self, text: str, instruction: str) -> str:
        return self.complete([
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ])
