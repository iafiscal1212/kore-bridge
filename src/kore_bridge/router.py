"""LLM Router: route queries to different providers based on complexity."""

from __future__ import annotations

from typing import Callable

from kore_bridge.providers import LLMProvider


def _default_route_fn(messages: list[dict[str, str]]) -> str:
    """Default routing: short messages → fast, long → quality."""
    total_len = sum(len(m.get("content", "")) for m in messages)
    return "fast" if total_len < 500 else "quality"


class RouterProvider(LLMProvider):
    """Routes queries to different LLM providers based on a routing function.

    Usage:
        router = RouterProvider(
            providers={"fast": ollama, "quality": openai},
            route_fn=my_router,  # optional, default: by message length
        )
        bridge = Bridge(mind=mind, llm=router)

    The route_fn receives messages and returns a provider key.
    Summarize always routes to "quality" if available.
    """

    def __init__(self, providers: dict[str, LLMProvider],
                 route_fn: Callable[[list[dict[str, str]]], str] | None = None,
                 default: str | None = None) -> None:
        if not providers:
            raise ValueError("At least one provider required")
        self._providers = providers
        self._route_fn = route_fn or _default_route_fn
        self._default = default or next(iter(providers))
        self._last_route: str | None = None

    @property
    def last_route(self) -> str | None:
        """Key del último provider usado."""
        return self._last_route

    def complete(self, messages: list[dict[str, str]]) -> str:
        key = self._route_fn(messages)
        if key not in self._providers:
            key = self._default
        self._last_route = key
        return self._providers[key].complete(messages)

    def summarize(self, text: str, instruction: str) -> str:
        # Summarize siempre va a quality si existe
        key = "quality" if "quality" in self._providers else self._default
        self._last_route = key
        return self._providers[key].summarize(text, instruction)
