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


# ── SC Router ──────────────────────────────────────────────────────────

# Default mapping: SC level → provider key
_DEFAULT_SC_MAPPING: dict[int, str] = {
    0: "fast",
    1: "fast",
    2: "quality",
    3: "quality",
}


class SCRouterProvider(RouterProvider):
    """Routes queries using Selector Complexity theory.

    Requires: pip install kore-bridge[sc]

    Instead of heuristics (message length), uses formal proof complexity
    classification (SC 0-3) to decide which LLM handles each query.

    Usage:
        from sc_router import ToolCatalog, Tool

        catalog = ToolCatalog()
        catalog.register(Tool(name="search", ...))

        router = SCRouterProvider(
            providers={"fast": ollama, "quality": openai},
            catalog=catalog,
        )
        bridge = Bridge(mind=mind, llm=router)
    """

    def __init__(self, providers: dict[str, LLMProvider],
                 catalog: object,
                 sc_mapping: dict[int, str] | None = None,
                 default: str | None = None) -> None:
        self._catalog = catalog
        self._sc_mapping = sc_mapping or _DEFAULT_SC_MAPPING
        self._last_sc_level: int | None = None
        self._last_classification: dict | None = None

        def _sc_route(messages: list[dict[str, str]]) -> str:
            try:
                from sc_router import route
            except ImportError:
                raise ImportError("pip install kore-bridge[sc]")

            query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

            result = route(query, self._catalog)
            self._last_sc_level = result.sc_level
            self._last_classification = result.classification
            return self._sc_mapping.get(result.sc_level, "quality")

        super().__init__(
            providers=providers,
            route_fn=_sc_route,
            default=default,
        )

    @property
    def last_sc_level(self) -> int | None:
        """SC level de la última clasificación."""
        return self._last_sc_level

    @property
    def last_classification(self) -> dict | None:
        """Clasificación completa de la última query."""
        return self._last_classification
