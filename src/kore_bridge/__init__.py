"""kore-bridge: LLM integration layer for kore-mind."""

from kore_bridge.bridge import Bridge
from kore_bridge.providers import LLMProvider, CallableLLM, OllamaProvider

__version__ = "0.1.0"
__all__ = ["Bridge", "LLMProvider", "CallableLLM", "OllamaProvider"]
