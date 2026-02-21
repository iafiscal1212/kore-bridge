"""kore-bridge: LLM integration layer for kore-mind."""

from kore_bridge.bridge import Bridge
from kore_bridge.providers import LLMProvider, CallableLLM, OllamaProvider
from kore_bridge.router import RouterProvider, SCRouterProvider
from kore_bridge.experiment import Experiment, ExperimentResult

__version__ = "0.3.0"
__all__ = [
    "Bridge", "LLMProvider", "CallableLLM", "OllamaProvider",
    "RouterProvider", "SCRouterProvider", "Experiment", "ExperimentResult",
]
