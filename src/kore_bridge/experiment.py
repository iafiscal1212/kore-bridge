"""A/B Testing: compare two LLM providers/strategies with the same queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from kore_mind import Mind

from kore_bridge.bridge import Bridge
from kore_bridge.providers import LLMProvider


@dataclass
class ExperimentResult:
    """Result of a single A/B experiment run."""

    prompt: str
    variant_a: str
    variant_b: str
    time_a_ms: float
    time_b_ms: float

    @property
    def faster(self) -> str:
        """Which variant was faster: 'a' or 'b'."""
        return "a" if self.time_a_ms <= self.time_b_ms else "b"


class Experiment:
    """Run A/B tests comparing two LLM providers through Bridge.

    Usage:
        exp = Experiment(mind, provider_a, provider_b)
        result = exp.run("Explain quantum computing")
        print(result.variant_a, result.variant_b, result.faster)
    """

    def __init__(self, mind: Mind,
                 variant_a: LLMProvider,
                 variant_b: LLMProvider,
                 remember: bool = False) -> None:
        self._mind = mind
        self._variant_a = variant_a
        self._variant_b = variant_b
        self._remember = remember

    def run(self, prompt: str, system: str = "",
            user: str = "") -> ExperimentResult:
        """Run same prompt through both variants. Returns ExperimentResult."""
        # Variant A
        bridge_a = Bridge(mind=self._mind, llm=self._variant_a)
        t0 = time.time()
        response_a = bridge_a.think(
            prompt, user=user, system=system,
            remember=self._remember, use_cache=False,
        )
        time_a = (time.time() - t0) * 1000

        # Variant B
        bridge_b = Bridge(mind=self._mind, llm=self._variant_b)
        t0 = time.time()
        response_b = bridge_b.think(
            prompt, user=user, system=system,
            remember=self._remember, use_cache=False,
        )
        time_b = (time.time() - t0) * 1000

        return ExperimentResult(
            prompt=prompt,
            variant_a=response_a,
            variant_b=response_b,
            time_a_ms=time_a,
            time_b_ms=time_b,
        )

    def run_batch(self, prompts: list[str], system: str = "",
                  user: str = "") -> list[ExperimentResult]:
        """Run multiple prompts through both variants."""
        return [self.run(p, system=system, user=user) for p in prompts]
