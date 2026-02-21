"""Tests for A/B testing (v0.2)."""

import os
import tempfile

import pytest

from kore_mind import Mind
from kore_bridge import CallableLLM, Experiment


def make_provider(prefix: str) -> CallableLLM:
    return CallableLLM(
        lambda msgs, p=prefix: f"{p}: {msgs[-1]['content']}"
    )


@pytest.fixture
def mind():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind(path)
    yield m
    m.close()
    os.unlink(path)


class TestExperiment:
    def test_run_returns_both_variants(self, mind):
        exp = Experiment(
            mind,
            variant_a=make_provider("A"),
            variant_b=make_provider("B"),
        )
        result = exp.run("Hello")
        assert "A:" in result.variant_a
        assert "B:" in result.variant_b
        assert result.prompt == "Hello"

    def test_faster_property(self, mind):
        exp = Experiment(
            mind,
            variant_a=make_provider("A"),
            variant_b=make_provider("B"),
        )
        result = exp.run("Test")
        assert result.faster in ("a", "b")

    def test_run_batch(self, mind):
        exp = Experiment(
            mind,
            variant_a=make_provider("A"),
            variant_b=make_provider("B"),
        )
        results = exp.run_batch(["Q1", "Q2", "Q3"])
        assert len(results) == 3
        assert all(r.variant_a and r.variant_b for r in results)

    def test_no_remember_by_default(self, mind):
        exp = Experiment(
            mind,
            variant_a=make_provider("A"),
            variant_b=make_provider("B"),
        )
        exp.run("Secret test")
        assert mind.count == 0

    def test_remember_when_enabled(self, mind):
        exp = Experiment(
            mind,
            variant_a=make_provider("A"),
            variant_b=make_provider("B"),
            remember=True,
        )
        exp.run("Remember this")
        assert mind.count > 0

    def test_timing_is_positive(self, mind):
        exp = Experiment(
            mind,
            variant_a=make_provider("A"),
            variant_b=make_provider("B"),
        )
        result = exp.run("Timing test")
        assert result.time_a_ms >= 0
        assert result.time_b_ms >= 0
