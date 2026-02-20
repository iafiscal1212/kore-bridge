"""Bridge: connects a Mind to an LLM. Observes, learns, evolves."""

from __future__ import annotations

from pathlib import Path

from kore_mind import Mind, Memory, Identity
from kore_mind.models import MemoryType

from kore_bridge.providers import LLMProvider

# Prompt templates
_REFLECT_INSTRUCTION = """You are analyzing memories of an AI assistant to generate its identity profile.

Given the following memories, extract:
1. A concise summary of who this assistant is and what it focuses on (2-3 sentences max)
2. Key personality traits as keywords with confidence scores (0.0-1.0)
3. Relationships: who does this assistant interact with and how

Respond in this exact JSON format:
{"summary": "...", "traits": {"trait": score, ...}, "relationships": {"name": "description", ...}}

Be concise. Only include clear patterns, not speculation."""

_CONTEXT_INSTRUCTION = """You are an AI assistant with persistent memory and evolving identity.

{identity}

Relevant memories:
{memories}

Use this context naturally. Don't explicitly mention "my memories say..." — just be informed by them."""


class Bridge:
    """Connects a Mind to any LLM. The cognitive bridge.

    Usage:
        bridge = Bridge(mind=Mind("agent.db"), llm=my_provider)
        response = bridge.think(prompt="Help me with my proof")
        bridge.observe("User seemed frustrated with the result")
    """

    def __init__(self, mind: Mind, llm: LLMProvider) -> None:
        self._mind = mind
        self._llm = llm

    @property
    def mind(self) -> Mind:
        return self._mind

    # ── think ──────────────────────────────────────────────────────────

    def think(self, prompt: str, user: str = "",
              system: str = "", remember: bool = True) -> str:
        """Think with context. Recall → inject → respond → observe.

        Args:
            prompt: what the user said
            user: user identifier (for relationship tracking)
            system: additional system prompt (merged with identity context)
            remember: whether to auto-remember this interaction
        """
        # 1. Recall relevant memories
        memories = self._mind.recall(prompt, limit=10)

        # 2. Build context
        identity = self._mind.identity()
        context = self._build_context(identity, memories, system)

        # 3. Call LLM
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.complete(messages)

        # 4. Auto-observe the interaction
        if remember:
            source = user or "conversation"
            self._mind.experience(
                content=f"User: {prompt[:200]}",
                type=MemoryType.EPISODIC,
                source=source,
                tags=["conversation", "user-input"],
            )
            self._mind.experience(
                content=f"Response: {response[:200]}",
                type=MemoryType.EPISODIC,
                source=source,
                tags=["conversation", "response"],
            )

        return response

    # ── observe ────────────────────────────────────────────────────────

    def observe(self, observation: str, source: str = "",
                type: str | MemoryType = MemoryType.SEMANTIC,
                tags: list[str] | None = None) -> Memory:
        """Manually observe something. Shortcut to mind.experience()."""
        return self._mind.experience(
            content=observation,
            type=type,
            source=source,
            tags=tags or [],
        )

    # ── reflect ────────────────────────────────────────────────────────

    def reflect(self) -> Identity:
        """Reflect using the LLM to generate identity. The magic operation."""
        import json

        def llm_summarizer(memories: list[Memory]) -> Identity:
            if not memories:
                return Identity(summary="No memories yet.")

            # Prepare memories text
            mem_text = "\n".join(
                f"- [{m.type.value}] (salience={m.salience:.2f}) {m.content}"
                for m in memories[:50]  # top 50 by salience
            )

            raw = self._llm.summarize(mem_text, _REFLECT_INSTRUCTION)

            # Parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                data = json.loads(raw.strip())
                return Identity(
                    summary=data.get("summary", ""),
                    traits=data.get("traits", {}),
                    relationships=data.get("relationships", {}),
                )
            except (json.JSONDecodeError, IndexError):
                # Fallback: use raw text as summary
                return Identity(summary=raw[:500])

        return self._mind.reflect(summarizer=llm_summarizer)

    # ── internals ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context(identity: Identity, memories: list[Memory],
                       extra_system: str) -> str:
        """Build system prompt from identity + memories."""
        parts = []

        if extra_system:
            parts.append(extra_system)

        identity_text = ""
        if identity.summary:
            identity_text = f"Your identity: {identity.summary}"
            if identity.traits:
                top = sorted(identity.traits.items(), key=lambda x: x[1], reverse=True)[:5]
                identity_text += f"\nKey traits: {', '.join(t[0] for t in top)}"

        mem_text = ""
        if memories:
            mem_lines = [f"- {m.content}" for m in memories[:10]]
            mem_text = "\n".join(mem_lines)

        context = _CONTEXT_INSTRUCTION.format(
            identity=identity_text or "No established identity yet.",
            memories=mem_text or "No relevant memories.",
        )

        if parts:
            context = "\n\n".join(parts) + "\n\n" + context

        return context

    def __repr__(self) -> str:
        return f"Bridge(mind={self._mind!r})"
