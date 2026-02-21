"""Bridge: connects a Mind to an LLM. Cognitive middleware."""

from __future__ import annotations

import hashlib
import re
import time

from kore_mind import Mind, Memory, Identity
from kore_mind.models import CacheEntry, MemoryType

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


def _normalize_query(text: str) -> str:
    """Normaliza query para hash: lowercase + colapsar whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _query_hash(text: str) -> str:
    """Hash normalizado: sha256[:16]."""
    normalized = _normalize_query(text)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class Bridge:
    """Connects a Mind to any LLM. The cognitive bridge.

    Usage:
        bridge = Bridge(mind=Mind("agent.db"), llm=my_provider)
        response = bridge.think(prompt="Help me with my proof")
        bridge.observe("User seemed frustrated with the result")

    v0.2 features:
        - Smart cache (cache_ttl)
        - Rate limiting (rate_limit, rate_window)
        - Per-user source filtering
        - Observability traces
    """

    def __init__(self, mind: Mind, llm: LLMProvider,
                 cache_ttl: float = 3600.0,
                 rate_limit: int = 0,
                 rate_window: float = 3600.0) -> None:
        self._mind = mind
        self._llm = llm
        self._cache_ttl = cache_ttl
        self._rate_limit = rate_limit
        self._rate_window = rate_window

    @property
    def mind(self) -> Mind:
        return self._mind

    # ── think ──────────────────────────────────────────────────────────

    def think(self, prompt: str, user: str = "",
              system: str = "", remember: bool = True,
              use_cache: bool = True) -> str:
        """Think with context. Cache → Rate check → Recall → LLM → Observe.

        Args:
            prompt: what the user said
            user: user identifier (for relationship tracking + source filtering)
            system: additional system prompt (merged with identity context)
            remember: whether to auto-remember this interaction
            use_cache: whether to check/store in cache (True by default)
        """
        t0 = time.time()
        source = user or "conversation"
        qhash = _query_hash(prompt)
        cache_hit = False
        memories_used = 0

        # 1. Check cache
        if use_cache and self._cache_ttl > 0:
            entry = self._mind._storage.find_cache_by_hash(qhash, source=source)
            if entry is not None:
                # Check TTL
                if (time.time() - entry.created_at) < entry.ttl:
                    self._mind._storage.cache_hit(entry.id)
                    cache_hit = True
                    self._trace_think(prompt, entry.response, source,
                                      t0, cache_hit=True, memories_used=0)
                    return entry.response

        # 2. Rate limiting — si preguntó lo mismo muchas veces, usar cache/memoria
        if self._rate_limit > 0:
            count = self._mind._storage.query_count(
                qhash, source=source, window=self._rate_window,
            )
            if count >= self._rate_limit:
                # Try cache (any source)
                entry = self._mind._storage.find_cache_by_hash(qhash)
                if entry is not None:
                    self._mind._storage.cache_hit(entry.id)
                    self._trace_think(prompt, entry.response, source,
                                      t0, cache_hit=True, memories_used=0,
                                      rate_limited=True)
                    return entry.response
                # Fallback: best memory match
                mems = self._mind.recall(prompt, limit=1, source=source)
                if mems:
                    fallback = mems[0].content
                    self._trace_think(prompt, fallback, source,
                                      t0, cache_hit=False, memories_used=1,
                                      rate_limited=True)
                    return fallback

        # 3. Log query for rate limiting
        if self._rate_limit > 0:
            self._mind._storage.log_query(qhash, source=source)

        # 4. Recall relevant memories (filtered by user/source)
        memories = self._mind.recall(prompt, limit=10, source=source)
        memories_used = len(memories)

        # 5. Build context
        identity = self._mind.identity()
        context = self._build_context(identity, memories, system)

        # 6. Call LLM
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.complete(messages)

        # 7. Save to cache
        if use_cache and self._cache_ttl > 0:
            entry = CacheEntry(
                query=prompt,
                response=response,
                query_hash=qhash,
                source=source,
                ttl=self._cache_ttl,
            )
            self._mind._storage.save_cache_entry(entry)

        # 8. Auto-observe the interaction
        if remember:
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

        self._trace_think(prompt, response, source, t0,
                          cache_hit=False, memories_used=memories_used)
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

    def _trace_think(self, prompt: str, response: str, source: str,
                     t0: float, cache_hit: bool = False,
                     memories_used: int = 0,
                     rate_limited: bool = False) -> None:
        """Traza para think() — usa el sistema de traces de Mind."""
        if not self._mind._enable_traces:
            return
        from kore_mind.models import Trace
        duration_ms = (time.time() - t0) * 1000
        trace = Trace(
            operation="bridge.think",
            input_text=prompt[:500],
            output_text=response[:500],
            source=source,
            duration_ms=duration_ms,
            metadata={
                "cache_hit": cache_hit,
                "memories_used": memories_used,
                "rate_limited": rate_limited,
            },
        )
        self._mind._storage.save_trace(trace)

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
