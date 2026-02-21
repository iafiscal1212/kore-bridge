# kore-bridge

LLM integration layer for [kore-mind](https://github.com/iafiscal1212/kore-mind). Runtime-agnostic cognitive bridge.

**Middleware cognitivo entre tu app y el LLM.** Cache, routing, rate limiting, A/B testing — todo antes de gastar tokens.

Part of [**kore-stack**](https://github.com/iafiscal1212/kore-stack) — the complete cognitive middleware for LLMs. `pip install kore-stack` for the full stack, or install individually:

## Install

```bash
pip install kore-bridge                # core (zero deps beyond kore-mind)
pip install kore-bridge[openai]        # + OpenAI
pip install kore-bridge[anthropic]     # + Anthropic
pip install kore-bridge[sc]            # + Selector Complexity routing
pip install kore-bridge[all]           # everything
```

## Quick start (Ollama — 100% local, zero API keys)

```bash
ollama pull llama3.2
```

```python
from kore_mind import Mind
from kore_bridge import Bridge, OllamaProvider

mind = Mind("agent.db")
llm = OllamaProvider(model="llama3.2")  # local, free, private
bridge = Bridge(mind=mind, llm=llm)

# Think with context (auto-remembers)
response = bridge.think("Help me with my proof", user="carlos")

# Observe something
bridge.observe("User prefers concise answers")

# Reflect: LLM generates emergent identity from memories
identity = bridge.reflect()
print(identity.summary)
```

## v0.2 Features

### Smart Cache

If the LLM already answered something similar, don't spend tokens.

```python
bridge = Bridge(mind=mind, llm=llm, cache_ttl=3600.0)

r1 = bridge.think("What is P vs NP?")   # calls LLM
r2 = bridge.think("What is P vs NP?")   # cache hit, no LLM call

# Bypass cache when needed
r3 = bridge.think("What is P vs NP?", use_cache=False)  # forces LLM
```

Cache is per-user: different users get different cached responses.

### Rate Limiting

If the user asked the same thing 3 times, respond from memory.

```python
bridge = Bridge(
    mind=mind, llm=llm,
    rate_limit=3,       # max 3 identical queries per window
    rate_window=3600.0, # 1 hour window
)

# 4th identical query → served from cache/memory, no LLM call
```

### Per-user Filtering

Each user gets their own context. The `user` parameter in `think()` filters memories by source.

```python
bridge.think("Help with Python", user="alice")  # alice's memories
bridge.think("Help with Rust", user="bob")       # bob's memories
```

### LLM Routing

Simple queries to local Ollama (free), complex ones to GPT-4 (powerful).

```python
from kore_bridge import RouterProvider, OllamaProvider
from kore_bridge.providers import OpenAIProvider

router = RouterProvider(
    providers={
        "fast": OllamaProvider(model="llama3.2"),
        "quality": OpenAIProvider(model="gpt-4o"),
    },
    # Optional: custom routing logic
    route_fn=lambda msgs: "fast" if len(msgs[-1]["content"]) < 100 else "quality",
)

bridge = Bridge(mind=mind, llm=router)
bridge.think("Hi")                    # → Ollama (fast)
bridge.think("Explain quantum...")    # → GPT-4 (quality)
print(router.last_route)             # "quality"
```

Summarize (used by `reflect()`) always routes to "quality".

### SC Routing (Selector Complexity)

Routing based on formal proof complexity theory. Not heuristics — mathematics.

```bash
pip install kore-bridge[sc]
```

```python
from kore_bridge import SCRouterProvider, OllamaProvider
from kore_bridge.providers import OpenAIProvider
from sc_router import ToolCatalog, Tool

catalog = ToolCatalog()
catalog.register(Tool(
    name="calculator",
    description="Arithmetic calculations",
    input_types={"expression"},
    output_types={"number"},
    capability_tags={"math", "calculate"},
))

router = SCRouterProvider(
    providers={
        "fast": OllamaProvider(model="llama3.2"),
        "quality": OpenAIProvider(model="gpt-4o"),
    },
    catalog=catalog,
)

bridge = Bridge(mind=mind, llm=router)
bridge.think("What is 2+2?")              # SC(0) → Ollama
bridge.think("Analyze and cross-reference market data")  # SC(2+) → GPT-4

print(router.last_sc_level)       # 0, 1, 2, or 3
print(router.last_classification) # full classification evidence
```

### A/B Testing

Compare two providers with the same query and context.

```python
from kore_bridge import Experiment

exp = Experiment(
    mind,
    variant_a=OllamaProvider(model="llama3.2"),
    variant_b=OllamaProvider(model="mistral"),
)

result = exp.run("Explain recursion")
print(result.variant_a)      # llama3.2's response
print(result.variant_b)      # mistral's response
print(result.faster)         # "a" or "b"
print(f"A: {result.time_a_ms:.0f}ms, B: {result.time_b_ms:.0f}ms")

# Batch comparison
results = exp.run_batch(["Q1", "Q2", "Q3"])
```

By default, `remember=False` — experiments don't contaminate memory.

### Observability

Full tracing when `enable_traces=True` on the Mind.

```python
mind = Mind("agent.db", enable_traces=True)
bridge = Bridge(mind=mind, llm=llm)

bridge.think("Hello", user="carlos")

traces = mind.traces(operation="bridge.think")
for t in traces:
    print(f"{t.operation} | {t.duration_ms:.0f}ms | cache_hit={t.metadata['cache_hit']}")
```

## Providers

```python
# Ollama (local, recommended for OSS)
from kore_bridge import OllamaProvider
llm = OllamaProvider(model="llama3.2")

# OpenAI
from kore_bridge.providers import OpenAIProvider
llm = OpenAIProvider(model="gpt-4o-mini")

# Anthropic
from kore_bridge.providers import AnthropicProvider
llm = AnthropicProvider(model="claude-sonnet-4-5-20250929")

# Any callable
from kore_bridge import CallableLLM
llm = CallableLLM(lambda msgs: my_custom_api(msgs))

# Router (multiple providers)
from kore_bridge import RouterProvider
llm = RouterProvider(providers={"fast": ollama, "quality": openai})
```

## Backward compatibility

All new parameters have defaults that preserve v0.1 behavior:

```python
# This works exactly the same as v0.1
bridge = Bridge(mind=mind, llm=llm)
bridge.think("Hello")
```

## Part of kore-stack

| Package | What it does |
|---------|-------------|
| [kore-mind](https://github.com/iafiscal1212/kore-mind) | Memory, identity, traces, cache storage |
| **kore-bridge** (this) | LLM integration, cache logic, rate limiting, A/B testing, SC routing |
| [sc-router](https://github.com/iafiscal1212/sc-router) | Query routing by Selector Complexity theory |
| [**kore-stack**](https://github.com/iafiscal1212/kore-stack) | All of the above, one install: `pip install kore-stack` |

## Demo

```bash
python examples/demo_llm.py              # uses llama3.2
python examples/demo_llm.py mistral      # uses mistral
```

## License

MIT
