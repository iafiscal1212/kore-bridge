# kore-bridge

LLM integration layer for [kore-mind](https://github.com/iafiscal1212/kore-mind). Runtime-agnostic cognitive bridge.

## Install

```bash
pip install kore-bridge                # core (zero deps beyond kore-mind)
pip install kore-bridge[openai]        # + OpenAI
pip install kore-bridge[anthropic]     # + Anthropic
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
```

## Demo

```bash
python examples/demo_llm.py              # uses llama3.2
python examples/demo_llm.py mistral      # uses mistral
```

## License

MIT
