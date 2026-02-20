# kore-bridge

LLM integration layer for [kore-mind](https://github.com/iafiscal1212/kore-mind). Runtime-agnostic cognitive bridge.

## Install

```bash
pip install kore-bridge                # core
pip install kore-bridge[openai]        # + OpenAI
pip install kore-bridge[anthropic]     # + Anthropic
pip install kore-bridge[all]           # everything
```

## Usage

```python
from kore_mind import Mind
from kore_bridge import Bridge
from kore_bridge.providers import OpenAIProvider

mind = Mind("agent.db")
llm = OpenAIProvider(model="gpt-4o-mini")
bridge = Bridge(mind=mind, llm=llm)

# Think with context (auto-remembers)
response = bridge.think("Help me with my proof", user="carlos")

# Manually observe something
bridge.observe("User prefers concise answers")

# Reflect: LLM generates emergent identity from memories
identity = bridge.reflect()
print(identity.summary)
```

## Custom LLM

Wrap any callable:

```python
from kore_bridge import CallableLLM

llm = CallableLLM(lambda msgs: my_ollama_call(msgs))
bridge = Bridge(mind=Mind("agent.db"), llm=llm)
```

## License

MIT
