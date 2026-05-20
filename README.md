# MimoHuman

Generic AI Agent framework with a Terminal UI (TUI) frontend.

## Architecture

```
mimohuman/
├── core/          # Framework library (no UI dependency)
│   ├── agent.py       # Agent: the central run loop
│   ├── provider.py    # LLMProvider ABC + StreamEvent
│   ├── tool.py        # Tool definitions + registry
│   ├── message.py     # Message types (User, Assistant, System, Tool)
│   ├── conversation.py # Message history management
│   ├── memory.py      # Pluggable long-term memory
│   ├── hooks.py       # Lifecycle hook system
│   └── orchestrator.py # Multi-agent coordination
├── providers/     # LLM provider implementations
│   ├── anthropic_provider.py
│   └── openai_provider.py
└── tui/           # Textual-based TUI frontend
    ├── app.py         # Textual App entry point
    ├── controller.py  # Bridge: TUI <-> Core
    ├── screens/
    └── widgets/
```

Core can be used independently from the TUI as a library.

## Installation

```bash
# Development install with all dependencies
pip install -e ".[all]"

# Or pick what you need
pip install -e ".[anthropic,tui]"
```

## Quick Start

### Library usage

```python
from mimohuman import Agent, AgentConfig, Tool, ToolRegistry
from mimohuman.providers import AnthropicProvider
from mimohuman.core.provider import ProviderConfig

async def my_tool(query: str) -> str:
    return f"Result for {query}"

registry = ToolRegistry()
registry.register(Tool(name="search", handler=my_tool))

config = AgentConfig(name="MyAgent", system_prompt="You are helpful.")
provider = AnthropicProvider(ProviderConfig(api_key="..."))
agent = Agent(config=config, provider=provider, tool_registry=registry)

async for event in agent.run("Hello!"):
    print(event)
```

### TUI

```bash
export ANTHROPIC_API_KEY="your-key"
mimohuman
```

Or programmatically:

```python
from mimohuman.tui import MimoHumanApp, TUIController

controller = TUIController(agent)
app = MimoHumanApp(controller)
app.run()
```

## Requirements

- Python 3.11+
- Anthropic API key (or OpenAI key) for LLM access
- Terminal with Unicode support for TUI

## License

MIT
