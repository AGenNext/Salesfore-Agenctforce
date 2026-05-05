# Salesforce Agentforce Agent

An AI agent built with Salesforce Agentforce framework concepts and integrated with open-source LLMs via LiteLLM.

## Overview

This project implements an AI agent that combines Salesforce Agentforce patterns with open-source LLM capabilities. The agent uses LiteLLM to connect to local models (Ollama) while following Agentforce agent development patterns.

## Features

- **Agent Architecture**: Implements agentic patterns following Salesforce Agentforce guidelines
- **Open-Source LLM Integration**: Uses LiteLLM to connect to Ollama and other open-source models
- **Function Calling**: Supports tool/actions for extended capabilities
- **Prompt Templates**: Includes prompt management similar to Agentforce
- **REST API**: Exposes agent chat interface via API

## Architecture

```
agentforce-open-source-agent/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # Main agent implementation
│   ├── config.py         # Configuration management
│   ├── llm.py            # LLM integration (LiteLLM)
│   ├── tools.py          # Tool/action definitions
│   └── prompt.py         # Prompt template management
├── api/
│   └── app.py            # FastAPI application
├── examples/
│   └── run_agent.py      # Usage examples
├── requirements.txt     # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.9+
- Ollama (optional, for local LLM)
- LiteLLM

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the following:

```env
# LLM Configuration
LITELLM_MODEL=ollama/llama3
LITELLM_API_BASE=http://localhost:4000
OPENAI_API_KEY=sk-dummy

# Agent Configuration
AGENT_NAME=SalesforceAgent
AGENT_DESCRIPTION=An AI agent built with Agentforce patterns
MAX_ITERATIONS=10
TEMPERATURE=0.7
```

## Quick Start

### Using Ollama

1. Start Ollama:
```bash
ollama serve
ollama pull llama3
```

2. Start LiteLLM proxy:
```bash
litellm --model ollama/llama3 --port 4000
```

3. Run the agent:
```bash
python examples/run_agent.py
```

### Using Docker

```bash
docker run -it -p 4000:4000 -v ~/.ollama:/root/.ollama ghcr.io/ollama/ollama:latest
```

## Usage Examples

### Basic Agent Interaction

```python
from agentforce_agent import Agent

# Initialize the agent
agent = Agent(
    name="SalesforceAgent",
    model="ollama/llama3",
    temperature=0.7
)

# Chat with the agent
response = agent.chat("Hello, how can you help me?")
print(response)
```

### With Custom Tools

```python
from agentforce_agent import Agent, Tool

# Define custom tools
tools = [
    Tool(
        name="search_knowledge_base",
        description="Search the knowledge base for information",
        function=search_kb
    ),
    Tool(
        name="create_salesforce_record",
        description="Create a record in Salesforce",
        function=create_sf_record
    )
]

agent = Agent(tools=tools)
response = agent.chat("Find information about our products")
```

### Via API Server

```bash
uvicorn api.app:app --reload
```

Then query:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "user123"}'
```

## Agent Patterns

This implementation follows Salesforce Agentforce agent patterns:

1. **Topic-Based Actions**: Agents can trigger specific actions based on topics
2. **Prompt Templates**: Configurable prompts for different scenarios
3. **Function Calling**: Tool/actions that extend agent capabilities
4. **Session Management**: Maintaining conversation context

## Supported Open-Source Models

- Ollama models (llama3, mistral, codellama, etc.)
- LM Studio models
- Local GGUF models via LiteLLM
- Any OpenAI-compatible API

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black agent/ api/
isort agent/ api/
flake8 agent/ api/
```

## License

Apache License 2.0 - See LICENSE file

## Resources

- [Salesforce Agentforce Documentation](https://developer.salesforce.com/docs/ai/agentforce)
- [LiteLLM Documentation](https://docs.litellm.ai)
- [Ollama Documentation](https://github.com/ollama/ollama)