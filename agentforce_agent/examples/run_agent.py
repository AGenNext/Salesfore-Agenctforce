"""Example: Running the Agentforce agent"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agentforce_agent import Agent, AgentConfig, Tool


# Example custom tools
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information"""
    # This is a placeholder - replace with actual implementation
    results = [
        {"title": "Salesforce Basics", "content": "Salesforce is a CRM platform..."},
        {"title": "Agentforce Guide", "content": "Agentforce helps build AI agents..."}
    ]
    return str(results)


def calculate(a: int, b: int, operation: str = "add") -> int:
    """Perform a calculation"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a // b if b != 0 else "Error: Division by zero"
    return "Unknown operation"


def get_weather(location: str) -> str:
    """Get weather for a location"""
    # This is a placeholder - replace with actual implementation
    return f"Weather in {location}: Sunny, 72°F"


def run_basic_example():
    """Run basic agent example"""
    print("=" * 50)
    print("BASIC AGENT EXAMPLE")
    print("=" * 50)
    
    # Create configuration
    config = AgentConfig(
        name="SalesforceAgent",
        description="An AI agent with Agentforce patterns",
        model="ollama/llama3",
        temperature=0.7
    )
    
    # Create agent
    agent = Agent(config=config)
    
    # Chat with the agent
    print("\nUser: Hello! What are you?")
    response = agent.chat("Hello! What are you?")
    print(f"Agent: {response}")
    
    print("\nUser: Help me understand this agent framework")
    response = agent.chat("Help me understand this agent framework")
    print(f"Agent: {response}")


def run_with_tools_example():
    """Run example with custom tools"""
    print("\n" + "=" * 50)
    print("AGENT WITH TOOLS EXAMPLE")
    print("=" * 50)
    
    # Create tools
    tools = [
        Tool(
            name="search_knowledge_base",
            description="Search the knowledge base for information",
            function=search_knowledge_base,
            parameters={
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculate",
            description="Perform arithmetic calculations",
            function=calculate,
            parameters={
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                    "operation": {
                        "type": "string",
                        "description": "Operation (add, subtract, multiply, divide)",
                        "default": "add"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="get_weather",
            description="Get weather information for a location",
            function=get_weather,
            parameters={
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        )
    ]
    
    # Create agent with tools
    config = AgentConfig(
        name="SalesforceAgent",
        model="ollama/llama3"
    )
    agent = Agent(config=config, tools=tools)
    
    print("\nUser: Calculate 10 + 5")
    response = agent.chat("Calculate 10 + 5")
    print(f"Agent: {response}")
    
    print("\nUser: What's the weather in San Francisco?")
    response = agent.chat("What's the weather in San Francisco?")
    print(f"Agent: {response}")


def run_async_example():
    """Run async example"""
    print("\n" + "=" * 50)
    print("ASYNC AGENT EXAMPLE")
    print("=" * 50)
    
    async def main():
        config = AgentConfig(model="ollama/llama3")
        agent = Agent(config=config)
        
        # Async chat
        response = await agent.achat("Hello! Tell me about yourself.")
        print(f"\nUser: Hello! Tell me about yourself.")
        print(f"Agent: {response}")
    
    asyncio.run(main())


def run_api_example():
    """Show API configuration"""
    print("\n" + "=" * 50)
    print("API SERVER EXAMPLE")
    print("=" * 50)
    print("""
To run the API server:

1. Set environment variables:
   export LITELLM_MODEL=ollama/llama3
   export LITELLM_API_BASE=http://localhost:4000

2. Run the server:
   python -m agentforce_agent.api

3. Use the API:
   curl -X POST http://localhost:8000/chat \\
     -H "Content-Type: application/json" \\
     -d '{"message": "Hello!"}'
""")


def main():
    """Run examples"""
    print("Agentforce Agent - Examples")
    print("=" * 50)
    print("""
This example demonstrates an AI agent built with
Salesforce Agentforce framework patterns and
integrated with open-source LLMs via LiteLLM.
""")
    
    # Run basic example
    run_basic_example()
    
    # Run with tools
    run_with_tools_example()
    
    # Run async
    run_async_example()
    
    # Show API
    run_api_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()