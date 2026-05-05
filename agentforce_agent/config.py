"""Agent configuration management"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentConfig:
    """Configuration for the Agentforce agent"""
    
    # Agent identity
    name: str = "SalesforceAgent"
    description: str = "An AI agent built with Agentforce patterns"
    version: str = "1.0.0"
    
    # LLM Configuration
    model: str = "ollama/llama3"
    api_base: str = "http://localhost:4000"
    api_key: str = "sk-dummy"
    temperature: float = 0.7
    max_tokens: int = 2048
    request_timeout: int = 120
    
    # Agent Behavior
    max_iterations: int = 10
    retry_attempts: int = 3
    system_prompt: Optional[str] = None
    
    # Session Configuration
    session_ttl: int = 3600  # Time to live in seconds
    max_history: int = 20
    
    # Tools/Actions
    enabled_tools: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables"""
        return cls(
            name=os.getenv("AGENT_NAME", "SalesforceAgent"),
            description=os.getenv("AGENT_DESCRIPTION", "An AI agent built with Agentforce patterns"),
            model=os.getenv("LITELLM_MODEL", "ollama/llama3"),
            api_base=os.getenv("LITELLM_API_BASE", "http://localhost:4000"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            system_prompt=os.getenv("SYSTEM_PROMPT", None),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "model": self.model,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
        }