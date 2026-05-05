"""LLM client using LiteLLM for open-source model integration"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

try:
    import litellm
    from litellm import acompletion, completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None
    acompletion = None

from .config import AgentConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with LLM models via LiteLLM"""
    
    def __init__(self, config: AgentConfig):
        """Initialize LLM client
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self._setup_litellm()
    
    def _setup_litellm(self):
        """Configure LiteLLM"""
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install with: pip install litellm"
            )
        
        # Configure LiteLLM settings
        litellm.drop_params = True
        litellm.set_verbose = False
        
        # Set API configuration
        import os
        os.environ["OPENAI_API_KEY"] = self.config.api_key
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a completion request to the LLM
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Response from the LLM
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "api_base": self.config.api_base,
        }
        
        if tools:
            params["tools"] = tools
        
        try:
            response = completion(
                **params,
                timeout=self.config.request_timeout
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise
    
    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async version of complete
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Response from the LLM
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "api_base": self.config.api_base,
        }
        
        if tools:
            params["tools"] = tools
        
        try:
            response = await acompletion(
                **params,
                timeout=self.config.request_timeout
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LiteLLM response into standard format
        
        Args:
            response: Raw response from LiteLLM
            
        Returns:
            Parsed response dictionary
        """
        # Extract message content
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message
            
            result = {
                "content": message.content or "",
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, "usage") else 0,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, "usage") else 0,
                    "total_tokens": response.usage.total_tokens if hasattr(response, "usage") else 0,
                }
            }
            
            # Check for tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = []
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return result
        
        return {"content": "", "model": "", "usage": {}}
    
    def format_messages(self, history: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format message history for LLM
        
        Args:
            history: Conversation history
            system_prompt: Optional system prompt
            
        Returns:
            Formatted messages
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt or self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.config.system_prompt
            })
        
        # Add conversation history
        messages.extend(history)
        
        return messages