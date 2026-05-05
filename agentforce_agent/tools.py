"""Tool/Action definitions for Agentforce agent"""

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a tool/action that the agent can use
    
    Similar to Salesforce Agentforce actions, tools extend the agent's capabilities
    by allowing it to interact with external systems and perform specific tasks.
    """
    
    name: str
    description: str
    function: Callable
    parameters: Optional[Dict[str, Any]] = None
    category: str = "general"
    enabled: bool = True
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling schema
        
        Returns:
            Dictionary in OpenAI tool schema format
        """
        # Generate parameter schema from function signature
        import inspect
        
        params_schema = {"type": "object", "properties": {}}
        
        if self.parameters:
            params_schema["properties"] = self.parameters.get("properties", {})
            if "required" in self.parameters:
                params_schema["required"] = self.parameters["required"]
        else:
            # Try to infer from function signature
            try:
                sig = inspect.signature(self.function)
                for param_name, param in sig.parameters.items():
                    param_type = "string"
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    
                    params_schema["properties"][param_name] = {
                        "type": param_type,
                        "description": f"Parameter {param_name}"
                    }
            except Exception:
                pass
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params_schema
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(**arguments)
            return self.function(**arguments)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {"error": str(e)}


class ToolRegistry:
    """Registry for managing agent tools"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def add_tool(self, tool: Tool):
        """Add a tool to the registry
        
        Args:
            tool: Tool to add
        """
        self._tools[tool.name] = tool
    
    def remove_tool(self, name: str):
        """Remove a tool from the registry
        
        Args:
            name: Tool name to remove
        """
        if name in self._tools:
            del self._tools[name]
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name
        
        Args:
            name: Tool name
            
        Returns:
            Tool or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools
        
        Returns:
            List of tools
        """
        return [tool for tool in self._tools.values() if tool.enabled]
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI tool schemas for all enabled tools
        
        Returns:
            List of tool schemas
        """
        return [tool.to_openai_schema() for tool in self.list_tools()]