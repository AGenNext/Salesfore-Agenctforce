"""Main agent implementation with Agentforce patterns"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .llm import LLMClient
from .prompt import PromptTemplate, PromptTemplateManager
from .tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class Session:
    """Agent conversation session"""
    
    def __init__(self, session_id: str, ttl: int = 3600):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.created_at = 0  # Would use time.time() in production
        self.ttl = ttl
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str):
        """Add message to history
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        self.history.append({"role": role, "content": content})
    
    def get_history(self, max_entries: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history
        
        Args:
            max_entries: Maximum number of entries to return
            
        Returns:
            Message history
        """
        if max_entries:
            return self.history[-max_entries:]
        return self.history


class Agent:
    """Agentforce-style AI agent with open-source LLM integration
    
    This agent implements Salesforce Agentforce patterns:
    - Topic-based conversations
    - Function calling (tools/actions)
    - Prompt templates
    - Session management
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the agent
        
        Args:
            name: Agent name (overrides config)
            model: Model name (overrides config)
            config: Agent configuration
            tools: List of tools
            system_prompt: Custom system prompt
        """
        self.config = config or AgentConfig.from_env()
        
        if name:
            self.config.name = name
        if model:
            self.config.model = model
        
        # Initialize components
        self.llm = LLMClient(self.config)
        self.tools = ToolRegistry()
        self.prompts = PromptTemplateManager()
        
        # Register tools
        if tools:
            for tool in tools:
                self.tools.add_tool(tool)
        
        # Sessions storage
        self._sessions: Dict[str, Session] = {}
        
        # System prompt
        if system_prompt:
            self.config.system_prompt = system_prompt
        elif not self.config.system_prompt:
            self.config.system_prompt = self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt
        
        Returns:
            Default system prompt string
        """
        template = self.prompts.get_template("agent_introduction")
        if template:
            return template.render(
                agent_name=self.config.name,
                agent_description=self.config.description
            )
        
        return f"""You are {self.config.name}, {self.config.description}.

You are an AI assistant that helps users accomplish their tasks."""
    
    def _get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get or create a session
        
        Args:
            session_id: Session ID or None for new session
            
        Returns:
            Session object
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        
        new_session_id = session_id or str(uuid.uuid4())
        session = Session(new_session_id, self.config.session_ttl)
        self._sessions[new_session_id] = session
        return session
    
    def _process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        session: Session
    ) -> List[Dict[str, Any]]:
        """Process tool calls from LLM response
        
        Args:
            tool_calls: List of tool calls
            session: Current session
            
        Returns:
            Tool execution results
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool = self.tools.get_tool(tool_name)
            
            if not tool:
                results.append({
                    "tool_call_id": tool_call.get("id"),
                    "content": f"Error: Tool {tool_name} not found"
                })
                continue
            
            # Parse arguments
            try:
                arguments = json.loads(
                    tool_call.get("function", {}).get("arguments", "{}")
                )
            except json.JSONDecodeError:
                arguments = {}
            
            # Execute tool
            try:
                result = tool.execute(arguments)
                results.append({
                    "tool_call_id": tool_call.get("id"),
                    "content": str(result)
                })
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                results.append({
                    "tool_call_id": tool_call.get("id"),
                    "content": f"Error: {str(e)}"
                })
        
        return results
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Chat with the agent
        
        Args:
            message: User message
            session_id: Optional session ID
            temperature: Optional temperature override
            
        Returns:
            Agent response
        """
        # Get or create session
        session = self._get_or_create_session(session_id)
        
        # Add user message to history
        session.add_message("user", message)
        
        # Format messages for LLM
        messages = session.get_history(self.config.max_history)
        formatted_messages = self.llm.format_messages(
            messages,
            self.config.system_prompt
        )
        
        # Get tool schemas if available
        tool_schemas = None
        if self.tools.list_tools():
            tool_schemas = self.tools.get_tool_schemas()
        
        # Call LLM
        response = self.llm.complete(
            formatted_messages,
            tools=tool_schemas,
            temperature=temperature
        )
        
        content = response.get("content", "")
        
        # Check for tool calls
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            # Process tool calls
            tool_results = self._process_tool_calls(tool_calls, session)
            
            # Add tool results to conversation
            for result in tool_results:
                session.add_message(
                    "tool",
                    result.get("content", "")
                )
            
            # Get updated messages and call LLM again
            messages = session.get_history(self.config.max_history)
            formatted_messages = self.llm.format_messages(
                messages,
                self.config.system_prompt
            )
            
            # Get tool schemas again
            tool_schemas = self.tools.get_tool_schemas() if self.tools.list_tools() else None
            
            # Call LLM with tool results
            response = self.llm.complete(
                formatted_messages,
                tools=tool_schemas,
                temperature=temperature
            )
            content = response.get("content", "")
        
        # Add assistant response to history
        session.add_message("assistant", content)
        
        return content
    
    async def achat(
        self,
        message: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Async version of chat
        
        Args:
            message: User message
            session_id: Optional session ID
            temperature: Optional temperature override
            
        Returns:
            Agent response
        """
        # Get or create session
        session = self._get_or_create_session(session_id)
        
        # Add user message to history
        session.add_message("user", message)
        
        # Format messages for LLM
        messages = session.get_history(self.config.max_history)
        formatted_messages = self.llm.format_messages(
            messages,
            self.config.system_prompt
        )
        
        # Get tool schemas if available
        tool_schemas = None
        if self.tools.list_tools():
            tool_schemas = self.tools.get_tool_schemas()
        
        # Call LLM
        response = await self.llm.acomplete(
            formatted_messages,
            tools=tool_schemas,
            temperature=temperature
        )
        
        content = response.get("content", "")
        
        # Check for tool calls
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            # Process tool calls
            tool_results = self._process_tool_calls(tool_calls, session)
            
            # Add tool results to conversation
            for result in tool_results:
                session.add_message(
                    "tool",
                    result.get("content", "")
                )
            
            # Get updated messages and call LLM again
            messages = session.get_history(self.config.max_history)
            formatted_messages = self.llm.format_messages(
                messages,
                self.config.system_prompt
            )
            
            # Get tool schemas again
            tool_schemas = self.tools.get_tool_schemas() if self.tools.list_tools() else None
            
            # Call LLM with tool results
            response = await self.llm.acomplete(
                formatted_messages,
                tools=tool_schemas,
                temperature=temperature
            )
            content = response.get("content", "")
        
        # Add assistant response to history
        session.add_message("assistant", content)
        
        return content
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent
        
        Args:
            tool: Tool to add
        """
        self.tools.add_tool(tool)
    
    def remove_tool(self, name: str):
        """Remove a tool from the agent
        
        Args:
            name: Tool name
        """
        self.tools.remove_tool(name)
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session or None
        """
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs
        
        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration
        
        Returns:
            Configuration dictionary
        """
        return self.config.to_dict()