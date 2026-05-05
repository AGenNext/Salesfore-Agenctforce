"""Prompt template management for Agentforce-style agents"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptVariable:
    """Represents a variable in a prompt template"""
    
    name: str
    description: str
    type: str = "string"
    default: Optional[Any] = None
    required: bool = False


@dataclass
class PromptTemplate:
    """Prompt template similar to Salesforce Agentforce prompt templates
    
    Provides a structured way to define prompts with variables and patterns
    for different agent scenarios.
    """
    
    name: str
    content: str
    description: str = ""
    version: str = "1.0.0"
    variables: List[PromptVariable] = field(default_factory=list)
    category: str = "general"
    enabled: bool = True
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Rendered prompt string
        """
        content = self.content
        
        for var in self.variables:
            if var.name in kwargs:
                value = kwargs[var.name]
            elif var.default is not None:
                value = var.default
            elif var.required:
                raise ValueError(f"Required variable {var.name} not provided")
            else:
                continue
            
            # Replace placeholder with value
            placeholder = f"{{{{{var.name}}}}}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "version": self.version,
            "variables": [
                {
                    "name": v.name,
                    "description": v.description,
                    "type": v.type,
                    "default": v.default,
                    "required": v.required
                }
                for v in self.variables
            ],
            "category": self.category,
            "enabled": self.enabled
        }


class PromptTemplateManager:
    """Manager for prompt templates"""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default templates"""
        default_templates = [
            PromptTemplate(
                name="agent_introduction",
                content="""You are {{agent_name}}, {{agent_description}}.

You are an AI assistant that helps users accomplish their tasks. You have access to various tools and capabilities that allow you to:
- Search and retrieve information
- Perform calculations
- Interact with external systems
- And more

Always be helpful, accurate, and follow user instructions carefully.""",
                description="Default introduction prompt",
                variables=[
                    PromptVariable(
                        name="agent_name",
                        description="The name of the agent",
                        default="AI Assistant"
                    ),
                    PromptVariable(
                        name="agent_description",
                        description="Description of what the agent does",
                        default="An AI assistant that helps users"
                    )
                ]
            ),
            PromptTemplate(
                name="tool_execution",
                content="""You need to use a tool to complete this request.

Tool: {{tool_name}}
Description: {{tool_description}}

Execute the tool with the following parameters:
{{parameters}}

Provide the result of the tool execution.""",
                description="Prompt for tool execution",
                variables=[
                    PromptVariable(
                        name="tool_name",
                        description="Name of the tool to execute",
                        required=True
                    ),
                    PromptVariable(
                        name="tool_description",
                        description="Description of the tool"
                    ),
                    PromptVariable(
                        name="parameters",
                        description="Tool parameters as JSON",
                        default="{}"
                    )
                ]
            ),
            PromptTemplate(
                name="error_handling",
                content="""An error occurred while processing your request:

Error: {{error_message}}

Please try again or modify your request. If the issue persists, contact support.""",
                description="Error response prompt",
                variables=[
                    PromptVariable(
                        name="error_message",
                        description="The error message",
                        required=True
                    )
                ]
            )
        ]
        
        for template in default_templates:
            self._templates[template.name] = template
    
    def add_template(self, template: PromptTemplate):
        """Add a template
        
        Args:
            template: Template to add
        """
        self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name
        
        Args:
            name: Template name
            
        Returns:
            Template or None
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[PromptTemplate]:
        """List all templates
        
        Returns:
            List of templates
        """
        return list(self._templates.values())
    
    def remove_template(self, name: str):
        """Remove a template
        
        Args:
            name: Template name
        """
        if name in self._templates:
            del self._templates[name]