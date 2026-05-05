"""REST API for Agentforce agent"""

import logging
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentforce_agent import Agent, AgentConfig

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentforce Agent API",
    description="REST API for AI agent built with Agentforce patterns",
    version="1.0.0"
)

# Global agent instance
agent: Optional[Agent] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str


class ConfigRequest(BaseModel):
    """Config request model"""
    name: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None


def get_agent() -> Agent:
    """Get or create agent instance"""
    global agent
    if agent is None:
        config = AgentConfig.from_env()
        agent = Agent(config=config)
    return agent


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    config = AgentConfig.from_env()
    agent = Agent(config=config)
    logger.info(f"Agent initialized: {config.name}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentforce Agent API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the agent
    
    Args:
        request: Chat request with message and optional session_id
        
    Returns:
        Chat response with agent reply
    """
    try:
        agent_instance = get_agent()
        response = agent_instance.chat(
            message=request.message,
            session_id=request.session_id,
            temperature=request.temperature
        )
        
        # Get session ID
        session_id = request.session_id
        if not session_id:
            # Get the last created session ID
            sessions = agent_instance.list_sessions()
            session_id = sessions[-1] if sessions else "default"
        
        return ChatResponse(
            response=response,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/config")
async def configure_agent(request: ConfigRequest):
    """Configure the agent
    
    Args:
        request: Configuration options
        
    Returns:
        Updated configuration
    """
    try:
        agent_instance = get_agent()
        
        if request.name:
            agent_instance.config.name = request.name
        if request.model:
            agent_instance.config.model = request.model
        if request.system_prompt:
            agent_instance.config.system_prompt = request.system_prompt
        if request.temperature:
            agent_instance.config.temperature = request.temperature
        
        return agent_instance.get_config()
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/sessions")
async def list_sessions():
    """List all sessions
    
    Returns:
        List of session IDs
    """
    try:
        agent_instance = get_agent()
        return {"sessions": agent_instance.list_sessions()}
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/tools")
async def list_tools():
    """List all available tools
    
    Returns:
        List of tools
    """
    try:
        agent_instance = get_agent()
        tools = agent_instance.tools.list_tools()
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category
                }
                for t in tools
            ]
        }
    except Exception as e:
        logger.error(f"List tools error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)