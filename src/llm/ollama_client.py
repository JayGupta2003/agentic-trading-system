"""
Ollama LLM Client with VRAM-aware sequential execution.

Designed for 8GB VRAM constraint - executes agents one at a time with proper context clearing.
"""

from typing import Optional, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from config.settings import settings
from src.llm.vram_manager import get_vram_manager, clear_vram_for_next_agent

import loguru
logger = loguru.logger


class VRAMAwareOllamaLLM(ChatOllama):
    """
    Custom Ollama LLM wrapper that manages VRAM context between agent calls.
    
    Key behavior:
    - After each agent run, automatically triggers VRAM cleanup
    - Tracks which agent ran last
    - Provides methods to coordinate sequential agent execution
    """

    def __init__(self, model: str = None, current_agent: str = "unknown", **kwargs):
        model = model or settings.ollama_model
        kwargs.pop("current_agent", None)
        super().__init__(model=model, **kwargs)
        object.__setattr__(self, 'current_agent', current_agent)
        object.__setattr__(self, '_vram', get_vram_manager())

    def invoke(self, input: list[BaseMessage], config: Optional[dict] = None, **kwargs) -> ChatResult:
        """Invoke LLM and clear VRAM after completion."""
        logger.info(f"[{self.current_agent}] Invoking LLM...")
        response = super().invoke(input, config, **kwargs)
        clear_vram_for_next_agent(self.current_agent)
        return response

    def set_agent_name(self, name: str) -> None:
        """Update the current agent name for logging."""
        self.current_agent = name


def create_agent_llm(agent_name: str) -> VRAMAwareOllamaLLM:
    """Factory function to create a VRAM-aware LLM for a specific agent."""
    return VRAMAwareOllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        top_p=settings.top_p,
        num_ctx=4096,
        current_agent=agent_name,
    )


def run_agent_sequentially(
    agent_name: str,
    system_prompt: str,
    user_input: str,
    max_tokens: int = None,
) -> str:
    """
    Run a single agent with the given prompts.
    
    This is the primary interface for sequential agent execution.
    Call this for each agent in order (1, 2, 3, 4, 5, 6).
    """
    llm = create_agent_llm(agent_name)
    llm.set_agent_name(agent_name)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    
    logger.info(f"Running {agent_name}...")
    result = llm.invoke(messages)
    
    output = result.content if hasattr(result, "content") else str(result)
    logger.info(f"{agent_name} completed")
    
    return output


def test_llm_connection() -> dict:
    """Test connection to Ollama and verify model is accessible."""
    status = {"connected": False, "model_loaded": False, "error": None}
    
    try:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
        response = llm.invoke([HumanMessage(content="Say 'OK' if you can hear me.")])
        status["connected"] = True
        status["model_loaded"] = True
        status["test_response"] = response.content if hasattr(response, "content") else str(response)
        logger.info("Ollama connection test successful")
    except Exception as e:
        status["error"] = str(e)
        logger.error(f"Ollama connection failed: {e}")
    
    return status


def get_vram_status() -> dict:
    """Get current VRAM status."""
    return get_vram_manager().get_vram_info()
