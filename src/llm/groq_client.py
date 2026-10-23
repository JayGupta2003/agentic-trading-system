"""
Groq LLM Client - For CIO Agent Only

Uses Groq API for fast inference with large context.
Other agents continue to use Ollama.

Available Groq Models:
- groq/compound: Agentic AI with web search and code execution (131K context)
- groq/compound-mini: Smaller agentic AI version
- llama-3.1-8b-instant: Fast Llama (131K context)
- llama-3.3-70b-versatile: Large Llama (131K context)
"""

from typing import Optional, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq
import os

import loguru
logger = loguru.logger


DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"


class GroqLLM:
    """
    Wrapper for Groq LLM client.
    
    Used specifically for CIO agent which benefits from:
    - Longer context window (131K tokens)
    - Better reasoning capabilities
    - Fast inference (~450 tokens/sec)
    """
    
    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        logger.info(f"GroqLLM initialized with model: {model}")
    
    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResult:
        """Invoke the Groq LLM."""
        logger.info(f"[CIO] Invoking Groq LLM (model: {self.model})...")
        
        response = self.client.invoke(messages, **kwargs)
        
        return response
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Simple callable interface."""
        messages = [HumanMessage(content=prompt)]
        response = self.invoke(messages, **kwargs)
        return response.content if hasattr(response, "content") else str(response)


def create_cio_llm(
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> GroqLLM:
    """
    Create a Groq LLM instance for CIO agent.
    
    Available models:
    - groq/compound: Full agentic AI (~450 tokens/sec, 131K context)
    - groq/compound-mini: Smaller agentic version
    - llama-3.1-8b-instant: Fast Llama (~560 tokens/sec, 131K context)
    - llama-3.3-70b-versatile: Large Llama (~280 tokens/sec, 131K context)
    
    Args:
        model: Groq model name
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
        
    Returns:
        GroqLLM instance configured for CIO
    """
    return GroqLLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def test_groq_connection(model: str = DEFAULT_GROQ_MODEL) -> Dict[str, Any]:
    """
    Test connection to Groq API.
    
    Args:
        model: Model to test (default: groq/compound)
        
    Returns:
        Dict with connection status
    """
    status = {
        "connected": False,
        "model_working": False,
        "error": None,
        "model_tested": model,
    }
    
    try:
        client = ChatGroq(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        
        response = client.invoke([
            HumanMessage(content="Say 'OK' if you can hear me.")
        ])
        
        status["connected"] = True
        status["model_working"] = True
        status["test_response"] = response.content if hasattr(response, "content") else str(response)
        
        logger.info(f"Groq connection test successful with model: {model}")
        
    except Exception as e:
        status["error"] = str(e)
        logger.error(f"Groq connection failed: {e}")
    
    return status


def run_cio_with_groq(
    system_prompt: str,
    user_input: str,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.3,
) -> str:
    """
    Run CIO analysis using Groq.
    
    This is a convenience function for running CIO with Groq.
    
    Args:
        system_prompt: System prompt for CIO role
        user_input: User input containing agent summaries
        model: Groq model to use
        temperature: Sampling temperature
        
    Returns:
        LLM response text
    """
    llm = create_cio_llm(model=model, temperature=temperature)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    
    logger.info(f"Running CIO with Groq model: {model}")
    response = llm.invoke(messages)
    
    output = response.content if hasattr(response, "content") else str(response)
    logger.info("CIO Groq analysis completed")
    
    return output


def list_available_models() -> List[str]:
    """List all available Groq models."""
    return [
        "groq/compound",       # Full agentic AI
        "groq/compound-mini",   # Smaller agentic
        "llama-3.1-8b-instant", # Fast Llama 8B
        "llama-3.3-70b-versatile", # Large Llama 70B
    ]


if __name__ == "__main__":
    print("Testing Groq connection...")
    print(f"Available models: {list_available_models()}")
    
    for model in ["groq/compound", "llama-3.1-8b-instant"]:
        print(f"\nTesting {model}...")
        status = test_groq_connection(model)
        
        if status["connected"]:
            print(f"✅ {model} - Connected!")
            print(f"   Response: {status.get('test_response', 'N/A')}")
        else:
            print(f"❌ {model} - Failed: {status.get('error', 'Unknown')}")
