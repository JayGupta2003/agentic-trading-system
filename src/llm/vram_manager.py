"""
VRAM Manager for sequential agent execution.

On 8GB VRAM systems, we must execute agents strictly sequentially:
1. Wake up Agent 1, extract output
2. Clear VRAM context
3. Wake up Agent 2, extract output
4. Repeat
"""

import gc
import loguru
from typing import Optional

logger = loguru.logger


class VRAMManager:
    """Manages VRAM context clearing between agent executions."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._agent_contexts: dict = {}

    def register_context(self, agent_name: str, context: dict) -> None:
        """Store agent output context for next agent."""
        self._agent_contexts[agent_name] = context
        logger.debug(f"Registered context for {agent_name}")

    def get_context(self, agent_name: str) -> Optional[dict]:
        """Retrieve stored context for an agent."""
        return self._agent_contexts.get(agent_name)

    def get_all_contexts(self) -> dict:
        """Get all stored contexts."""
        return self._agent_contexts.copy()

    def clear_agent_context(self, agent_name: str) -> None:
        """Clear specific agent context and free VRAM."""
        if agent_name in self._agent_contexts:
            del self._agent_contexts[agent_name]
            self._force_garbage_collection()
            logger.info(f"Cleared context and VRAM for {agent_name}")

    def clear_all_contexts(self) -> None:
        """Clear all contexts and free all VRAM."""
        self._agent_contexts.clear()
        self._force_garbage_collection()
        logger.info("Cleared all agent contexts and VRAM")

    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free VRAM."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug(f"GPU VRAM after clear: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        except ImportError:
            pass

    def get_vram_info(self) -> dict:
        """Get current VRAM usage info."""
        info = {"cuda_available": False, "memory_allocated_gb": 0, "memory_reserved_gb": 0}
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                info["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                info["model_size_gb"] = 5.7
                info["available_after_model_gb"] = max(0, 8 - 5.7 - info["memory_reserved_gb"])
        except ImportError:
            pass
        return info


_vram_manager: Optional[VRAMManager] = None


def get_vram_manager(model_name: str = "huggingface.co/QuantFactory/finance-Llama3-8B-GGUF:Q5_K_M") -> VRAMManager:
    """Get singleton VRAM manager instance."""
    global _vram_manager
    if _vram_manager is None:
        _vram_manager = VRAMManager(model_name)
    return _vram_manager


def clear_vram_for_next_agent(current_agent: str) -> None:
    """Utility to clear VRAM before running the next agent."""
    vram = get_vram_manager()
    vram.clear_agent_context(current_agent)
    logger.info(f"VRAM cleared after {current_agent}, ready for next agent")
