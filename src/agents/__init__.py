"""
Trading System Agents Package

All 6 specialized agents:
1. Fundamental Analyst - Value investing, financial health
2. Technical Analyst - Price action, momentum
3. Sentiment Analyst - News, smart money tracking
4. Derivatives Analyst - Options flow, institutional positioning
5. Risk Manager - Position sizing, portfolio protection
6. CIO - Final synthesis and trading decision

Plus integration layer for connecting data providers to agents.
"""

from src.agents.base import BaseAgent, AgentState, SequentialPipeline, create_agent_graph
from src.agents.fundamental import FundamentalAgent
from src.agents.technical import TechnicalAgent
from src.agents.sentiment import SentimentAgent
from src.agents.derivatives import DerivativesAgent
from src.agents.risk import RiskAgent
from src.agents.cio import CIOAgent
from src.agents.integrator import AgentIntegrator, agent_integrator, analyze_stock
from src.agents.langgraph_orchestrator import LangGraphOrchestrator, AgenticState, create_orchestrator
from src.agents.routing import (
    get_agents_for_trigger,
    get_agents_to_skip,
    get_agent_routing_summary,
    validate_routing,
    should_skip_derivatives,
    get_previous_derivatives_summary,
    estimate_vram_usage,
    get_optimal_agent_order,
    format_routing_report,
)

__all__ = [
    "BaseAgent",
    "AgentState", 
    "SequentialPipeline",
    "create_agent_graph",
    "FundamentalAgent",
    "TechnicalAgent",
    "SentimentAgent",
    "DerivativesAgent",
    "RiskAgent",
    "CIOAgent",
    "AgentIntegrator",
    "agent_integrator",
    "analyze_stock",
    "LangGraphOrchestrator",
    "AgenticState",
    "create_orchestrator",
    "get_agents_for_trigger",
    "get_agents_to_skip",
    "get_agent_routing_summary",
    "validate_routing",
    "should_skip_derivatives",
    "get_previous_derivatives_summary",
    "estimate_vram_usage",
    "get_optimal_agent_order",
    "format_routing_report",
]
