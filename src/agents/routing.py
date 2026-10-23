"""
Agent Routing Module - Trigger-Based Agent Selection

This module implements the agent routing logic based on trigger types
as specified in Section 5.3 of REDESIGN_PLAN.md.

Key routing rules:
- Scheduled Monthly → All 6 Agents
- Price Drop → Tech, Sentiment, Risk, CIO (skip Fund, Derivatives)
- Earnings → Fundamental, Sentiment, CIO (skip Tech, Derivatives)
- VIX Spike → Technical, Derivatives, Risk, CIO (skip Fund, Sentiment)
- News Anomaly → Sentiment, Technical, CIO (skip Fund, Derivatives)
- Max Drawdown → Risk, CIO only (Circuit Breaker)
- Expiry Week → Skip Derivatives Agent
"""

from typing import List, Dict, Set, Optional
from datetime import datetime

from src.trading.types import TriggerType, AgentName, AGENT_ROUTING, TradingSignal


def get_agents_for_trigger(trigger_type: TriggerType) -> List[AgentName]:
    """Returns list of agents to run based on trigger type."""
    return AGENT_ROUTING.get(trigger_type, [AgentName.CIO])


def get_agents_to_skip(trigger_type: TriggerType) -> List[AgentName]:
    """Returns list of agents to skip based on trigger type."""
    all_agents = [
        AgentName.FUNDAMENTAL,
        AgentName.TECHNICAL,
        AgentName.SENTIMENT,
        AgentName.DERIVATIVES,
        AgentName.RISK,
        AgentName.CIO,
    ]
    run_agents = get_agents_for_trigger(trigger_type)
    return [a for a in all_agents if a not in run_agents]


def get_agent_routing_summary(trigger_type: TriggerType) -> Dict[str, List[str]]:
    """Get a summary of which agents run vs skip for a trigger type."""
    return {
        "run": [a.value for a in get_agents_for_trigger(trigger_type)],
        "skip": [a.value for a in get_agents_to_skip(trigger_type)],
    }


def validate_routing(trigger_type: TriggerType) -> bool:
    """Validate that routing for a trigger type is properly configured."""
    run_agents = get_agents_for_trigger(trigger_type)
    
    if AgentName.CIO not in run_agents:
        return False
    
    if trigger_type == TriggerType.SCHEDULED and len(run_agents) != 6:
        return False
    
    if trigger_type == TriggerType.MAX_DRAWDOWN and len(run_agents) > 2:
        return False
    
    return True


def should_skip_derivatives(trigger_type: TriggerType, is_expiry_week: bool = False) -> bool:
    """Determine if derivatives agent should be skipped.
    
    Per Section 5.6 of REDESIGN_PLAN.md:
    - Always skip during expiry week
    - Skip during VIX spike (derivatives already baked in)
    - Skip during price drops (unreliable OI data during volatility)
    """
    if is_expiry_week:
        return True
    
    skip_triggers = {
        TriggerType.PRICE_DROP,
        TriggerType.NEWS_ANOMALY,
        TriggerType.MAX_DRAWDOWN,
    }
    
    return trigger_type in skip_triggers


def get_previous_derivatives_summary(signal: Optional[TradingSignal]) -> str:
    """Get previous derivatives summary for expiry week fallback.
    
    When derivatives agent is skipped, use the previous summary.
    """
    if signal and signal.previous_derivatives_summary:
        return signal.previous_derivatives_summary
    
    return "[Derivatives data unavailable - using previous summary]"


def estimate_vram_usage(agent_names: List[str]) -> int:
    """Estimate VRAM usage in MB for running given agents.
    
    Based on approximate LLM context sizes:
    - fundamental: ~800MB (larger context)
    - technical: ~600MB
    - sentiment: ~600MB
    - derivatives: ~500MB
    - risk: ~500MB
    - cio: ~700MB (needs all context)
    """
    agent_vram = {
        "fundamental": 800,
        "technical": 600,
        "sentiment": 600,
        "derivatives": 500,
        "risk": 500,
        "cio": 700,
    }
    
    total = sum(agent_vram.get(a, 500) for a in agent_names)
    return total


def get_optimal_agent_order(trigger_type: TriggerType) -> List[str]:
    """Get optimal execution order for agents based on trigger type.
    
    Optimization: Run agents in order that maximizes information flow
    and minimizes redundant context.
    """
    base_order = ["fundamental", "technical", "sentiment", "derivatives", "risk", "cio"]
    run_agents = [a for a in base_order if any(a in ag.value for ag in get_agents_for_trigger(trigger_type))]
    
    if trigger_type == TriggerType.SCHEDULED:
        return run_agents
    
    if trigger_type == TriggerType.PRICE_DROP:
        return ["technical", "sentiment", "risk", "cio"]
    
    if trigger_type == TriggerType.EARNINGS:
        return ["fundamental", "sentiment", "cio"]
    
    if trigger_type == TriggerType.VIX_SPIKE:
        return ["technical", "derivatives", "risk", "cio"]
    
    if trigger_type == TriggerType.NEWS_ANOMALY:
        return ["sentiment", "technical", "cio"]
    
    if trigger_type == TriggerType.MAX_DRAWDOWN:
        return ["risk", "cio"]
    
    if trigger_type == TriggerType.SECTOR_COLLAPSE:
        return ["technical", "sentiment", "risk", "cio"]
    
    return run_agents


def format_routing_report(trigger_type: TriggerType, is_expiry_week: bool = False) -> str:
    """Generate a human-readable routing report."""
    summary = get_agent_routing_summary(trigger_type)
    
    lines = [
        f"Trigger Type: {trigger_type.value}",
        f"",
        f"Agents to RUN ({len(summary['run'])}):",
    ]
    for agent in summary["run"]:
        lines.append(f"  ✓ {agent}")
    
    skip_agents = summary["skip"]
    if is_expiry_week and "derivatives" not in skip_agents:
        skip_agents = skip_agents + ["derivatives (expiry week)"]
    
    lines.append(f"")
    lines.append(f"Agents to SKIP ({len(skip_agents)}):")
    for agent in skip_agents:
        lines.append(f"  ✗ {agent}")
    
    lines.append(f"")
    lines.append(f"Estimated VRAM: ~{estimate_vram_usage(summary['run'])}MB")
    
    return "\n".join(lines)
