"""Phase 4: LangGraph Integration Tests

Tests for:
- Trigger-based agent routing
- VRAM-efficient AgenticState
- LangGraph Orchestrator
- Derivatives agent expiry week handling
- Agent routing module
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.trading.types import TriggerType, AgentName, AGENT_ROUTING, TradingSignal
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
from src.agents.langgraph_orchestrator import (
    AgenticState,
    LangGraphOrchestrator,
    ExecutionParams,
)


class TestAgentRouting:
    def test_scheduled_routing_all_agents(self):
        agents = get_agents_for_trigger(TriggerType.SCHEDULED)
        assert AgentName.FUNDAMENTAL in agents
        assert AgentName.TECHNICAL in agents
        assert AgentName.SENTIMENT in agents
        assert AgentName.DERIVATIVES in agents
        assert AgentName.RISK in agents
        assert AgentName.CIO in agents
        assert len(agents) == 6

    def test_price_drop_routing(self):
        agents = get_agents_for_trigger(TriggerType.PRICE_DROP)
        assert AgentName.FUNDAMENTAL not in agents
        assert AgentName.TECHNICAL in agents
        assert AgentName.SENTIMENT in agents
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.RISK in agents
        assert AgentName.CIO in agents

    def test_earnings_routing(self):
        agents = get_agents_for_trigger(TriggerType.EARNINGS)
        assert AgentName.FUNDAMENTAL in agents
        assert AgentName.TECHNICAL not in agents
        assert AgentName.SENTIMENT in agents
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.RISK not in agents
        assert AgentName.CIO in agents

    def test_vix_spike_routing(self):
        agents = get_agents_for_trigger(TriggerType.VIX_SPIKE)
        assert AgentName.FUNDAMENTAL not in agents
        assert AgentName.TECHNICAL in agents
        assert AgentName.SENTIMENT not in agents
        assert AgentName.DERIVATIVES in agents
        assert AgentName.RISK in agents
        assert AgentName.CIO in agents

    def test_news_anomaly_routing(self):
        agents = get_agents_for_trigger(TriggerType.NEWS_ANOMALY)
        assert AgentName.FUNDAMENTAL not in agents
        assert AgentName.TECHNICAL in agents
        assert AgentName.SENTIMENT in agents
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.RISK not in agents
        assert AgentName.CIO in agents

    def test_max_drawdown_routing(self):
        agents = get_agents_for_trigger(TriggerType.MAX_DRAWDOWN)
        assert AgentName.FUNDAMENTAL not in agents
        assert AgentName.TECHNICAL not in agents
        assert AgentName.SENTIMENT not in agents
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.RISK in agents
        assert AgentName.CIO in agents
        assert len(agents) == 2

    def test_expiry_week_routing(self):
        agents = get_agents_for_trigger(TriggerType.EXPIRY_WEEK)
        assert AgentName.FUNDAMENTAL in agents
        assert AgentName.TECHNICAL in agents
        assert AgentName.SENTIMENT in agents
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.RISK in agents
        assert AgentName.CIO in agents


class TestGetAgentsToSkip:
    def test_scheduled_no_skip(self):
        skip = get_agents_to_skip(TriggerType.SCHEDULED)
        assert len(skip) == 0

    def test_max_drawdown_skip_most(self):
        skip = get_agents_to_skip(TriggerType.MAX_DRAWDOWN)
        assert AgentName.FUNDAMENTAL in skip
        assert AgentName.TECHNICAL in skip
        assert AgentName.SENTIMENT in skip
        assert AgentName.DERIVATIVES in skip


class TestRoutingSummary:
    def test_routing_summary_format(self):
        summary = get_agent_routing_summary(TriggerType.SCHEDULED)
        assert "run" in summary
        assert "skip" in summary
        assert len(summary["run"]) == 6
        assert len(summary["skip"]) == 0

    def test_routing_summary_price_drop(self):
        summary = get_agent_routing_summary(TriggerType.PRICE_DROP)
        assert "fundamental" in summary["skip"]
        assert "derivatives" in summary["skip"]
        assert "technical" in summary["run"]
        assert "cio" in summary["run"]


class TestValidateRouting:
    def test_scheduled_valid(self):
        assert validate_routing(TriggerType.SCHEDULED) is True

    def test_max_drawdown_valid(self):
        assert validate_routing(TriggerType.MAX_DRAWDOWN) is True

    def test_cio_always_included(self):
        for trigger_type in TriggerType:
            agents = get_agents_for_trigger(trigger_type)
            assert AgentName.CIO in agents


class TestShouldSkipDerivatives:
    def test_skip_on_expiry_week(self):
        assert should_skip_derivatives(TriggerType.SCHEDULED, is_expiry_week=True) is True

    def test_skip_on_price_drop(self):
        assert should_skip_derivatives(TriggerType.PRICE_DROP, is_expiry_week=False) is True

    def test_skip_on_news_anomaly(self):
        assert should_skip_derivatives(TriggerType.NEWS_ANOMALY, is_expiry_week=False) is True

    def test_skip_on_max_drawdown(self):
        assert should_skip_derivatives(TriggerType.MAX_DRAWDOWN, is_expiry_week=False) is True

    def test_dont_skip_on_scheduled(self):
        assert should_skip_derivatives(TriggerType.SCHEDULED, is_expiry_week=False) is False

    def test_dont_skip_on_vix_spike(self):
        assert should_skip_derivatives(TriggerType.VIX_SPIKE, is_expiry_week=False) is False


class TestPreviousDerivativesSummary:
    def test_with_existing_signal(self):
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            previous_derivatives_summary="PCR=0.8, Bullish sentiment",
        )
        result = get_previous_derivatives_summary(signal)
        assert result == "PCR=0.8, Bullish sentiment"

    def test_without_signal(self):
        result = get_previous_derivatives_summary(None)
        assert "unavailable" in result.lower()

    def test_without_previous_summary(self):
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        result = get_previous_derivatives_summary(signal)
        assert "unavailable" in result.lower()


class TestVramUsage:
    def test_scheduled_full_usage(self):
        agents = get_agents_for_trigger(TriggerType.SCHEDULED)
        usage = estimate_vram_usage([a.value for a in agents])
        assert usage > 3000

    def test_max_drawdown_minimal_usage(self):
        agents = get_agents_for_trigger(TriggerType.MAX_DRAWDOWN)
        usage = estimate_vram_usage([a.value for a in agents])
        assert usage < 1500

    def test_unknown_agent_default(self):
        usage = estimate_vram_usage(["unknown_agent"])
        assert usage == 500


class TestOptimalAgentOrder:
    def test_scheduled_order(self):
        order = get_optimal_agent_order(TriggerType.SCHEDULED)
        assert order[0] == "fundamental"
        assert order[-1] == "cio"
        assert order.index("technical") < order.index("sentiment")

    def test_price_drop_order(self):
        order = get_optimal_agent_order(TriggerType.PRICE_DROP)
        assert order == ["technical", "sentiment", "risk", "cio"]

    def test_earnings_order(self):
        order = get_optimal_agent_order(TriggerType.EARNINGS)
        assert order == ["fundamental", "sentiment", "cio"]

    def test_max_drawdown_order(self):
        order = get_optimal_agent_order(TriggerType.MAX_DRAWDOWN)
        assert order == ["risk", "cio"]


class TestRoutingReport:
    def test_report_format(self):
        report = format_routing_report(TriggerType.SCHEDULED)
        assert "Trigger Type:" in report
        assert "Agents to RUN" in report
        assert "Agents to SKIP" in report
        assert "Estimated VRAM:" in report

    def test_report_with_expiry_week(self):
        report = format_routing_report(TriggerType.SCHEDULED, is_expiry_week=True)
        assert "derivatives (expiry week)" in report


class TestAgenticState:
    def test_state_creation(self):
        state: AgenticState = {
            "symbol": "RELIANCE",
            "current_date": datetime.now(),
            "cap_type": "large",
            "price_summary": "Trading at 2500, RSI=45",
            "fundamental_summary": "P/E=18, ROE=15%",
            "news_summary": "3 positive articles",
            "derivatives_summary": "PCR=0.8",
            "fundamental_analysis": None,
            "technical_analysis": None,
            "sentiment_analysis": None,
            "derivatives_analysis": None,
            "risk_assessment": None,
            "cio_synthesis": None,
            "current_verdict": None,
            "conviction_score": None,
            "entry_price": None,
            "stop_loss": None,
            "target_price": None,
            "trailing_stop_atr": None,
            "trigger_type": "scheduled",
            "agents_to_run": ["fundamental", "technical", "sentiment", "derivatives", "risk", "cio"],
            "errors": [],
            "completed_agents": [],
        }
        assert state["symbol"] == "RELIANCE"
        assert state["fundamental_analysis"] is None

    def test_state_with_summaries_only(self):
        state: AgenticState = {
            "symbol": "RELIANCE",
            "current_date": datetime.now(),
            "cap_type": "large",
            "price_summary": "Trading at 2500, RSI=45",
            "fundamental_summary": "P/E=18, ROE=15%",
            "news_summary": "3 positive articles",
            "derivatives_summary": "PCR=0.8",
            "fundamental_analysis": "BUY - Strong fundamentals",
            "technical_analysis": "Bullish - Above 50 EMA",
            "sentiment_analysis": "Positive sentiment",
            "derivatives_analysis": None,
            "risk_assessment": None,
            "cio_synthesis": None,
            "current_verdict": "BUY",
            "conviction_score": 8,
            "entry_price": 2550.0,
            "stop_loss": 2400.0,
            "target_price": 2800.0,
            "trailing_stop_atr": 2.5,
            "trigger_type": "scheduled",
            "agents_to_run": ["fundamental", "technical", "sentiment", "derivatives", "risk", "cio"],
            "errors": [],
            "completed_agents": ["fundamental", "technical", "sentiment", "derivatives", "risk"],
        }
        assert state["fundamental_analysis"] is not None
        assert state["current_verdict"] == "BUY"
        assert state["conviction_score"] == 8


class TestLangGraphOrchestrator:
    def test_orchestrator_creation(self):
        orchestrator = LangGraphOrchestrator()
        assert orchestrator is not None
        assert orchestrator.checkpointer is not None

    def test_build_graph_all_agents(self):
        orchestrator = LangGraphOrchestrator()
        graph = orchestrator.build_graph()
        assert graph is not None

    def test_build_graph_subset_agents(self):
        orchestrator = LangGraphOrchestrator()
        graph = orchestrator.build_graph(agents_to_include=["fundamental", "technical", "cio"])
        assert graph is not None

    def test_build_trigger_graph(self):
        orchestrator = LangGraphOrchestrator()
        graph = orchestrator.build_trigger_graph(TriggerType.SCHEDULED)
        assert graph is not None

    def test_build_max_drawdown_graph(self):
        orchestrator = LangGraphOrchestrator()
        graph = orchestrator.build_trigger_graph(TriggerType.MAX_DRAWDOWN)
        assert graph is not None

    def test_orchestrator_clear_cache(self):
        orchestrator = LangGraphOrchestrator()
        orchestrator._llm_cache["test_agent"] = MagicMock()
        orchestrator._clear_llm_cache()
        assert len(orchestrator._llm_cache) == 0


class TestExecutionParams:
    def test_execution_params_creation(self):
        params = ExecutionParams(
            entry_price=2550.0,
            stop_loss=2400.0,
            target_price=2800.0,
            trailing_stop_pct=2.5,
            thesis="Strong earnings growth",
            key_reasons=["P/E below sector", "High ROE"],
            risks=["Sector rotation", "Interest rate risk"],
            conviction=8,
            cap_type="large",
        )
        assert params.entry_price == 2550.0
        assert params.conviction == 8
        assert len(params.key_reasons) == 2
        assert len(params.risks) == 2


class TestTriggerRoutingConsistency:
    def test_all_triggers_have_routing(self):
        for trigger_type in TriggerType:
            agents = get_agents_for_trigger(trigger_type)
            assert len(agents) > 0, f"No agents for {trigger_type}"

    def test_cio_in_all_triggers(self):
        for trigger_type in TriggerType:
            agents = get_agents_for_trigger(trigger_type)
            assert AgentName.CIO in agents, f"CIO missing for {trigger_type}"

    def test_agent_count_by_trigger_type(self):
        scheduled_agents = get_agents_for_trigger(TriggerType.SCHEDULED)
        assert len(scheduled_agents) == 6

        max_dd_agents = get_agents_for_trigger(TriggerType.MAX_DRAWDOWN)
        assert len(max_dd_agents) == 2

    def test_skip_derivatives_expiry_week(self):
        skip_agents = get_agents_to_skip(TriggerType.EXPIRY_WEEK)
        assert AgentName.DERIVATIVES in skip_agents


class TestVramOptimization:
    def test_summary_size_limits(self):
        state: AgenticState = {
            "symbol": "RELIANCE",
            "current_date": datetime.now(),
            "cap_type": "large",
            "price_summary": "A" * 2000,
            "fundamental_summary": "B" * 2000,
            "news_summary": "C" * 2000,
            "derivatives_summary": "D" * 2000,
            "fundamental_analysis": None,
            "technical_analysis": None,
            "sentiment_analysis": None,
            "derivatives_analysis": None,
            "risk_assessment": None,
            "cio_synthesis": None,
            "current_verdict": None,
            "conviction_score": None,
            "entry_price": None,
            "stop_loss": None,
            "target_price": None,
            "trailing_stop_atr": None,
            "trigger_type": "scheduled",
            "agents_to_run": [],
            "errors": [],
            "completed_agents": [],
        }
        total_summary_size = (
            len(state["price_summary"]) +
            len(state["fundamental_summary"]) +
            len(state["news_summary"]) +
            len(state["derivatives_summary"])
        )
        assert total_summary_size <= 10000

    def test_no_raw_data_in_state(self):
        state: AgenticState = {
            "symbol": "RELIANCE",
            "current_date": datetime.now(),
            "cap_type": "large",
            "price_summary": "Price data summary",
            "fundamental_summary": "Fundamental data summary",
            "news_summary": "News data summary",
            "derivatives_summary": "Derivatives data summary",
            "fundamental_analysis": None,
            "technical_analysis": None,
            "sentiment_analysis": None,
            "derivatives_analysis": None,
            "risk_assessment": None,
            "cio_synthesis": None,
            "current_verdict": None,
            "conviction_score": None,
            "entry_price": None,
            "stop_loss": None,
            "target_price": None,
            "trailing_stop_atr": None,
            "trigger_type": "scheduled",
            "agents_to_run": [],
            "errors": [],
            "completed_agents": [],
        }
        assert "raw_price_data" not in state
        assert "news_raw" not in state
        assert "options_chain_raw" not in state


class TestDerivativesExpiryWeek:
    def test_derivatives_agent_init_expiry_week(self):
        from src.agents.derivatives import DerivativesAgent
        agent = DerivativesAgent(is_expiry_week=True)
        assert agent._is_expiry_week is True

    def test_derivatives_agent_init_normal(self):
        from src.agents.derivatives import DerivativesAgent
        agent = DerivativesAgent(is_expiry_week=False)
        assert agent._is_expiry_week is False

    def test_derivatives_set_expiry_week(self):
        from src.agents.derivatives import DerivativesAgent
        agent = DerivativesAgent()
        agent.set_expiry_week(True)
        assert agent._is_expiry_week is True

    def test_derivatives_system_prompt_expiry(self):
        from src.agents.derivatives import DerivativesAgent
        agent = DerivativesAgent(is_expiry_week=True)
        prompt = agent.system_prompt
        assert "expiry week" in prompt.lower()

    def test_derivatives_format_data_expiry_warning(self):
        from src.agents.derivatives import DerivativesAgent
        agent = DerivativesAgent(is_expiry_week=True)
        data = {
            "pcr": 0.8,
            "vix": 15.0,
            "max_pain": 2500.0,
        }
        formatted = agent._format_data_for_llm("RELIANCE", data)
        assert "WARNING" in formatted
        assert "expiry" in formatted.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
