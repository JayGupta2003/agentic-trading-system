"""Trading module for event-driven trading system.

Components:
- SignalStore: Persistent storage for trading signals
- TriggerEngine: Evaluates market conditions for LLM review triggers
- ExecutionEngine: Entry/exit logic with Half-Kelly sizing and risk management
- brokers: Base broker interface and implementations (Historical, Simulated)
- types: Data classes and enums for trading signals and triggers

Usage:
    from src.trading import SignalStore, TriggerEngine, trigger_decision
    from src.trading.execution_engine import ExecutionEngine
    from src.trading.brokers import SimulatedBroker
    
    store = SignalStore()
    engine = TriggerEngine()
    
    result = trigger_decision(context)
    if result.should_run_agents:
        print(f"Run agents: {[a.value for a in result.agents_to_run]}")
"""

from .types import (
    AgentName,
    AGENT_ROUTING,
    CapType,
    MarketContext,
    SignalType,
    TradingSignal,
    TriggerConfig,
    TriggerResult,
    TriggerType,
    TRIGGER_PRIORITY,
)

from .signal_store import SignalStore

from .trigger_engine import (
    TriggerEngine,
    check_price_drop_trigger,
    check_vix_trigger,
    get_last_trading_day_of_month,
    is_expiry_week,
    is_scheduled_review_day,
    trigger_decision,
)

from .execution_engine import (
    BaseExecutionEngine,
    ExecutionConfig,
    ExecutionEngine,
    ExecutionResult,
    ExitDecision,
    ExitReason,
    RiskValidation,
)

__all__ = [
    # Types
    "AgentName",
    "AGENT_ROUTING",
    "CapType",
    "MarketContext",
    "SignalType",
    "TradingSignal",
    "TriggerConfig",
    "TriggerResult",
    "TriggerType",
    "TRIGGER_PRIORITY",
    # SignalStore
    "SignalStore",
    # TriggerEngine
    "TriggerEngine",
    "check_price_drop_trigger",
    "check_vix_trigger",
    "get_last_trading_day_of_month",
    "is_expiry_week",
    "is_scheduled_review_day",
    "trigger_decision",
    # Execution Engine
    "BaseExecutionEngine",
    "ExecutionConfig",
    "ExecutionEngine",
    "ExecutionResult",
    "ExitDecision",
    "ExitReason",
    "RiskValidation",
]
