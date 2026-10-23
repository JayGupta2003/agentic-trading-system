"""
Main Event Loop - Daily Execution System

This module implements the daily execution loop that ties together all components
built in phases 1-5 of the redesigned trading system:

- SignalStore: Persist and retrieve trading signals
- TriggerEngine: Monitor market conditions and fire triggers
- ExecutionEngine: Handle entry/exit decisions
- HistoricalBroker: Portfolio management
- EventDrivenBacktester: Historical validation

The event loop runs after market close each day and:
1. Checks for triggers (scheduled, price drop, VIX, earnings, etc.)
2. Runs appropriate agents based on trigger type
3. Updates signals and executes trades
4. Logs all decisions for audit trail
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import pandas as pd

from src.trading.types import (
    TradingSignal, TriggerType, AgentName, AGENT_ROUTING,
    MarketContext, TriggerConfig, TriggerResult
)
from src.trading.signal_store import SignalStore
from src.trading.trigger_engine import (
    TriggerEngine, is_expiry_week, is_scheduled_review_day,
    check_vix_trigger, check_price_drop_trigger,
    get_last_trading_day_of_month
)
from src.trading.execution_engine import ExecutionEngine, ExecutionConfig
from src.trading.brokers.historical import HistoricalBroker
from src.trading.brokers.base import Position
from src.data.summarizer import DataSummarizer


@dataclass
class EventLoopConfig:
    """Configuration for the event loop."""
    storage_path: str = "data/signals"
    trigger_log_path: str = "data/trigger_log.json"
    execution_log_path: str = "data/execution_log.json"
    
    use_scheduled_reviews: bool = True
    scheduled_review_day: int = -1
    
    include_vix_triggers: bool = True
    include_price_drop_triggers: bool = True
    include_earnings_triggers: bool = True
    include_news_triggers: bool = True
    include_max_drawdown_circuit_breaker: bool = True
    
    max_drawdown_threshold: float = 10.0
    vix_zscore_threshold: float = 2.0
    vix_history_window: int = 20
    
    max_positions: int = 5
    max_position_pct: float = 15.0
    min_conviction: int = 6
    
    trailing_stop_atr_large: float = 2.5
    trailing_stop_atr_mid: float = 3.0
    trailing_stop_atr_small: float = 3.5


@dataclass
class TriggerLogEntry:
    """A log entry for trigger events."""
    timestamp: datetime
    trigger_type: str
    symbols_affected: List[str]
    agents_run: List[str]
    decision: str
    details: Dict[str, Any]


@dataclass
class ExecutionLogEntry:
    """A log entry for execution events."""
    timestamp: datetime
    action: str
    symbol: str
    quantity: int
    price: float
    reason: str
    pnl: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class EventLoopStatus:
    """Current status of the event loop."""
    last_run: Optional[datetime]
    next_scheduled_review: datetime
    active_signals: int
    open_positions: int
    portfolio_value: float
    portfolio_drawdown_pct: float
    recent_triggers: List[TriggerLogEntry]
    recent_executions: List[ExecutionLogEntry]
    pending_reviews: List[str]


class TradingEventLoop:
    """Main event loop for daily trading operations.
    
    This class orchestrates the complete trading workflow:
    1. Load current signals and market context
    2. Evaluate triggers for each holding
    3. Generate new signals for triggered events
    4. Execute entries and exits
    5. Update trailing stops
    6. Log all decisions
    """
    
    def __init__(self, config: Optional[EventLoopConfig] = None):
        self.config = config or EventLoopConfig()
        
        self.signal_store = SignalStore(self.config.storage_path)
        
        self.trigger_engine = TriggerEngine(TriggerConfig(
            scheduled_review_day=self.config.scheduled_review_day,
            max_drawdown_threshold=self.config.max_drawdown_threshold,
            vix_zscore_threshold=self.config.vix_zscore_threshold,
            vix_history_window=self.config.vix_history_window,
        ))
        
        self.execution_engine = ExecutionEngine(ExecutionConfig(
            max_position_pct=self.config.max_position_pct,
            max_positions=self.config.max_positions,
            min_conviction=self.config.min_conviction,
            trailing_stop_atr_large=self.config.trailing_stop_atr_large,
            trailing_stop_atr_mid=self.config.trailing_stop_atr_mid,
            trailing_stop_atr_small=self.config.trailing_stop_atr_small,
        ))
        
        self.broker = HistoricalBroker()
        self.summarizer = DataSummarizer()
        
        self._trigger_log: List[TriggerLogEntry] = []
        self._execution_log: List[ExecutionLogEntry] = []
        
        self._load_logs()
    
    def run_daily(
        self,
        current_date: Optional[datetime] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> EventLoopStatus:
        """Run the daily event loop.
        
        Args:
            current_date: Current date (defaults to now)
            market_data: Optional pre-fetched market data
            
        Returns:
            EventLoopStatus with current system state
        """
        if current_date is None:
            current_date = datetime.now()
        
        self._load_current_state()
        
        if market_data is None:
            market_data = self._fetch_market_data(current_date)
        
        triggers_fired = self._evaluate_triggers(current_date, market_data)
        
        self._update_trailing_stops(current_date, market_data)
        
        executions = self._execute_trades(current_date, market_data)
        
        self._save_logs()
        
        return self._get_status(current_date, triggers_fired, executions)
    
    def _load_current_state(self) -> None:
        """Load current state from signal store and broker."""
        pass
    
    def _fetch_market_data(self, current_date: datetime) -> Dict[str, Any]:
        """Fetch current market data."""
        import yfinance as yf
        
        market_data: Dict[str, Any] = {
            "timestamp": current_date,
            "vix": 20.0,
            "vix_history": [20.0] * 30,
            "nifty_change_pct": 0.0,
        }
        
        try:
            vix = yf.Ticker("^INDIAVIX")
            vix_df = vix.history(start=current_date - timedelta(days=60), end=current_date)
            if len(vix_df) > 0:
                market_data["vix"] = float(vix_df["Close"].iloc[-1])
                market_data["vix_history"] = list(vix_df["Close"].tail(30).values)
        except Exception:
            pass
        
        try:
            nifty = yf.Ticker("^NSEI")
            nifty_df = nifty.history(start=current_date - timedelta(days=5), end=current_date)
            if len(nifty_df) >= 2:
                prices = nifty_df["Close"].values
                market_data["nifty_change_pct"] = (prices[-1] - prices[-2]) / prices[-2] * 100
        except Exception:
            pass
        
        return market_data
    
    def _evaluate_triggers(
        self,
        current_date: datetime,
        market_data: Dict[str, Any]
    ) -> List[TriggerLogEntry]:
        """Evaluate all triggers and return triggered events."""
        triggered = []
        
        if self.config.use_scheduled_reviews:
            if is_scheduled_review_day(current_date, self.config.scheduled_review_day):
                entry = TriggerLogEntry(
                    timestamp=current_date,
                    trigger_type="scheduled",
                    symbols_affected=self._get_all_symbols(),
                    agents_run=[a.value for a in AGENT_ROUTING[TriggerType.SCHEDULED]],
                    decision="run_agents",
                    details={"reason": "Monthly scheduled review"}
                )
                triggered.append(entry)
                self._trigger_log.append(entry)
        
        if self.config.include_max_drawdown_circuit_breaker:
            portfolio_value = self.broker.get_account_balance() + sum(
                p.market_value for p in self.broker.get_positions()
            )
            starting_capital = self.broker.starting_capital
            drawdown = (starting_capital - portfolio_value) / starting_capital * 100
            
            if drawdown >= self.config.max_drawdown_threshold:
                entry = TriggerLogEntry(
                    timestamp=current_date,
                    trigger_type="max_drawdown",
                    symbols_affected=[],
                    agents_run=[AgentName.RISK.value, AgentName.CIO.value],
                    decision="close_all",
                    details={
                        "drawdown_pct": drawdown,
                        "threshold": self.config.max_drawdown_threshold
                    }
                )
                triggered.append(entry)
                self._trigger_log.append(entry)
        
        if self.config.include_vix_triggers:
            if check_vix_trigger(
                market_data["vix_history"],
                market_data["vix"],
                self.config.vix_zscore_threshold,
                self.config.vix_history_window
            ):
                entry = TriggerLogEntry(
                    timestamp=current_date,
                    trigger_type="vix_spike",
                    symbols_affected=self._get_all_symbols(),
                    agents_run=[a.value for a in AGENT_ROUTING[TriggerType.VIX_SPIKE]],
                    decision="run_agents",
                    details={
                        "current_vix": market_data["vix"],
                        "threshold_type": "zscore"
                    }
                )
                triggered.append(entry)
                self._trigger_log.append(entry)
        
        if is_expiry_week(current_date):
            entry = TriggerLogEntry(
                timestamp=current_date,
                trigger_type="expiry_week",
                symbols_affected=[],
                agents_run=[a.value for a in AGENT_ROUTING[TriggerType.EXPIRY_WEEK]],
                decision="skip_derivatives",
                details={"is_expiry_week": True}
            )
            triggered.append(entry)
            self._trigger_log.append(entry)
        
        return triggered
    
    def _update_trailing_stops(
        self,
        current_date: datetime,
        market_data: Dict[str, Any]
    ) -> None:
        """Update trailing stops for all open positions."""
        for position in self.broker.get_positions():
            symbol = position.symbol
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{symbol}.NS")
                df = ticker.history(start=current_date - timedelta(days=30), end=current_date)
                
                if len(df) > 14:
                    high = df["High"]
                    low = df["Low"]
                    close = df["Close"]
                    
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
                    atr = float(tr_df.rolling(14).mean().iloc[-1])
                    
                    current_price = float(df["Close"].iloc[-1])
                    position.update_price(current_price)
                    
                    signal = self.signal_store.get(symbol)
                    if signal:
                        atr_multiplier = self.execution_engine.get_atr_multiplier(signal.cap_type)
                        new_stop = position.highest_price - (atr * atr_multiplier)
                        
                        if new_stop > signal.trailing_stop_pct:
                            signal.trailing_stop_pct = new_stop
                            position.trailing_stop_price = new_stop
                            self.signal_store.save(signal)
            except Exception:
                pass
    
    def _execute_trades(
        self,
        current_date: datetime,
        market_data: Dict[str, Any]
    ) -> List[ExecutionLogEntry]:
        """Execute any required trades based on current signals."""
        executions = []
        
        for position in self.broker.get_positions():
            symbol = position.symbol
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{symbol}.NS")
                df = ticker.history(start=current_date - timedelta(days=1), end=current_date)
                
                if len(df) > 0:
                    current_price = float(df["Close"].iloc[-1])
                    position.update_price(current_price)
                    
                    signal = self.signal_store.get(symbol)
                    
                    if signal and signal.signal in ["SELL", "STRONG_SELL"]:
                        result = self.broker.place_market_order(symbol, position.quantity, "sell")
                        
                        execution = ExecutionLogEntry(
                            timestamp=current_date,
                            action="EXIT",
                            symbol=symbol,
                            quantity=position.quantity,
                            price=result.price if result.status.value == "filled" else 0,
                            reason="signal_override",
                            success=result.status.value == "filled",
                            error="" if result.status.value == "filled" else result.message
                        )
                        executions.append(execution)
                        self._execution_log.append(execution)
                        
                        if result.status.value == "filled":
                            signal.position_open = False
                            self.signal_store.save(signal)
                    
                    elif position.trailing_stop_price > 0 and current_price <= position.trailing_stop_price:
                        result = self.broker.place_market_order(symbol, position.quantity, "sell")
                        
                        execution = ExecutionLogEntry(
                            timestamp=current_date,
                            action="EXIT",
                            symbol=symbol,
                            quantity=position.quantity,
                            price=result.price if result.status.value == "filled" else 0,
                            reason="trailing_stop",
                            success=result.status.value == "filled"
                        )
                        executions.append(execution)
                        self._execution_log.append(execution)
                        
                        if result.status.value == "filled":
                            signal.position_open = False
                            self.signal_store.save(signal)
            except Exception as e:
                execution = ExecutionLogEntry(
                    timestamp=current_date,
                    action="ERROR",
                    symbol=symbol,
                    quantity=0,
                    price=0,
                    reason="price_fetch_error",
                    success=False,
                    error=str(e)
                )
                executions.append(execution)
                self._execution_log.append(execution)
        
        return executions
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols from signal store."""
        return [s.symbol for s in self.signal_store.get_all()]
    
    def _get_status(
        self,
        current_date: datetime,
        triggers: List[TriggerLogEntry],
        executions: List[ExecutionLogEntry]
    ) -> EventLoopStatus:
        """Get current status of the event loop."""
        portfolio_value = self.broker.get_account_balance() + sum(
            p.market_value for p in self.broker.get_positions()
        )
        starting_capital = self.broker.starting_capital
        drawdown = (starting_capital - portfolio_value) / starting_capital * 100
        
        next_review = get_last_trading_day_of_month(current_date)
        if next_review <= current_date:
            from calendar import monthrange
            next_month = current_date.month + 1 if current_date.month < 12 else 1
            next_year = current_date.year if current_date.month < 12 else current_date.year + 1
            last_day = monthrange(next_year, next_month)[1]
            next_review = datetime(next_year, next_month, last_day)
        
        return EventLoopStatus(
            last_run=current_date,
            next_scheduled_review=next_review,
            active_signals=len(self.signal_store.get_active()),
            open_positions=len(self.broker.get_positions()),
            portfolio_value=portfolio_value,
            portfolio_drawdown_pct=drawdown,
            recent_triggers=self._trigger_log[-10:],
            recent_executions=self._execution_log[-10:],
            pending_reviews=[s.symbol for s in self.signal_store.get_active() if not s.position_open]
        )
    
    def _load_logs(self) -> None:
        """Load logs from disk."""
        if Path(self.config.trigger_log_path).exists():
            try:
                with open(self.config.trigger_log_path) as f:
                    data = json.load(f)
                    self._trigger_log = [
                        TriggerLogEntry(
                            timestamp=datetime.fromisoformat(e["timestamp"]),
                            trigger_type=e["trigger_type"],
                            symbols_affected=e["symbols_affected"],
                            agents_run=e["agents_run"],
                            decision=e["decision"],
                            details=e["details"]
                        ) for e in data
                    ]
            except Exception:
                self._trigger_log = []
        
        if Path(self.config.execution_log_path).exists():
            try:
                with open(self.config.execution_log_path) as f:
                    data = json.load(f)
                    self._execution_log = [
                        ExecutionLogEntry(
                            timestamp=datetime.fromisoformat(e["timestamp"]),
                            action=e["action"],
                            symbol=e["symbol"],
                            quantity=e["quantity"],
                            price=e["price"],
                            reason=e["reason"],
                            pnl=e.get("pnl", 0),
                            success=e.get("success", True),
                            error=e.get("error", "")
                        ) for e in data
                    ]
            except Exception:
                self._execution_log = []
    
    def _save_logs(self) -> None:
        """Save logs to disk."""
        Path(self.config.trigger_log_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config.trigger_log_path, "w") as f:
            json.dump([
                {
                    "timestamp": e.timestamp.isoformat(),
                    "trigger_type": e.trigger_type,
                    "symbols_affected": e.symbols_affected,
                    "agents_run": e.agents_run,
                    "decision": e.decision,
                    "details": e.details
                } for e in self._trigger_log[-1000:]
            ], f, indent=2)
        
        with open(self.config.execution_log_path, "w") as f:
            json.dump([
                {
                    "timestamp": e.timestamp.isoformat(),
                    "action": e.action,
                    "symbol": e.symbol,
                    "quantity": e.quantity,
                    "price": e.price,
                    "reason": e.reason,
                    "pnl": e.pnl,
                    "success": e.success,
                    "error": e.error
                } for e in self._execution_log[-1000:]
            ], f, indent=2)
    
    def get_trigger_history(
        self,
        days: int = 30,
        trigger_type: Optional[str] = None
    ) -> List[TriggerLogEntry]:
        """Get trigger history for the specified period."""
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = [e for e in self._trigger_log if e.timestamp >= cutoff]
        
        if trigger_type:
            filtered = [e for e in filtered if e.trigger_type == trigger_type]
        
        return filtered
    
    def get_execution_history(
        self,
        days: int = 30,
        symbol: Optional[str] = None
    ) -> List[ExecutionLogEntry]:
        """Get execution history for the specified period."""
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = [e for e in self._execution_log if e.timestamp >= cutoff]
        
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]
        
        return filtered
    
    def get_review_schedule(self) -> Dict[str, datetime]:
        """Get the review schedule for all active signals."""
        schedule = {}
        
        for signal in self.signal_store.get_active():
            schedule[signal.symbol] = signal.valid_until
        
        return schedule
    
    def get_next_review(self) -> Tuple[str, datetime]:
        """Get the symbol and date of the next scheduled review."""
        schedule = self.get_review_schedule()
        
        if not schedule:
            next_review = get_last_trading_day_of_month(datetime.now())
            return "NONE", next_review
        
        next_symbol = min(schedule, key=schedule.get)
        next_date = schedule[next_symbol]
        
        return next_symbol, next_date
    
    def get_trigger_summary(self) -> Dict[str, int]:
        """Get a summary of triggers by type."""
        summary: Dict[str, int] = {}
        
        for entry in self._trigger_log:
            summary[entry.trigger_type] = summary.get(entry.trigger_type, 0) + 1
        
        return summary
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of executions."""
        total = len(self._execution_log)
        successful = len([e for e in self._execution_log if e.success])
        failed = total - successful
        
        total_pnl = sum(e.pnl for e in self._execution_log if e.action == "EXIT")
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_pnl": total_pnl,
            "by_action": {
                "ENTRY": len([e for e in self._execution_log if e.action == "ENTRY"]),
                "EXIT": len([e for e in self._execution_log if e.action == "EXIT"]),
            }
        }


def run_daily_event_loop(
    config: Optional[EventLoopConfig] = None
) -> EventLoopStatus:
    """Convenience function to run the daily event loop."""
    loop = TradingEventLoop(config)
    return loop.run_daily()
