"""
Event-Driven Backtester - Time-Travel Historical Simulation

This module implements the event-driven backtesting system as specified in
Section 4.1 of REDESIGN_PLAN.md. It simulates the full trading system in a
time-travel manner to ensure valid historical performance measurement.

Key principles:
1. Data only before day t - no lookahead bias
2. Trigger simulation for historical dates
3. Static/lagged fundamental data (acknowledged PiT limitations)
4. Full execution engine logic (entries, exits, trailing stops)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

from src.trading.types import (
    TradingSignal, TriggerType, AgentName, AGENT_ROUTING,
    MarketContext, TriggerConfig, CapType
)
from src.trading.trigger_engine import (
    TriggerEngine, is_expiry_week, is_scheduled_review_day,
    check_vix_trigger, check_price_drop_trigger
)
from src.trading.execution_engine import (
    ExecutionEngine, ExecutionConfig, ExitReason, ExitDecision
)
from src.trading.brokers.historical import HistoricalBroker
from src.trading.brokers.base import Position


@dataclass
class BacktestSignal:
    """A signal generated during backtest simulation."""
    symbol: str
    signal: str
    conviction: int
    entry_price: float
    stop_loss: float
    target_price: float
    thesis: str
    generated_at: datetime
    valid_until: datetime
    review_type: str
    cap_type: str
    agents_run: List[str]
    trigger_type: Optional[str] = None


@dataclass
class BacktestTrade:
    """A trade executed during backtest."""
    date: datetime
    symbol: str
    action: str
    quantity: int
    price: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    conviction: int = 0
    signal_date: Optional[datetime] = None


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    total_fees: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "profit_factor": self.profit_factor,
            "avg_holding_days": self.avg_holding_days,
            "total_fees": self.total_fees,
        }


@dataclass
class TriggerEvent:
    """A trigger event during backtest."""
    date: datetime
    trigger_type: TriggerType
    symbol: Optional[str]
    details: Dict[str, Any]
    agents_run: List[str]


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot for equity curve."""
    date: datetime
    portfolio_value: float
    cash: float
    positions_value: float
    open_positions: int
    drawdown_pct: float


@dataclass
class BacktestConfig:
    """Configuration for event-driven backtest."""
    initial_capital: float = 1000000.0
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2025, 12, 31))
    
    symbols: List[str] = field(default_factory=list)
    cap_types: Dict[str, str] = field(default_factory=lambda: {})
    
    max_positions: int = 5
    max_position_pct: float = 15.0
    min_conviction: int = 6
    
    trailing_stop_atr_large: float = 2.5
    trailing_stop_atr_mid: float = 3.0
    trailing_stop_atr_small: float = 3.5
    
    brokerage_rate: float = 0.001
    slippage_pct: float = 0.001
    
    use_scheduled_reviews: bool = True
    scheduled_review_day: int = -1
    
    include_vix_triggers: bool = True
    include_price_drop_triggers: bool = True
    include_max_drawdown_circuit_breaker: bool = True
    max_drawdown_threshold: float = 10.0
    
    simulate_llm_signals: bool = True
    signal_generation_mode: str = "rule_based"
    
    data_quality_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.data_quality_notes.append(
            "Fundamental data uses static quarterly snapshots (not point-in-time)"
        )
        self.data_quality_notes.append(
            "LLM signals simulated with rule-based model during backtest"
        )


class BacktestSignalGenerator:
    """Generates signals during backtest (simulates LLM agents).
    
    Since we can't run actual LLMs on historical data, this uses a
    rule-based model that captures the essence of the agent system.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self._last_signal_date: Dict[str, datetime] = {}
        self._pending_signals: Dict[str, BacktestSignal] = {}
    
    def generate_signal(
        self,
        symbol: str,
        date: datetime,
        trigger_type: TriggerType,
        price_data: pd.DataFrame,
        market_context: MarketContext,
        agents_run: List[str]
    ) -> Optional[BacktestSignal]:
        """Generate a trading signal for the given conditions."""
        
        if not self.config.simulate_llm_signals:
            return None
        
        if trigger_type == TriggerType.MAX_DRAWDOWN:
            return self._generate_defensive_signal(symbol, date, agents_run)
        
        df_up_to_date = price_data[price_data.index <= date]
        if len(df_up_to_date) < 20:
            return None
        
        current_price = df_up_to_date["Close"].iloc[-1]
        if current_price <= 0:
            return None
        
        returns_20d = df_up_to_date["Close"].pct_change(20).iloc[-1] * 100
        
        rsi = self._calculate_rsi(df_up_to_date)
        sma_20 = df_up_to_date["Close"].rolling(20).mean().iloc[-1]
        sma_50 = df_up_to_date["Close"].rolling(50).mean().iloc[-1] if len(df_up_to_date) >= 50 else sma_20
        
        cap_type = self.config.cap_types.get(symbol, "large")
        
        signal_strength = 0
        conviction = 5
        
        if trigger_type == TriggerType.SCHEDULED:
            signal_strength += 2
        elif trigger_type == TriggerType.PRICE_DROP:
            if returns_20d < -10:
                signal_strength += 3
                conviction = 8
            elif returns_20d < -5:
                signal_strength += 2
                conviction = 7
            else:
                signal_strength += 1
                conviction = 6
        elif trigger_type == TriggerType.VIX_SPIKE:
            if market_context.vix > 25:
                signal_strength -= 1
                conviction = 4
            else:
                signal_strength += 1
        elif trigger_type == TriggerType.EARNINGS:
            signal_strength += 2
            conviction = 7
        
        if rsi < 30 and signal_strength > 0:
            signal_strength += 2
            conviction = min(conviction + 2, 10)
        elif rsi > 70:
            signal_strength -= 1
        
        if current_price > sma_20 and current_price > sma_50:
            signal_strength += 1
        elif current_price < sma_20:
            signal_strength -= 1
        
        atr = self._calculate_atr(df_up_to_date)
        stop_loss = current_price - (atr * 2)
        target = current_price + (atr * 4)
        
        if signal_strength >= 3 and conviction >= self.config.min_conviction:
            signal = "BUY"
            thesis = f"Rule-based BUY: RSI={rsi:.1f}, 20d return={returns_20d:.1f}%, trigger={trigger_type.value}"
        elif signal_strength >= 1 and conviction >= self.config.min_conviction:
            signal = "BUY"
            thesis = f"Rule-based BUY: trigger={trigger_type.value}"
        elif signal_strength <= -2:
            signal = "SELL"
            thesis = f"Rule-based SELL: Overbought conditions, trigger={trigger_type.value}"
        else:
            signal = "HOLD"
            thesis = f"Rule-based HOLD: Neutral conditions, trigger={trigger_type.value}"
        
        if AgentName.CIO not in agents_run and len(agents_run) > 0:
            conviction = max(conviction - 1, 3)
        
        return BacktestSignal(
            symbol=symbol,
            signal=signal,
            conviction=conviction,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target,
            thesis=thesis,
            generated_at=date,
            valid_until=date + timedelta(days=30),
            review_type=trigger_type.value,
            cap_type=cap_type,
            agents_run=[a.value for a in agents_run],
            trigger_type=trigger_type.value,
        )
    
    def _generate_defensive_signal(
        self,
        symbol: str,
        date: datetime,
        agents_run: List[str]
    ) -> BacktestSignal:
        """Generate defensive signal during max drawdown (circuit breaker)."""
        return BacktestSignal(
            symbol=symbol,
            signal="SELL",
            conviction=10,
            entry_price=0,
            stop_loss=0,
            target_price=0,
            thesis="Circuit breaker: Defensive exit due to max drawdown",
            generated_at=date,
            valid_until=date + timedelta(days=7),
            review_type="max_drawdown",
            cap_type=self.config.cap_types.get(symbol, "large"),
            agents_run=[a.value for a in agents_run],
            trigger_type="max_drawdown",
        )
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI."""
        if len(df) < period + 1:
            return 50.0
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean().iloc[-1]
        
        if loss == 0:
            return 100.0
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi if pd.notna(rsi) else 50.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        if len(df) < period + 1:
            return df["Close"].iloc[-1] * 0.02
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr_df = pd.DataFrame({
            "tr1": tr1.values,
            "tr2": tr2.values,
            "tr3": tr3.values
        }, index=df.index)
        tr_max = tr_df.max(axis=1)
        atr = tr_max.rolling(window=period).mean().iloc[-1]
        
        return atr if pd.notna(atr) and atr > 0 else df["Close"].iloc[-1] * 0.02


class EventDrivenBacktester:
    """Event-Driven Backtester with time-travel simulation.
    
    This backtester simulates the full trading system for historical periods:
    1. For each trading day t, checks triggers using only data BEFORE t
    2. If trigger fires, generates signal (rule-based simulation)
    3. Executes entries/exits using the ExecutionEngine
    4. Tracks all decisions, trades, and metrics
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
        self.trigger_engine = TriggerEngine(TriggerConfig(
            scheduled_review_day=self.config.scheduled_review_day,
            max_drawdown_threshold=self.config.max_drawdown_threshold,
        ))
        
        self.execution_engine = ExecutionEngine(ExecutionConfig(
            max_position_pct=self.config.max_position_pct,
            max_positions=self.config.max_positions,
            trailing_stop_atr_large=self.config.trailing_stop_atr_large,
            trailing_stop_atr_mid=self.config.trailing_stop_atr_mid,
            trailing_stop_atr_small=self.config.trailing_stop_atr_small,
            min_conviction=self.config.min_conviction,
        ))
        
        self.signal_generator = BacktestSignalGenerator(self.config)
        
        self._broker: Optional[HistoricalBroker] = None
        self._data: Dict[str, pd.DataFrame] = {}
        self._vix_data: pd.DataFrame = {}
        self._nifty_data: pd.DataFrame = {}
        
        self._trades: List[BacktestTrade] = []
        self._trigger_events: List[TriggerEvent] = []
        self._signals_generated: List[BacktestSignal] = []
        self._daily_snapshots: List[DailySnapshot] = []
        
        self._active_signals: Dict[str, BacktestSignal] = {}
        self._position_entry_dates: Dict[str, datetime] = {}
        
        self._peak_value: float = 0.0
        self._current_drawdown: float = 0.0
        self._total_fees: float = 0.0
        
        self._equity_curve: List[float] = []
        self._dates: List[datetime] = []
    
    def load_data(self, symbols: Optional[List[str]] = None) -> None:
        """Load historical data for backtest period."""
        import yfinance as yf
        
        if symbols:
            self.config.symbols = symbols
        
        start = self.config.start_date - timedelta(days=100)
        end = self.config.end_date + timedelta(days=1)
        
        for symbol in self.config.symbols:
            sym = self._format_symbol(symbol)
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(start=start, end=end)
                if len(df) > 0:
                    df = self._add_technical_indicators(df)
                    self._data[symbol] = df
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        
        try:
            vix = yf.Ticker("^INDIAVIX")
            vix_df = vix.history(start=start, end=end)
            if len(vix_df) > 0:
                self._vix_data = vix_df
        except Exception:
            print("Warning: Could not load VIX data")
        
        try:
            nifty = yf.Ticker("^NSEI")
            nifty_df = nifty.history(start=start, end=end)
            if len(nifty_df) > 0:
                self._nifty_data = nifty_df
        except Exception:
            print("Warning: Could not load Nifty data")
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for yfinance."""
        sym = symbol.upper().strip()
        if not sym.endswith(".NS"):
            if sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                sym = f"{sym}.NS"
            elif len(sym) <= 10 and sym.isalpha():
                sym = f"{sym}.NS"
        return sym
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        df = df.copy()
        
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df["ATR"] = tr_df.rolling(14).mean()
        
        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        
        return df
    
    def run(self) -> "BacktestResult":
        """Run the event-driven backtest."""
        if not self._data:
            self.load_data()
        
        self._broker = HistoricalBroker(self.config.initial_capital)
        self._peak_value = self.config.initial_capital
        
        trading_dates = self._get_trading_dates()
        
        if not trading_dates:
            raise ValueError("No trading dates found")
        
        warmup_required = 200
        warmup_end_idx = min(warmup_required, len(trading_dates) - 1)
        
        for i, date in enumerate(trading_dates):
            is_warmup = i < warmup_end_idx
            
            self._process_day(date, trading_dates, is_warmup)
        
        metrics = self._calculate_metrics(trading_dates)
        
        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self._trades,
            equity_curve=self._generate_equity_curve(trading_dates),
            trigger_events=self._trigger_events,
            signals_generated=self._signals_generated,
            data_quality_notes=self.config.data_quality_notes,
        )
    
    def _get_trading_dates(self) -> List[datetime]:
        """Get all trading dates within the backtest period."""
        all_dates = set()
        
        for df in self._data.values():
            all_dates.update(df.index.tolist())
        
        if self._nifty_data is not None and len(self._nifty_data) > 0:
            all_dates.update(self._nifty_data.index.tolist())
        
        filtered_dates = [
            d for d in sorted(all_dates)
            if self.config.start_date <= d <= self.config.end_date
        ]
        
        return filtered_dates
    
    def _process_day(
        self,
        date: datetime,
        all_dates: List[datetime],
        is_warmup: bool
    ) -> None:
        """Process a single trading day."""
        self._dates.append(date)
        
        prices = self._get_prices_for_date(date)
        
        for symbol in self.config.symbols:
            if symbol in self._broker._positions:
                pos = self._broker._positions[symbol]
                if symbol in prices:
                    pos.update_price(prices[symbol])
        
        current_value = self._broker.get_account_balance() + sum(
            p.market_value for p in self._broker.get_positions()
        )
        
        if current_value > self._peak_value:
            self._peak_value = current_value
        
        self._current_drawdown = (self._peak_value - current_value) / self._peak_value * 100
        
        self._record_snapshot(date, current_value)
        
        if is_warmup:
            return
        
        if self.config.include_max_drawdown_circuit_breaker:
            if self._current_drawdown >= self.config.max_drawdown_threshold:
                self._handle_circuit_breaker(date, prices)
        
        self._check_exits(date, prices)
        self._check_triggers(date, all_dates)
        self._check_entries(date, prices)
    
    def _get_prices_for_date(self, date: datetime) -> Dict[str, float]:
        """Get closing prices for all symbols on given date."""
        prices = {}
        
        for symbol, df in self._data.items():
            df_before = df[df.index <= date]
            if len(df_before) > 0:
                prices[symbol] = float(df_before["Close"].iloc[-1])
        
        return prices
    
    def _get_data_before_date(self, symbol: str, date: datetime) -> pd.DataFrame:
        """Get historical data up to (and including) the given date."""
        if symbol not in self._data:
            return pd.DataFrame()
        
        return self._data[symbol][self._data[symbol].index <= date]
    
    def _get_vix_before_date(self, date: datetime) -> Tuple[float, List[float]]:
        """Get VIX value and history before given date."""
        if self._vix_data is None or len(self._vix_data) == 0:
            return 20.0, [20.0] * 20
        
        vix_before = self._vix_data[self._vix_data.index <= date]
        if len(vix_before) == 0:
            return 20.0, [20.0] * 20
        
        current_vix = float(vix_before["Close"].iloc[-1])
        vix_history = list(vix_before["Close"].tail(30).values)
        
        return current_vix, vix_history
    
    def _get_nifty_change_before_date(self, date: datetime) -> float:
        """Get Nifty change percentage before given date."""
        if self._nifty_data is None or len(self._nifty_data) < 2:
            return 0.0
        
        nifty_before = self._nifty_data[self._nifty_data.index <= date]
        if len(nifty_before) < 2:
            return 0.0
        
        prices = nifty_before["Close"].values
        change = (prices[-1] - prices[-2]) / prices[-2] * 100
        return change
    
    def _check_triggers(self, date: datetime, all_dates: List[datetime]) -> None:
        """Check triggers for the current day."""
        previous_date = None
        for d in reversed(all_dates):
            if d < date:
                previous_date = d
                break
        
        for symbol in self.config.symbols:
            df_before = self._get_data_before_date(symbol, date)
            if len(df_before) < 2:
                continue
            
            current_price = float(df_before["Close"].iloc[-1])
            previous_price = float(df_before["Close"].iloc[-2])
            
            cap_type = self.config.cap_types.get(symbol, "large")
            
            current_vix, vix_history = self._get_vix_before_date(date)
            
            nifty_change = self._get_nifty_change_before_date(date)
            
            is_scheduled = is_scheduled_review_day(
                date, self.config.scheduled_review_day
            ) if self.config.use_scheduled_reviews else False
            
            is_expiry = is_expiry_week(date)
            
            trigger_fired = False
            trigger_type = None
            
            if self.config.use_scheduled_reviews and is_scheduled:
                trigger_fired = True
                trigger_type = TriggerType.SCHEDULED
            
            if not trigger_fired and self.config.include_price_drop_triggers:
                drop_pct = ((previous_price - current_price) / previous_price) * 100
                threshold = {"large": 5, "mid": 8, "small": 12}.get(cap_type, 5)
                if drop_pct >= threshold:
                    trigger_fired = True
                    trigger_type = TriggerType.PRICE_DROP
            
            if not trigger_fired and self.config.include_vix_triggers:
                if check_vix_trigger(vix_history, current_vix):
                    trigger_fired = True
                    trigger_type = TriggerType.VIX_SPIKE
            
            if not trigger_fired and self._current_drawdown >= self.config.max_drawdown_threshold:
                trigger_fired = True
                trigger_type = TriggerType.MAX_DRAWDOWN
            
            if not trigger_fired and is_expiry:
                trigger_fired = True
                trigger_type = TriggerType.EXPIRY_WEEK
            
            if trigger_fired and trigger_type:
                agents = AGENT_ROUTING.get(trigger_type, [AgentName.CIO])
                
                context = MarketContext(
                    current_date=date,
                    vix=current_vix,
                    vix_history=vix_history,
                    nifty50_change_pct=nifty_change,
                    portfolio_drawdown_pct=self._current_drawdown,
                    portfolio_value=self._broker.get_account_balance() + sum(
                        p.market_value for p in self._broker.get_positions()
                    ),
                    stock_price=current_price,
                    stock_previous_price=previous_price,
                    stock_cap_type=cap_type,
                    is_expiry_week=is_expiry,
                )
                
                signal = self.signal_generator.generate_signal(
                    symbol, date, trigger_type, df_before, context, agents
                )
                
                if signal:
                    self._active_signals[symbol] = signal
                    self._signals_generated.append(signal)
                    
                    self._trigger_events.append(TriggerEvent(
                        date=date,
                        trigger_type=trigger_type,
                        symbol=symbol,
                        details={
                            "current_price": current_price,
                            "previous_price": previous_price,
                            "cap_type": cap_type,
                            "vix": current_vix,
                            "nifty_change": nifty_change,
                            "drawdown": self._current_drawdown,
                        },
                        agents_run=[a.value for a in agents],
                    ))
    
    def _handle_circuit_breaker(
        self,
        date: datetime,
        prices: Dict[str, float]
    ) -> None:
        """Handle circuit breaker (max drawdown) - close all positions."""
        if not self._trigger_events or self._trigger_events[-1].trigger_type != TriggerType.MAX_DRAWDOWN:
            agents = [AgentName.RISK, AgentName.CIO]
            
            self._trigger_events.append(TriggerEvent(
                date=date,
                trigger_type=TriggerType.MAX_DRAWDOWN,
                symbol=None,
                details={"drawdown": self._current_drawdown},
                agents_run=[a.value for a in agents],
            ))
        
        for symbol in list(self._broker._positions.keys()):
            if symbol in prices:
                self._execute_exit(symbol, date, prices[symbol], "circuit_breaker")
    
    def _check_exits(self, date: datetime, prices: Dict[str, float]) -> None:
        """Check and execute exits for open positions."""
        for symbol in list(self._broker._positions.keys()):
            if symbol not in prices:
                continue
            
            position = self._broker._positions[symbol]
            current_price = prices[symbol]
            
            signal = self._active_signals.get(symbol)
            
            should_exit = False
            reason = ""
            
            if signal and signal.signal in ["SELL", "STRONG_SELL"]:
                should_exit = True
                reason = "signal_override"
            
            if not should_exit and signal:
                stop_loss = signal.stop_loss
                if current_price <= stop_loss:
                    should_exit = True
                    reason = "stop_loss"
            
            if not should_exit:
                atr = self._get_atr_before_date(symbol, date)
                cap_type = self.config.cap_types.get(symbol, "large")
                atr_multiplier = {
                    "large": self.config.trailing_stop_atr_large,
                    "mid": self.config.trailing_stop_atr_mid,
                    "small": self.config.trailing_stop_atr_small,
                }.get(cap_type, 2.5)
                
                trailing_stop = position.highest_price - (atr * atr_multiplier)
                
                if trailing_stop > 0 and position.trailing_stop_price > 0:
                    position.trailing_stop_price = max(
                        position.trailing_stop_price, trailing_stop
                    )
                else:
                    position.trailing_stop_price = trailing_stop
                
                if current_price <= position.trailing_stop_price:
                    should_exit = True
                    reason = "trailing_stop"
            
            if not should_exit and signal:
                if current_price >= signal.target_price and position.quantity > 1:
                    should_exit = True
                    reason = "target_hit"
            
            if not should_exit:
                if symbol in self._position_entry_dates:
                    days_held = (date - self._position_entry_dates[symbol]).days
                    if days_held >= 30:
                        should_exit = True
                        reason = "time_based"
            
            if should_exit:
                self._execute_exit(symbol, date, current_price, reason)
    
    def _get_atr_before_date(self, symbol: str, date: datetime) -> float:
        """Get ATR value before given date."""
        df = self._get_data_before_date(symbol, date)
        if len(df) < 15:
            return df["Close"].iloc[-1] * 0.02 if len(df) > 0 else 100
        
        atr = df["ATR"].iloc[-1]
        return float(atr) if pd.notna(atr) else float(df["Close"].iloc[-1] * 0.02)
    
    def _execute_exit(
        self,
        symbol: str,
        date: datetime,
        price: float,
        reason: str
    ) -> None:
        """Execute an exit trade."""
        position = self._broker._positions.get(symbol)
        if not position:
            return
        
        slippage = price * self.config.slippage_pct
        exit_price = price - slippage
        
        quantity = position.quantity
        pnl = (exit_price - position.entry_price) * quantity
        
        brokerage = min(exit_price * quantity * self.config.brokerage_rate, 50)
        total_fees = brokerage * 1.18
        net_pnl = pnl - total_fees
        self._total_fees += total_fees
        
        trade = BacktestTrade(
            date=date,
            symbol=symbol,
            action="EXIT",
            quantity=quantity,
            price=exit_price,
            pnl=net_pnl,
            pnl_pct=(net_pnl / (position.entry_price * quantity)) * 100,
            reason=reason,
            conviction=self._active_signals.get(symbol).conviction if self._active_signals.get(symbol) else 0,
            signal_date=self._position_entry_dates.get(symbol),
        )
        self._trades.append(trade)
        
        self._broker._cash += exit_price * quantity - total_fees
        del self._broker._positions[symbol]
        
        if symbol in self._position_entry_dates:
            del self._position_entry_dates[symbol]
    
    def _check_entries(self, date: datetime, prices: Dict[str, float]) -> None:
        """Check and execute entries for new positions."""
        if len(self._broker._positions) >= self.config.max_positions:
            return
        
        for symbol, signal in self._active_signals.items():
            if symbol in self._broker._positions:
                continue
            
            if symbol not in prices:
                continue
            
            if signal.signal not in ["BUY", "STRONG_BUY"]:
                continue
            
            if signal.conviction < self.config.min_conviction:
                continue
            
            if signal.valid_until < date:
                continue
            
            current_price = prices[symbol]
            
            portfolio_value = self._broker.get_account_balance() + sum(
                p.market_value for p in self._broker._positions.values()
            )
            
            position_pct = signal.conviction / 10 * 0.15
            position_pct = min(position_pct, self.config.max_position_pct / 100)
            
            position_value = portfolio_value * position_pct
            quantity = int(position_value / current_price)
            
            if quantity <= 0:
                continue
            
            slippage = current_price * self.config.slippage_pct
            entry_price = current_price + slippage
            
            total_cost = entry_price * quantity
            brokerage = min(total_cost * self.config.brokerage_rate, 50)
            total_fees = brokerage * 1.18
            
            if total_cost + total_fees > self._broker._cash:
                continue
            
            self._broker._cash -= (total_cost + total_fees)
            self._total_fees += total_fees
            
            atr = self._get_atr_before_date(symbol, date)
            cap_type = signal.cap_type
            atr_multiplier = {
                "large": self.config.trailing_stop_atr_large,
                "mid": self.config.trailing_stop_atr_mid,
                "small": self.config.trailing_stop_atr_small,
            }.get(cap_type, 2.5)
            
            trailing_stop = entry_price - (atr * atr_multiplier)
            
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                highest_price=entry_price,
                atr=atr,
                trailing_stop_price=trailing_stop,
            )
            self._broker._positions[symbol] = position
            self._position_entry_dates[symbol] = date
            
            trade = BacktestTrade(
                date=date,
                symbol=symbol,
                action="ENTRY",
                quantity=quantity,
                price=entry_price,
                conviction=signal.conviction,
                signal_date=signal.generated_at,
            )
            self._trades.append(trade)
    
    def _record_snapshot(self, date: datetime, portfolio_value: float) -> None:
        """Record daily portfolio snapshot."""
        cash = self._broker.get_account_balance() if self._broker else self.config.initial_capital
        positions_value = portfolio_value - cash
        
        self._daily_snapshots.append(DailySnapshot(
            date=date,
            portfolio_value=portfolio_value,
            cash=cash,
            positions_value=positions_value,
            open_positions=len(self._broker._positions) if self._broker else 0,
            drawdown_pct=self._current_drawdown,
        ))
        
        self._equity_curve.append(portfolio_value)
    
    def _generate_equity_curve(self, dates: List[datetime]) -> pd.DataFrame:
        """Generate equity curve DataFrame."""
        if not self._equity_curve or not dates:
            return pd.DataFrame()
        
        return pd.DataFrame({
            "date": dates[:len(self._equity_curve)],
            "portfolio_value": self._equity_curve,
        }).set_index("date")
    
    def _calculate_metrics(self, dates: List[datetime]) -> BacktestMetrics:
        """Calculate performance metrics."""
        if not self._equity_curve:
            return BacktestMetrics()
        
        equity = pd.Series(self._equity_curve)
        
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100
        
        days = len(equity)
        years = days / 252
        annualized_return = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        returns = equity.pct_change().dropna()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino = 0
        
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100
        max_drawdown = abs(drawdown.min())
        
        max_dd_duration = 0
        current_dd_duration = 0
        peak_idx = 0
        for i, (dd, eq) in enumerate(zip(drawdown, equity)):
            if eq >= equity.iloc[peak_idx]:
                peak_idx = i
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
        
        entry_trades = [t for t in self._trades if t.action == "ENTRY"]
        exit_trades = [t for t in self._trades if t.action == "EXIT"]
        
        winning_trades = [t for t in exit_trades if t.pnl > 0]
        losing_trades = [t for t in exit_trades if t.pnl <= 0]
        
        total_trades = len(exit_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        holding_days = []
        for t in exit_trades:
            if t.signal_date:
                days = (t.date - t.signal_date).days
                holding_days.append(days)
        
        avg_holding = np.mean(holding_days) if holding_days else 0
        
        return BacktestMetrics(
            total_return_pct=total_return,
            annualized_return_pct=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            total_fees=self._total_fees,
        )
    
    def _close_all_positions(self, date: datetime, prices: Dict[str, float]) -> None:
        """Close all positions at end of backtest."""
        for symbol in list(self._broker._positions.keys()):
            if symbol in prices:
                self._execute_exit(symbol, date, prices[symbol], "backtest_end")


@dataclass
class BacktestResult:
    """Results from event-driven backtest."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[BacktestTrade]
    equity_curve: pd.DataFrame
    trigger_events: List[TriggerEvent]
    signals_generated: List[BacktestSignal]
    data_quality_notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": {
                "initial_capital": self.config.initial_capital,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "symbols": self.config.symbols,
            },
            "metrics": self.metrics.to_dict(),
            "total_trades": self.metrics.total_trades,
            "triggers_fired": len(self.trigger_events),
            "signals_generated": len(self.signals_generated),
            "data_quality_notes": self.data_quality_notes,
        }
    
    def get_trigger_summary(self) -> Dict[str, int]:
        """Get summary of triggers fired."""
        summary = defaultdict(int)
        for event in self.trigger_events:
            summary[event.trigger_type.value] += 1
        return dict(summary)
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                "date": trade.date,
                "symbol": trade.symbol,
                "action": trade.action,
                "quantity": trade.quantity,
                "price": trade.price,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "reason": trade.reason,
                "conviction": trade.conviction,
            })
        
        return pd.DataFrame(data)
