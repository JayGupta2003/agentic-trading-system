"""
Main Backtesting Engine.

Custom pandas-based backtester that processes LLM-generated signals against historical data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.backtesting.signals import Signal, TradingSignal, SignalProcessor
from src.backtesting.portfolio import PortfolioState
from src.backtesting.metrics import MetricsCalculator
from src.backtesting.risk import RiskManager, RiskLimits
from src.backtesting.data import DataLoader


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 1000000.0
    
    brokerage_rate: float = 0.001
    stt_rate: float = 0.001
    exchange_txn_charge: float = 0.0000345
    gst_rate: float = 0.18
    
    max_positions: int = 5
    max_position_pct: float = 20.0
    min_conviction: int = 6
    
    default_stop_loss_atr: float = 2.0
    default_trailing_stop_atr: float = 2.0
    default_target_pct: float = 10.0
    max_target_pct: float = 50.0
    
    warmup_days: int = 200
    signal_validity_days: int = 5
    
    risk_free_rate: float = 0.06
    max_drawdown_pct: float = 15.0
    max_daily_loss_pct: float = 5.0
    
    entry_type: str = "eod"
    exit_type: str = "eod"
    
    allow_short: bool = False
    allow_partial_exit: bool = True
    
    benchmark_symbol: str = "^NSEI"
    
    def __post_init__(self):
        self.risk_limits = RiskLimits(
            max_position_size_pct=self.max_position_pct,
            max_open_positions=self.max_positions,
            max_drawdown_pct=self.max_drawdown_pct,
            max_daily_loss_pct=self.max_daily_loss_pct,
            stop_loss_atr_multiplier=self.default_stop_loss_atr,
            trailing_stop_atr_multiplier=self.default_trailing_stop_atr,
        )
    
    def calculate_total_commission(self, trade_value: float) -> float:
        """Calculate total commission and charges for a trade."""
        brokerage = min(trade_value * self.brokerage_rate, 50)
        stt = trade_value * self.stt_rate
        exchange_txn = trade_value * self.exchange_txn_charge
        gst = (brokerage + exchange_txn) * self.gst_rate
        return brokerage + stt + exchange_txn + gst


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    trade_log: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "config": {
                "initial_capital": self.config.initial_capital,
                "max_positions": self.config.max_positions,
                "min_conviction": self.config.min_conviction,
            },
            "metrics": self.metrics,
            "total_trades": len(self.trades),
            "total_return_pct": self.metrics.get("total_return_pct", 0),
            "sharpe_ratio": self.metrics.get("sharpe_ratio", 0),
            "max_drawdown_pct": self.metrics.get("max_drawdown_pct", 0),
            "win_rate": self.metrics.get("win_rate", 0),
        }


class Backtester:
    """
    Custom pandas-based backtesting engine.
    
    Processes LLM-generated trading signals against historical data to evaluate
    strategy performance.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
        self.signal_processor = SignalProcessor(min_conviction=self.config.min_conviction)
        self.risk_manager = RiskManager(self.config.risk_limits)
        self.metrics_calculator = MetricsCalculator(risk_free_rate=self.config.risk_free_rate)
        self.data_loader = DataLoader()
        
        self.portfolio: Optional[PortfolioState] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.trade_log: List[dict] = []
        
        self._warmup_days = self.config.warmup_days
        self._executed_signals: set = set()
        self._processed_signals: Dict[str, set] = {}
    
    def run(
        self,
        symbols: List[str],
        signals: Dict[str, List[TradingSignal]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "3y",
        benchmark_symbol: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest with given symbols and signals.
        
        Args:
            symbols: List of stock symbols to backtest
            signals: Dict mapping symbol to list of TradingSignals with dates
            start_date: Backtest start date
            end_date: Backtest end date
            period: Data period if dates not provided
            benchmark_symbol: Symbol for benchmark comparison (e.g., "^NSEI")
        
        Returns:
            BacktestResult with metrics, trades, and equity curve
        """
        import time
        start_time = time.time()
        
        self._reset()
        
        self.data = self.data_loader.load_data(symbols, start_date, end_date, period)
        
        is_valid, issues = self.data_loader.validate_data(self.data)
        if not is_valid:
            print(f"Data validation issues: {issues}")
        
        for symbol in self.data:
            self.data[symbol] = self.data_loader.add_technical_indicators(self.data[symbol])
        
        self.portfolio = PortfolioState(
            initial_capital=self.config.initial_capital,
            brokerage_rate=self.config.brokerage_rate,
            max_positions=self.config.max_positions,
        )
        
        trading_dates = self._get_trading_dates()
        
        if len(trading_dates) <= self._warmup_days:
            print(f"Warning: Not enough trading days ({len(trading_dates)}) for warmup ({self._warmup_days})")
        
        warmup_end_idx = min(self._warmup_days, len(trading_dates) - 1)
        
        for i, date in enumerate(trading_dates):
            is_warmup = i < warmup_end_idx
            
            self._process_day(date, signals, is_warmup)
        
        self._close_all_positions(trading_dates[-1], "backtest_end")
        
        benchmark_returns = None
        if benchmark_symbol:
            benchmark_df = self.data_loader._load_symbol_data(
                benchmark_symbol, start_date, end_date, period
            )
            if benchmark_df is not None:
                benchmark_returns = benchmark_df['Close'].pct_change().dropna()
        
        self.metrics = self.metrics_calculator.calculate_all_metrics(
            self.portfolio, benchmark_returns
        )
        
        execution_time = time.time() - start_time
        
        return BacktestResult(
            config=self.config,
            metrics=self.metrics,
            trades=self.portfolio.get_trade_log_df(),
            equity_curve=self.portfolio.get_equity_curve_df(),
            trade_log=self.trade_log,
            execution_time_seconds=execution_time,
        )
    
    def _reset(self) -> None:
        """Reset state for new backtest."""
        self.portfolio = None
        self.data = {}
        self.trade_log = []
        self.risk_manager.reset_circuit_breaker()
        self._executed_signals = set()
        self._processed_signals = {}
    
    def _get_trading_dates(self) -> List[datetime]:
        """Get list of trading dates across all symbols."""
        if not self.data:
            return []
        
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index.tolist())
        
        return sorted(all_dates)
    
    def _process_day(
        self,
        date: datetime,
        signals: Dict[str, List[TradingSignal]],
        is_warmup: bool,
    ) -> None:
        """Process a single trading day."""
        prices = self.data_loader.get_prices_for_date(self.data, date)
        
        self.portfolio.record_equity(date, prices)
        
        if is_warmup:
            return
        
        self.risk_manager.update_peak(self.portfolio.portfolio_value)
        
        self._check_exits(date, prices, signals)
        
        if not self.risk_manager.is_trading_paused():
            self._check_entries(date, signals, prices)
    
    def _check_exits(self, date: datetime, prices: Dict[str, float], signals: Dict[str, List[TradingSignal]]) -> None:
        """Check and execute exits for open positions."""
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            df = self.data.get(symbol)
            
            atr = df['ATR'].asof(date) if df is not None and 'ATR' in df.columns else current_price * 0.02
            
            should_exit, reason = self.risk_manager.should_trigger_stop_loss(
                current_price,
                position.entry_price,
                atr,
                position.highest_price,
            )
            
            if not should_exit and position.target_price:
                should_exit, reason = self.risk_manager.should_trigger_target(
                    current_price,
                    position.target_price,
                    partial_exit=True,
                )
            
            if symbol in signals:
                for sig in signals[symbol]:
                    sig_date = sig.timestamp.date() if sig.timestamp else None
                    sig_id = f"{symbol}_{sig.timestamp.isoformat() if sig.timestamp else 'no_ts'}_exit"
                    
                    if sig_id in self._executed_signals:
                        continue
                    
                    # Exit on first trading day AFTER signal date (handles holidays/weekends)
                    if sig_date is not None and date.date() > sig_date and sig.signal in [Signal.SELL, Signal.STRONG_SELL]:
                        self._executed_signals.add(sig_id)
                        should_exit = True
                        reason = "signal_sell"
                        break
            
            if should_exit:
                trade = self.portfolio.close_position(symbol, current_price, date, reason)
                if trade:
                    self.trade_log.append({
                        "date": date.isoformat(),
                        "symbol": symbol,
                        "action": "EXIT",
                        "price": current_price,
                        "reason": reason,
                        "pnl": trade.pnl,
                        "pnl_pct": trade.pnl_pct,
                    })
    
    def _check_entries(
        self,
        date: datetime,
        signals: Dict[str, List[TradingSignal]],
        prices: Dict[str, float],
    ) -> None:
        """Check and execute entries for new positions."""
        if not self.portfolio.can_open_position():
            return
        
        for symbol, symbol_signals in signals.items():
            if not self.portfolio.can_open_position():
                break
            
            if symbol in self.portfolio.positions:
                continue
            
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            
            for sig in symbol_signals:
                sig_date = sig.timestamp.date() if sig.timestamp else None
                sig_id = f"{symbol}_{sig.timestamp.isoformat() if sig.timestamp else 'no_ts'}"
                
                if sig_id in self._executed_signals:
                    continue
                
                # Trigger entry on first trading day AFTER signal date (handles holidays/weekends)
                if sig_date is not None and date.date() > sig_date:
                    if sig.signal not in [Signal.BUY, Signal.STRONG_BUY]:
                        continue
                    
                    # Mark signal as processed to avoid duplicate entries
                    self._executed_signals.add(sig_id)
                    
                    df = self.data.get(symbol)
                    atr = df['ATR'].asof(date) if df is not None and 'ATR' in df.columns else current_price * 0.02
                    
                    position_size_pct = sig.position_size_pct if sig.position_size_pct > 0 else self.signal_processor.calculate_position_size(
                        sig.signal, sig.conviction, self.config.max_position_pct
                    )
                    position_value = self.portfolio.portfolio_value * (position_size_pct / 100)
                    quantity = int(position_value / current_price)
                    
                    if quantity <= 0:
                        continue
                    
                    stop_loss = current_price - (atr * self.config.default_stop_loss_atr)
                    target = sig.target_price or (current_price * (1 + self.config.default_target_pct / 100))
                    
                    success = self.portfolio.open_position(
                        symbol=symbol,
                        quantity=quantity,
                        price=current_price,
                        date=date,
                        stop_loss=stop_loss,
                        target=target,
                        reasoning=sig.reasoning,
                    )
                    
                    if success:
                        self.trade_log.append({
                            "date": date.isoformat(),
                            "symbol": symbol,
                            "action": "ENTRY",
                            "price": current_price,
                            "quantity": quantity,
                            "conviction": sig.conviction,
                            "reasoning": sig.reasoning[:100],
                        })
                    break
    
    def _close_all_positions(self, date: datetime, reason: str) -> None:
        """Close all open positions at end of backtest."""
        for symbol in list(self.portfolio.positions.keys()):
            if symbol in self.data:
                df = self.data[symbol]
                current_price = df['Close'].asof(date)
            else:
                current_price = self.portfolio.positions[symbol].entry_price
            
            trade = self.portfolio.close_position(symbol, current_price, date, reason)
            if trade:
                self.trade_log.append({
                    "date": date.isoformat(),
                    "symbol": symbol,
                    "action": "FORCE_EXIT",
                    "price": current_price,
                    "reason": reason,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                })
