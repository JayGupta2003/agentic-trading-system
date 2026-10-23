"""
Signal definitions and processing for backtesting.

Defines signal enums and signal processing logic for converting CIO output to trade signals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_SIGNAL = "NO_SIGNAL"


class TradeDirection(Enum):
    """Trade direction types."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """A trading signal with metadata."""
    symbol: str
    signal: Signal
    conviction: int
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_pct: float = 0.0
    timestamp: Optional[datetime] = None
    reasoning: str = ""
    agents_scores: Optional[Dict[str, float]] = None
    expires_days: int = 5
    
    def is_expired(self, current_date: datetime) -> bool:
        """Check if signal has expired."""
        if self.timestamp is None:
            return False
        days_since_signal = (current_date.date() - self.timestamp.date()).days
        return days_since_signal > self.expires_days
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "conviction": self.conviction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "position_size_pct": self.position_size_pct,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "reasoning": self.reasoning,
            "expires_days": self.expires_days,
        }


class SignalProcessor:
    """
    Processes and validates trading signals.
    
    Converts CIO output to standardized TradingSignal format.
    """
    
    DEFAULT_MIN_CONVICTION = 6
    DEFAULT_STRONG_BUY_THRESHOLD = 8
    
    def __init__(self, min_conviction: int = DEFAULT_MIN_CONVICTION):
        self.min_conviction = min_conviction
    
    def verdict_to_signal(self, verdict: str, conviction: int = 5) -> Signal:
        """Convert CIO verdict string to Signal enum."""
        verdict_upper = verdict.upper()
        
        if any(x in verdict_upper for x in ["STRONG BUY", "STRONG_BUY", "VERY STRONG BUY"]):
            return Signal.STRONG_BUY
        elif any(x in verdict_upper for x in ["STRONG SELL", "STRONG_SELL", "VERY STRONG SELL"]):
            return Signal.STRONG_SELL
        elif "BUY" in verdict_upper and "SELL" not in verdict_upper:
            return Signal.BUY
        elif "SELL" in verdict_upper:
            return Signal.SELL
        elif any(x in verdict_upper for x in ["HOLD", "NEUTRAL", "REDUCE"]):
            return Signal.HOLD
        else:
            return Signal.NO_SIGNAL
    
    def signal_to_direction(self, signal: Signal) -> TradeDirection:
        """Convert signal to trade direction."""
        if signal in [Signal.STRONG_BUY, Signal.BUY]:
            return TradeDirection.LONG
        elif signal in [Signal.STRONG_SELL, Signal.SELL]:
            return TradeDirection.EXIT
        else:
            return TradeDirection.HOLD
    
    def calculate_position_size(self, signal: Signal, conviction: int, max_size: float = 20.0) -> float:
        """Calculate position size based on signal and conviction."""
        if signal == Signal.STRONG_BUY:
            return min(max_size, conviction * 2.5)
        elif signal == Signal.BUY:
            return min(max_size, conviction * 1.5)
        elif signal == Signal.HOLD:
            return 0.0
        else:
            return 0.0
    
    def is_valid_signal(self, signal: Signal, conviction: int) -> bool:
        """Check if signal meets minimum requirements."""
        if signal in [Signal.STRONG_BUY, Signal.BUY]:
            return conviction >= self.min_conviction
        return True
    
    def process_cio_output(
        self,
        symbol: str,
        verdict: str,
        conviction: int,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reasoning: str = "",
    ) -> Optional[TradingSignal]:
        """
        Process CIO output into a TradingSignal.
        
        Args:
            symbol: Stock symbol
            verdict: CIO verdict string
            conviction: Conviction score (1-10)
            entry_price: Suggested entry price
            target_price: Target price from CIO
            stop_loss: Stop loss price
            reasoning: CIO reasoning text
        
        Returns:
            TradingSignal if valid, None otherwise
        """
        signal = self.verdict_to_signal(verdict, conviction)
        
        if not self.is_valid_signal(signal, conviction):
            return None
        
        position_size = self.calculate_position_size(signal, conviction)
        
        return TradingSignal(
            symbol=symbol,
            signal=signal,
            conviction=conviction,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size_pct=position_size,
            timestamp=datetime.now(),
            reasoning=reasoning,
        )
    
    def get_signal_priority(self, signal: Signal) -> int:
        """
        Get priority rank for signal (higher = more urgent).
        
        Priority: STRONG_SELL > STRONG_BUY > SELL > BUY > HOLD
        """
        priority_map = {
            Signal.STRONG_SELL: 5,
            Signal.STRONG_BUY: 4,
            Signal.SELL: 3,
            Signal.BUY: 2,
            Signal.HOLD: 1,
            Signal.NO_SIGNAL: 0,
        }
        return priority_map.get(signal, 0)
    
    def resolve_conflict(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Resolve conflicts when multiple signals exist for same symbol.
        
        Priority: STRONG_SELL > STRONG_BUY > SELL > BUY > HOLD
        Only the highest priority signal is kept.
        """
        if len(signals) <= 1:
            return signals
        
        signals.sort(key=lambda s: self.get_signal_priority(s.signal), reverse=True)
        return [signals[0]]
