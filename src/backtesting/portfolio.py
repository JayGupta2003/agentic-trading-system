"""
Portfolio state management for backtesting.

Manages positions, trades, and portfolio equity tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


@dataclass
class Position:
    """Represents an open position in a stock."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    def update(self, current_price: float, current_date: datetime) -> None:
        """Update position with current price."""
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        if current_price > self.highest_price:
            self.highest_price = current_price
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat(),
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
        }


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    direction: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    reasoning: str = ""
    exit_reason: str = ""
    
    @property
    def duration_days(self) -> int:
        return (self.exit_date - self.entry_date).days
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": self.entry_date.isoformat(),
            "exit_date": self.exit_date.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "duration_days": self.duration_days,
            "exit_reason": self.exit_reason,
            "reasoning": self.reasoning,
        }


class PortfolioState:
    """
    Manages portfolio state including cash, positions, and equity curve.
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        brokerage_rate: float = 0.001,
        max_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.brokerage_rate = brokerage_rate
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        
        self.total_pnl = 0.0
        self.total_commission = 0.0
    
    @property
    def portfolio_value(self) -> float:
        """Current total portfolio value including cash and positions."""
        positions_value = sum(
            pos.unrealized_pnl + (pos.entry_price * pos.quantity)
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """Total return in rupees."""
        return self.portfolio_value - self.initial_capital
    
    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        return (self.total_return / self.initial_capital) * 100
    
    @property
    def open_positions_count(self) -> int:
        return len(self.positions)
    
    @property
    def available_capital(self) -> float:
        """Capital available for new positions."""
        return self.cash * (1 / self.max_positions) if self.open_positions_count == 0 else 0
    
    def can_open_position(self) -> bool:
        """Check if a new position can be opened."""
        return self.open_positions_count < self.max_positions and self.cash > 0
    
    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        date: datetime,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        reasoning: str = "",
    ) -> bool:
        """
        Open a new position.
        
        Returns:
            True if position opened successfully
        """
        if symbol in self.positions:
            return False
        
        if not self.can_open_position():
            return False
        
        cost = quantity * price
        commission = cost * self.brokerage_rate
        total_cost = cost + commission
        
        if total_cost > self.cash:
            quantity = int(self.cash / (price * (1 + self.brokerage_rate)))
            if quantity <= 0:
                return False
            cost = quantity * price
            commission = cost * self.brokerage_rate
            total_cost = cost + commission
        
        self.cash -= total_cost
        self.total_commission += commission
        
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=date,
            stop_loss=stop_loss,
            target_price=target,
            highest_price=price,
        )
        
        return True
    
    def close_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        reason: str = "",
    ) -> Optional[Trade]:
        """
        Close an existing position.
        
        Returns:
            Trade object if position closed, None otherwise
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        proceeds = position.quantity * price
        commission = proceeds * self.brokerage_rate
        net_proceeds = proceeds - commission
        
        pnl = net_proceeds - (position.entry_price * position.quantity)
        pnl_pct = (price - position.entry_price) / position.entry_price * 100
        
        trade = Trade(
            symbol=symbol,
            direction="LONG",
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            exit_reason=reason,
        )
        
        self.trades.append(trade)
        self.cash += net_proceeds
        self.total_commission += commission
        self.total_pnl += pnl
        
        del self.positions[symbol]
        
        return trade
    
    def update_positions(self, prices: Dict[str, float], date: datetime) -> None:
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update(prices[symbol], date)
    
    def record_equity(self, date: datetime, prices: Dict[str, float]) -> None:
        """Record portfolio state for equity curve."""
        self.update_positions(prices, date)
        
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "positions_value": self.portfolio_value - self.cash,
            "total_value": self.portfolio_value,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "open_positions": self.open_positions_count,
        })
    
    def get_trade_log_df(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.total_pnl = 0.0
        self.total_commission = 0.0
