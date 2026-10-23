from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    price: float
    status: OrderStatus
    timestamp: datetime
    message: str = ""
    fills: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float = 0.0
    highest_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    atr: float = 0.0
    trailing_stop_price: float = 0.0

    def update_price(self, price: float) -> None:
        self.current_price = price
        if price > self.highest_price:
            self.highest_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity

    def update_trailing_stop(self, atr_multiplier: float = 2.5) -> float:
        if self.highest_price > 0 and self.atr > 0:
            new_stop = self.highest_price - (atr_multiplier * self.atr)
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
                return new_stop
        return self.trailing_stop_price

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    starting_capital: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    def total_value(self) -> float:
        return self.cash + self.total_market_value()

    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def total_pnl(self) -> float:
        return self.total_realized_pnl() + self.total_unrealized_pnl()

    def total_return_pct(self) -> float:
        if self.starting_capital > 0:
            return (self.total_value() - self.starting_capital) / self.starting_capital * 100
        return 0.0

    def drawdown_pct(self) -> float:
        if self.starting_capital > 0:
            peak = self.starting_capital
            current = self.total_value()
            return (peak - current) / peak * 100
        return 0.0

    def position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions


class BaseBroker(ABC):
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def get_historical_prices(self, symbol: str, days: int = 252) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_atr(self, symbol: str, period: int = 14) -> float:
        pass

    @abstractmethod
    def place_market_order(
        self, symbol: str, quantity: int, side: str
    ) -> OrderResult:
        pass

    @abstractmethod
    def place_limit_order(
        self, symbol: str, quantity: int, price: float, side: str
    ) -> OrderResult:
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        pass

    def calculate_var(
        self, portfolio_value: float, confidence: float = 0.95, days: int = 1
    ) -> float:
        daily_var = 0.02 * portfolio_value
        return daily_var * (days ** 0.5)
