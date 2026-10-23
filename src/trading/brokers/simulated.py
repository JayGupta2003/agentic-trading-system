from datetime import datetime
from typing import Optional, List
import pandas as pd
from uuid import uuid4

from .base import BaseBroker, OrderResult, OrderStatus, Position, PortfolioState


class SimulatedBroker(BaseBroker):
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        slippage_pct: float = 0.001,
        commission_pct: float = 0.001,
    ):
        self._cash = initial_capital
        self._starting_capital = initial_capital
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._prices: dict[str, float] = {}
        self._slippage_pct = slippage_pct
        self._commission_pct = commission_pct
        self._portfolio_history: List[PortfolioState] = []

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    def get_current_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def get_historical_prices(self, symbol: str, days: int = 252) -> pd.DataFrame:
        return pd.DataFrame()

    def get_atr(self, symbol: str, period: int = 14) -> float:
        price = self.get_current_price(symbol)
        return price * 0.02

    def place_market_order(
        self, symbol: str, quantity: int, side: str
    ) -> OrderResult:
        order_id = str(uuid4())
        base_price = self.get_current_price(symbol)

        if base_price <= 0:
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=0,
                price=0.0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(),
                message="No price available for symbol",
            )

        slippage = base_price * self._slippage_pct
        execution_price = base_price + slippage if side.lower() == "buy" else base_price - slippage
        commission = execution_price * quantity * self._commission_pct
        total_cost = execution_price * quantity + commission

        if side.lower() == "buy":
            if total_cost > self._cash:
                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    price=execution_price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    message=f"Insufficient cash: {self._cash:.2f}",
                )

            self._cash -= total_cost
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos.quantity + quantity
                total_cost_basis = pos.entry_price * pos.quantity + execution_price * quantity
                pos.entry_price = total_cost_basis / total_qty
                pos.quantity = total_qty
                pos.update_price(execution_price)
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    current_price=execution_price,
                    highest_price=execution_price,
                    atr=self.get_atr(symbol),
                )

            result = OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=quantity,
                price=execution_price,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(),
                message=f"Filled at {execution_price:.2f} with {commission:.2f} commission",
            )
        else:
            if symbol not in self._positions or self._positions[symbol].quantity < quantity:
                return OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    price=execution_price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    message="Insufficient position",
                )

            pos = self._positions[symbol]
            pnl = (execution_price - pos.entry_price) * quantity
            commission_cost = execution_price * quantity * self._commission_pct
            pos.realized_pnl += pnl - commission_cost
            pos.quantity -= quantity
            self._cash += execution_price * quantity - commission_cost

            if pos.quantity == 0:
                del self._positions[symbol]

            result = OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=quantity,
                price=execution_price,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(),
                message=f"Sold {quantity} at {execution_price:.2f}",
            )

        self._orders[order_id] = result
        return result

    def place_limit_order(
        self, symbol: str, quantity: int, price: float, side: str
    ) -> OrderResult:
        current = self.get_current_price(symbol)
        order_id = str(uuid4())

        if side.lower() == "buy" and current <= price:
            return self.place_market_order(symbol, quantity, side)
        elif side.lower() == "sell" and current >= price:
            return self.place_market_order(symbol, quantity, side)
        else:
            result = OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=0,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                message="Limit order placed",
            )
            self._orders[order_id] = result
            return result

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id in self._orders:
            return self._orders[order_id].status
        return OrderStatus.REJECTED

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders and self._orders[order_id].status == OrderStatus.PENDING:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_account_balance(self) -> float:
        return self._cash

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_portfolio_state(self) -> PortfolioState:
        for symbol, pos in self._positions.items():
            price = self.get_current_price(symbol)
            pos.update_price(price)

        return PortfolioState(
            cash=self._cash,
            positions=self._positions.copy(),
            starting_capital=self._starting_capital,
            timestamp=datetime.now(),
        )

    def update_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    def set_cash(self, cash: float) -> None:
        self._cash = cash

    @property
    def starting_capital(self) -> float:
        return self._starting_capital

    @property
    def total_value(self) -> float:
        return self._cash + sum(
            pos.market_value for pos in self._positions.values()
        )
