from datetime import datetime
from typing import Optional, List
import pandas as pd
import yfinance as yf
from uuid import uuid4

from .base import BaseBroker, OrderResult, OrderStatus, Position


class HistoricalBroker(BaseBroker):
    def __init__(self, initial_capital: float = 1000000.0):
        self._cash = initial_capital
        self._starting_capital = initial_capital
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_ttl: dict[str, datetime] = {}
        self._cache_validity_seconds = 300

    def get_current_price(self, symbol: str) -> float:
        sym = self._format_symbol(symbol)
        try:
            ticker = yf.Ticker(sym)
            data = ticker.history(period="1d")
            if len(data) > 0:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def get_historical_prices(self, symbol: str, days: int = 252) -> pd.DataFrame:
        sym = self._format_symbol(symbol)
        try:
            ticker = yf.Ticker(sym)
            data = ticker.history(period=f"{days}d")
            return data
        except Exception:
            return pd.DataFrame()

    def get_atr(self, symbol: str, period: int = 14) -> float:
        df = self.get_historical_prices(symbol, days=period * 3)
        if len(df) < period + 1:
            return 0.0

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr_df = pd.DataFrame({"tr1": tr1.values, "tr2": tr2.values, "tr3": tr3.values}, index=df.index)
        tr_max = tr_df.max(axis=1)
        atr_series = tr_max.rolling(window=period).mean()
        atr_arr = atr_series.to_numpy()
        
        if len(atr_arr) == 0:
            return 0.0
        atr = float(atr_arr[-1])
        return atr if pd.notna(atr) else 0.0

    def place_market_order(
        self, symbol: str, quantity: int, side: str
    ) -> OrderResult:
        order_id = str(uuid4())
        price = self.get_current_price(symbol)

        if price <= 0:
            result = OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=0,
                price=0.0,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(),
                message="Could not fetch current price",
            )
        elif side.lower() == "buy":
            total_cost = price * quantity
            if total_cost > self._cash:
                result = OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    price=price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    message=f"Insufficient cash. Required: {total_cost}, Available: {self._cash}",
                )
            else:
                self._cash -= total_cost
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    new_qty = pos.quantity + quantity
                    new_entry = (pos.entry_price * pos.quantity + price * quantity) / new_qty
                    pos.quantity = new_qty
                    pos.entry_price = new_entry
                    pos.update_price(price)
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=price,
                        current_price=price,
                        highest_price=price,
                    )
                result = OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=quantity,
                    price=price,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(),
                    message="Order filled successfully",
                )
        else:
            if symbol not in self._positions or self._positions[symbol].quantity < quantity:
                result = OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    price=price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    message="Insufficient position to sell",
                )
            else:
                pos = self._positions[symbol]
                pnl = (price - pos.entry_price) * quantity
                pos.realized_pnl += pnl
                pos.quantity -= quantity
                self._cash += price * quantity

                if pos.quantity == 0:
                    del self._positions[symbol]
                result = OrderResult(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    filled_quantity=quantity,
                    price=price,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(),
                    message=f"Sold {quantity} shares at {price}",
                )

        self._orders[order_id] = result
        return result

    def place_limit_order(
        self, symbol: str, quantity: int, price: float, side: str
    ) -> OrderResult:
        current = self.get_current_price(symbol)
        order_id = str(uuid4())

        filled = False
        if side.lower() == "buy" and current <= price:
            filled = True
        elif side.lower() == "sell" and current >= price:
            filled = True

        if filled:
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
                message="Limit order placed (not filled at current price)",
            )
            self._orders[order_id] = result
            return result

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id in self._orders:
            return self._orders[order_id].status
        return OrderStatus.REJECTED

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def get_account_balance(self) -> float:
        return self._cash

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def _format_symbol(self, symbol: str) -> str:
        sym = symbol.upper().strip()
        if not sym.endswith(".NS"):
            if sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                sym = f"{sym}.NS"
            elif len(sym) <= 10 and sym.isalpha():
                sym = f"{sym}.NS"
        return sym

    def update_prices(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_price(price)

    def set_cash(self, cash: float) -> None:
        self._cash = cash

    @property
    def starting_capital(self) -> float:
        return self._starting_capital
