from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .brokers.base import BaseBroker, PortfolioState

from .types import TradingSignal, CapType


class ExitReason(Enum):
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TARGET_HIT = "target_hit"
    SIGNAL_OVERRIDE = "signal_override"
    TIME_BASED = "time_based"
    NONE = "none"


@dataclass
class ExitDecision:
    should_exit: bool
    reason: ExitReason = ExitReason.NONE
    quantity: int = 0
    message: str = ""


@dataclass
class EntryDecision:
    should_enter: bool
    quantity: int = 0
    price: float = 0.0
    message: str = ""


@dataclass
class RiskValidation:
    is_valid: bool
    var_pct: float = 0.0
    max_position_pct: float = 0.0
    position_count: int = 0
    drawdown_pct: float = 0.0
    violations: List[str] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    max_position_pct: float = 15.0
    max_positions: int = 5
    max_portfolio_var_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    trailing_stop_atr_large: float = 2.5
    trailing_stop_atr_mid: float = 3.0
    trailing_stop_atr_small: float = 3.5
    min_conviction: int = 6
    win_rate: float = 0.55
    win_loss_ratio: float = 2.0
    max_holding_days: int = 30
    target_partial_pct: float = 0.5


@dataclass
class ExecutionResult:
    success: bool
    action: str
    symbol: str
    quantity: int = 0
    price: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class BaseExecutionEngine(ABC):
    @abstractmethod
    def check_entry_conditions(
        self, signal: TradingSignal, broker: "BaseBroker"
    ) -> EntryDecision:
        pass

    @abstractmethod
    def check_exit_conditions(
        self, signal: TradingSignal, position: Any, broker: "BaseBroker"
    ) -> ExitDecision:
        pass

    @abstractmethod
    def calculate_position_size(
        self, signal: TradingSignal, portfolio_value: float
    ) -> int:
        pass

    @abstractmethod
    def validate_risk_limits(
        self, signal: TradingSignal, broker: "BaseBroker"
    ) -> RiskValidation:
        pass

    @abstractmethod
    def calculate_var(
        self, portfolio_value: float, confidence: float = 0.95
    ) -> float:
        pass


class ExecutionEngine(BaseExecutionEngine):
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def check_entry_conditions(
        self, signal: TradingSignal, broker: "BaseBroker"
    ) -> EntryDecision:
        if signal.signal not in ["BUY", "STRONG_BUY"]:
            return EntryDecision(
                should_enter=False,
                message=f"Signal is {signal.signal}, not BUY or STRONG_BUY",
            )

        if signal.conviction < self.config.min_conviction:
            return EntryDecision(
                should_enter=False,
                message=f"Conviction {signal.conviction} below minimum {self.config.min_conviction}",
            )

        if signal.position_open:
            return EntryDecision(
                should_enter=False,
                message="Position already open for this signal",
            )

        current_price = broker.get_current_price(signal.symbol)
        if current_price <= 0:
            return EntryDecision(
                should_enter=False,
                message="Could not fetch current price",
            )

        risk_validation = self.validate_risk_limits(signal, broker)
        if not risk_validation.is_valid:
            return EntryDecision(
                should_enter=False,
                message=f"Risk validation failed: {', '.join(risk_validation.violations)}",
            )

        if risk_validation.position_count >= self.config.max_positions:
            return EntryDecision(
                should_enter=False,
                message=f"Max positions ({self.config.max_positions}) reached",
            )

        position_pct = (current_price * 1) / broker.get_account_balance() * 100
        if position_pct > self.config.max_position_pct:
            return EntryDecision(
                should_enter=False,
                message=f"Position size {position_pct:.1f}% exceeds max {self.config.max_position_pct}%",
            )

        return EntryDecision(should_enter=True, price=current_price)

    def check_exit_conditions(
        self, signal: TradingSignal, position: Any, broker: "BaseBroker"
    ) -> ExitDecision:
        if not signal.position_open or position is None or position.quantity <= 0:
            return ExitDecision(should_exit=False, message="No position to exit")

        current_price = broker.get_current_price(signal.symbol)
        if current_price <= 0:
            return ExitDecision(should_exit=False, message="Could not fetch current price")

        if signal.signal in ["SELL", "STRONG_SELL"]:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.SIGNAL_OVERRIDE,
                quantity=position.quantity,
                message="Signal override: new signal is SELL",
            )

        if current_price <= signal.stop_loss:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                quantity=position.quantity,
                message=f"Stop loss triggered: {current_price} <= {signal.stop_loss}",
            )

        if position.trailing_stop_price > 0 and current_price <= position.trailing_stop_price:
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    quantity=position.quantity,
                    message=f"Trailing stop triggered: {current_price} <= {position.trailing_stop_price:.2f}",
                )

        if current_price >= signal.target_price and position.quantity > 1:
            partial_qty = int(position.quantity * self.config.target_partial_pct)
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TARGET_HIT,
                quantity=partial_qty,
                message=f"Target hit: partial exit of {partial_qty} shares",
            )

        return ExitDecision(should_exit=False, reason=ExitReason.NONE)

    def calculate_position_size(
        self, signal: TradingSignal, portfolio_value: float
    ) -> int:
        base_kelly = self.config.win_rate - ((1 - self.config.win_rate) / self.config.win_loss_ratio)
        half_kelly = base_kelly / 2
        conviction_adjusted = half_kelly * (signal.conviction / 10)
        max_position = self.config.max_position_pct / 100

        final_position_pct = min(conviction_adjusted, max_position)
        position_value = portfolio_value * final_position_pct

        current_price = signal.entry_price if signal.entry_price > 0 else 100
        quantity = int(position_value / current_price)

        return max(1, quantity)

    def validate_risk_limits(
        self, signal: TradingSignal, broker: "BaseBroker"
    ) -> RiskValidation:
        violations: List[str] = []
        positions = broker.get_positions()
        portfolio_value = broker.get_account_balance() + sum(
            p.market_value for p in positions
        )
        position_count = len(positions)
        var_pct = (self.calculate_var(portfolio_value) / portfolio_value * 100) if portfolio_value > 0 else 0

        if position_count >= self.config.max_positions:
            violations.append(f"Max positions ({self.config.max_positions}) reached")

        if var_pct > self.config.max_portfolio_var_pct:
            violations.append(f"VaR {var_pct:.2f}% exceeds max {self.config.max_portfolio_var_pct}%")

        return RiskValidation(
            is_valid=len(violations) == 0,
            var_pct=var_pct,
            max_position_pct=self.config.max_position_pct,
            position_count=position_count,
            violations=violations,
        )

    def calculate_var(self, portfolio_value: float, confidence: float = 0.95) -> float:
        daily_var = 0.02 * portfolio_value
        return daily_var

    def get_atr_multiplier(self, cap_type: str) -> float:
        multipliers = {
            "large": self.config.trailing_stop_atr_large,
            "mid": self.config.trailing_stop_atr_mid,
            "small": self.config.trailing_stop_atr_small,
        }
        return multipliers.get(cap_type, self.config.trailing_stop_atr_large)

    def execute_entry(
        self, signal: TradingSignal, broker: "BaseBroker"
    ) -> ExecutionResult:
        entry_decision = self.check_entry_conditions(signal, broker)

        if not entry_decision.should_enter:
            return ExecutionResult(
                success=False,
                action="entry_check",
                symbol=signal.symbol,
                message=entry_decision.message,
            )

        portfolio_value = broker.get_account_balance() + sum(
            p.market_value for p in broker.get_positions()
        )
        quantity = self.calculate_position_size(signal, portfolio_value)

        result = broker.place_market_order(signal.symbol, quantity, "buy")

        if result.status.value == "filled":
            position = broker.get_position(signal.symbol)
            if position:
                atr_multiplier = self.get_atr_multiplier(signal.cap_type)
                position.atr = broker.get_atr(signal.symbol)
                position.trailing_stop_price = result.price - (position.atr * atr_multiplier)

            signal.position_open = True
            signal.entry_executed_at = result.price
            signal.quantity = quantity
            signal.highest_price_since_entry = result.price

            return ExecutionResult(
                success=True,
                action="entry",
                symbol=signal.symbol,
                quantity=quantity,
                price=result.price,
                message=f"Entered {quantity} shares at {result.price}",
            )
        else:
            return ExecutionResult(
                success=False,
                action="entry",
                symbol=signal.symbol,
                message=result.message,
            )

    def execute_exit(
        self, signal: TradingSignal, broker: "BaseBroker", quantity: Optional[int] = None
    ) -> ExecutionResult:
        position = broker.get_position(signal.symbol)
        if position:
            current_price = broker.get_current_price(signal.symbol)
            if current_price > 0:
                position.update_price(current_price)

        if not position:
            return ExecutionResult(
                success=False,
                action="exit",
                symbol=signal.symbol,
                message="No position found",
            )

        exit_decision = self.check_exit_conditions(signal, position, broker)

        if not exit_decision.should_exit:
            return ExecutionResult(
                success=False,
                action="exit_check",
                symbol=signal.symbol,
                message="Exit conditions not met",
            )

        exit_qty = quantity or exit_decision.quantity

        result = broker.place_market_order(signal.symbol, exit_qty, "sell")

        if result.status.value == "filled":
            remaining = position.quantity

            if exit_decision.reason == ExitReason.TARGET_HIT and exit_qty > 0:
                atr = broker.get_atr(signal.symbol)
                atr_multiplier = self.get_atr_multiplier(signal.cap_type)
                signal.trailing_stop_pct = position.highest_price - (atr * atr_multiplier)
                signal.quantity = remaining

            if remaining <= 0:
                signal.position_open = False
                signal.quantity = 0

            return ExecutionResult(
                success=True,
                action="exit",
                symbol=signal.symbol,
                quantity=exit_qty,
                price=result.price,
                message=f"Exited {exit_qty} shares at {result.price}. Reason: {exit_decision.reason.value}",
            )
        else:
            return ExecutionResult(
                success=False,
                action="exit",
                symbol=signal.symbol,
                message=result.message,
            )

    def update_trailing_stops(self, signal: TradingSignal, broker: "BaseBroker") -> None:
        position = broker.get_position(signal.symbol)
        if not position:
            return

        current_price = broker.get_current_price(signal.symbol)
        if current_price > position.highest_price:
            position.highest_price = current_price

        atr = broker.get_atr(signal.symbol)
        if atr > 0:
            position.atr = atr

        atr_multiplier = self.get_atr_multiplier(signal.cap_type)
        new_stop = position.highest_price - (position.atr * atr_multiplier)

        if new_stop > signal.trailing_stop_pct:
            signal.trailing_stop_pct = new_stop
            position.trailing_stop_price = new_stop
