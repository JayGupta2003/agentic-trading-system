"""Phase 3: Execution Engine Tests

Tests for:
- Position and PortfolioState classes
- BaseBroker, HistoricalBroker, SimulatedBroker
- ExecutionEngine entry/exit logic
- Half-Kelly position sizing
- Risk validation and VaR
- Trailing stop calculations
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.trading.brokers.base import (
    BaseBroker,
    OrderResult,
    OrderStatus,
    Position,
    PortfolioState,
)
from src.trading.brokers.historical import HistoricalBroker
from src.trading.brokers.simulated import SimulatedBroker
from src.trading.execution_engine import (
    BaseExecutionEngine,
    ExecutionConfig,
    ExecutionEngine,
    ExitDecision,
    ExitReason,
    RiskValidation,
)
from src.trading.types import TradingSignal


class TestPosition:
    def test_position_creation(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2500.0,
        )
        assert pos.symbol == "RELIANCE"
        assert pos.quantity == 100
        assert pos.entry_price == 2500.0
        assert pos.highest_price == 0.0

    def test_update_price(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2500.0,
        )
        pos.update_price(2600.0)
        assert pos.current_price == 2600.0
        assert pos.highest_price == 2600.0
        assert pos.unrealized_pnl == 10000.0

    def test_update_price_does_not_lower_highest(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2600.0,
        )
        pos.highest_price = 2700.0
        pos.update_price(2550.0)
        assert pos.highest_price == 2700.0

    def test_update_trailing_stop(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2500.0,
            atr=20.0,
        )
        pos.highest_price = 2600.0
        
        new_stop = pos.update_trailing_stop(atr_multiplier=2.5)
        expected_stop = 2600.0 - (2.5 * 20.0)
        assert new_stop == expected_stop
        assert pos.trailing_stop_price == expected_stop

    def test_trailing_stop_only_increases(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            atr=20.0,
        )
        pos.trailing_stop_price = 2550.0
        
        new_stop = pos.update_trailing_stop(atr_multiplier=2.5)
        assert new_stop <= pos.trailing_stop_price

    def test_market_value(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2600.0,
        )
        assert pos.market_value == 260000.0

    def test_cost_basis(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
        )
        assert pos.cost_basis == 250000.0

    def test_total_pnl(self):
        pos = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2600.0,
            realized_pnl=500.0,
        )
        pos.update_price(2600.0)
        assert pos.total_pnl == 10500.0


class TestPortfolioState:
    def test_portfolio_creation(self):
        portfolio = PortfolioState(
            cash=500000.0,
            starting_capital=1000000.0,
        )
        assert portfolio.cash == 500000.0
        assert portfolio.starting_capital == 1000000.0

    def test_total_market_value(self):
        portfolio = PortfolioState(cash=500000.0)
        portfolio.positions["RELIANCE"] = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2600.0,
        )
        portfolio.positions["TCS"] = Position(
            symbol="TCS",
            quantity=50,
            entry_price=3500.0,
            current_price=3600.0,
        )
        expected = (100 * 2600) + (50 * 3600)
        assert portfolio.total_market_value() == expected

    def test_total_value(self):
        portfolio = PortfolioState(cash=500000.0)
        portfolio.positions["RELIANCE"] = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2600.0,
        )
        expected = 500000.0 + (100 * 2600.0)
        assert portfolio.total_value() == expected

    def test_total_return_pct(self):
        portfolio = PortfolioState(
            cash=600000.0,
            starting_capital=1000000.0,
        )
        portfolio.positions["RELIANCE"] = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2750.0,
        )
        total_value = 600000.0 + (100 * 2750.0)
        expected_return = (total_value - 1000000.0) / 1000000.0 * 100
        assert abs(portfolio.total_return_pct() - expected_return) < 0.01

    def test_drawdown_pct(self):
        portfolio = PortfolioState(
            cash=800000.0,
            starting_capital=1000000.0,
        )
        portfolio.positions["RELIANCE"] = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
            current_price=2250.0,
        )
        total_value = 800000.0 + (100 * 2250.0)
        peak = 1000000.0
        expected_dd = (peak - total_value) / peak * 100
        assert abs(portfolio.drawdown_pct() - expected_dd) < 0.01

    def test_has_position(self):
        portfolio = PortfolioState(cash=500000.0)
        portfolio.positions["RELIANCE"] = Position(
            symbol="RELIANCE",
            quantity=100,
            entry_price=2500.0,
        )
        assert portfolio.has_position("RELIANCE") is True
        assert portfolio.has_position("TCS") is False


class TestSimulatedBroker:
    def test_broker_creation(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        assert broker.get_account_balance() == 1000000.0
        assert len(broker.get_positions()) == 0

    def test_set_price(self):
        broker = SimulatedBroker()
        broker.set_price("RELIANCE", 2500.0)
        assert broker.get_current_price("RELIANCE") == 2500.0

    def test_set_prices(self):
        broker = SimulatedBroker()
        broker.set_prices({"RELIANCE": 2500.0, "TCS": 3500.0})
        assert broker.get_current_price("RELIANCE") == 2500.0
        assert broker.get_current_price("TCS") == 3500.0

    def test_buy_order_success(self):
        broker = SimulatedBroker(initial_capital=1000000.0, slippage_pct=0.0, commission_pct=0.0)
        broker.set_price("RELIANCE", 2500.0)
        
        result = broker.place_market_order("RELIANCE", 100, "buy")
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 100
        assert result.price == 2500.0
        assert broker.get_account_balance() == 1000000.0 - (100 * 2500.0)
        
        position = broker.get_position("RELIANCE")
        assert position is not None
        assert position.quantity == 100
        assert position.entry_price == 2500.0

    def test_buy_order_insufficient_cash(self):
        broker = SimulatedBroker(initial_capital=100000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        result = broker.place_market_order("RELIANCE", 100, "buy")
        
        assert result.status == OrderStatus.REJECTED
        assert "Insufficient cash" in result.message

    def test_buy_order_no_price(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        result = broker.place_market_order("RELIANCE", 100, "buy")
        
        assert result.status == OrderStatus.REJECTED

    def test_sell_order_success(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        result = broker.place_market_order("RELIANCE", 50, "sell")
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 50
        
        position = broker.get_position("RELIANCE")
        assert position.quantity == 50

    def test_sell_order_no_position(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        result = broker.place_market_order("RELIANCE", 100, "sell")
        
        assert result.status == OrderStatus.REJECTED
        assert "Insufficient position" in result.message

    def test_sell_order_partial_position(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 50, "buy")
        result = broker.place_market_order("RELIANCE", 100, "sell")
        
        assert result.status == OrderStatus.REJECTED

    def test_full_exit(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        broker.set_price("RELIANCE", 2600.0)
        result = broker.place_market_order("RELIANCE", 100, "sell")
        
        assert result.status == OrderStatus.FILLED
        assert broker.get_position("RELIANCE") is None

    def test_get_portfolio_state(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        broker.set_price("TCS", 3500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        broker.place_market_order("TCS", 50, "buy")
        
        state = broker.get_portfolio_state()
        
        assert state.cash < 1000000.0
        assert len(state.positions) == 2
        assert "RELIANCE" in state.positions
        assert "TCS" in state.positions


class TestExecutionConfig:
    def test_default_config(self):
        config = ExecutionConfig()
        assert config.max_position_pct == 15.0
        assert config.max_positions == 5
        assert config.max_portfolio_var_pct == 5.0
        assert config.max_drawdown_pct == 10.0
        assert config.min_conviction == 6

    def test_custom_config(self):
        config = ExecutionConfig(
            max_position_pct=20.0,
            max_positions=3,
            win_rate=0.60,
        )
        assert config.max_position_pct == 20.0
        assert config.max_positions == 3
        assert config.win_rate == 0.60


class TestExecutionEngine:
    def test_engine_creation(self):
        engine = ExecutionEngine()
        assert engine.config is not None
        assert engine.config.max_position_pct == 15.0

    def test_get_atr_multiplier_large_cap(self):
        engine = ExecutionEngine()
        assert engine.get_atr_multiplier("large") == 2.5

    def test_get_atr_multiplier_mid_cap(self):
        engine = ExecutionEngine()
        assert engine.get_atr_multiplier("mid") == 3.0

    def test_get_atr_multiplier_small_cap(self):
        engine = ExecutionEngine()
        assert engine.get_atr_multiplier("small") == 3.5

    def test_get_atr_multiplier_unknown(self):
        engine = ExecutionEngine()
        assert engine.get_atr_multiplier("unknown") == 2.5


class TestEntryConditions:
    def test_entry_not_buy_signal(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker()
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="HOLD",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        decision = engine.check_entry_conditions(signal, broker)
        assert decision.should_enter is False
        assert "not BUY" in decision.message

    def test_entry_low_conviction(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker()
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=4,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        decision = engine.check_entry_conditions(signal, broker)
        assert decision.should_enter is False
        assert "Conviction" in decision.message

    def test_entry_position_already_open(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker()
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        decision = engine.check_entry_conditions(signal, broker)
        assert decision.should_enter is False
        assert "already open" in decision.message

    def test_entry_max_positions_reached(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=10000000.0)
        
        for i in range(5):
            broker.set_price(f"STOCK{i}", 100.0)
            broker.place_market_order(f"STOCK{i}", 1000, "buy")
        
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        decision = engine.check_entry_conditions(signal, broker)
        assert decision.should_enter is False
        assert "Max positions" in decision.message

    def test_entry_success(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        decision = engine.check_entry_conditions(signal, broker)
        assert decision.should_enter is True
        assert decision.price == 2500.0


class TestExitConditions:
    def test_exit_no_position(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker()
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        decision = engine.check_exit_conditions(signal, None, broker)
        assert decision.should_exit is False

    def test_exit_signal_override_sell(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="SELL",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        position = broker.get_position("RELIANCE")
        decision = engine.check_exit_conditions(signal, position, broker)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.SIGNAL_OVERRIDE
        assert decision.quantity == 100

    def test_exit_stop_loss_triggered(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        broker.set_price("RELIANCE", 2350.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        position = broker.get_position("RELIANCE")
        decision = engine.check_exit_conditions(signal, position, broker)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.STOP_LOSS

    def test_exit_trailing_stop_triggered(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        
        position = broker.get_position("RELIANCE")
        position.highest_price = 2700.0
        position.atr = 20.0
        position.trailing_stop_price = 2650.0
        
        broker.set_price("RELIANCE", 2640.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
            trailing_stop_pct=2650.0,
        )
        
        decision = engine.check_exit_conditions(signal, position, broker)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.TRAILING_STOP

    def test_exit_target_hit_partial(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        broker.set_price("RELIANCE", 2750.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        position = broker.get_position("RELIANCE")
        decision = engine.check_exit_conditions(signal, position, broker)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.TARGET_HIT
        assert decision.quantity == 50

    def test_exit_no_conditions_met(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        position = broker.get_position("RELIANCE")
        decision = engine.check_exit_conditions(signal, position, broker)
        
        assert decision.should_exit is False
        assert decision.reason == ExitReason.NONE


class TestPositionSizing:
    def test_half_kelly_calculation(self):
        config = ExecutionConfig(
            win_rate=0.55,
            win_loss_ratio=2.0,
            max_position_pct=15.0,
        )
        engine = ExecutionEngine(config)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=10,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        quantity = engine.calculate_position_size(signal, 1000000.0)
        
        base_kelly = 0.55 - (0.45 / 2.0)
        half_kelly = base_kelly / 2
        conviction_adjusted = half_kelly * 1.0
        max_position = 0.15
        
        expected_pct = min(conviction_adjusted, max_position)
        expected_value = 1000000.0 * expected_pct
        expected_qty = int(expected_value / 2500.0)
        
        assert quantity == expected_qty

    def test_position_capped_at_max(self):
        config = ExecutionConfig(
            win_rate=0.55,
            win_loss_ratio=2.0,
            max_position_pct=10.0,
        )
        engine = ExecutionEngine(config)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=10,
            entry_price=100.0,
            stop_loss=95.0,
            target_price=120.0,
        )
        
        quantity = engine.calculate_position_size(signal, 1000000.0)
        
        max_value = 1000000.0 * 0.10
        expected_qty = int(max_value / 100.0)
        
        assert quantity == expected_qty

    def test_position_scaled_by_conviction(self):
        config = ExecutionConfig(
            win_rate=0.55,
            win_loss_ratio=2.0,
            max_position_pct=50.0,
        )
        engine = ExecutionEngine(config)
        
        signal_low = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=6,
            entry_price=100.0,
            stop_loss=95.0,
            target_price=120.0,
        )
        
        signal_high = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=10,
            entry_price=100.0,
            stop_loss=95.0,
            target_price=120.0,
        )
        
        qty_low = engine.calculate_position_size(signal_low, 1000000.0)
        qty_high = engine.calculate_position_size(signal_high, 1000000.0)
        
        assert qty_high > qty_low
        assert qty_high / qty_low == pytest.approx(10.0 / 6.0, rel=0.1)


class TestRiskValidation:
    def test_risk_validation_pass(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        validation = engine.validate_risk_limits(signal, broker)
        
        assert validation.is_valid is True
        assert len(validation.violations) == 0

    def test_risk_validation_max_positions(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=10000000.0)
        
        for i in range(5):
            broker.set_price(f"STOCK{i}", 100.0)
            broker.place_market_order(f"STOCK{i}", 1000, "buy")
        
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        validation = engine.validate_risk_limits(signal, broker)
        
        assert validation.is_valid is False
        assert any("Max positions" in v for v in validation.violations)


class TestVaR:
    def test_var_calculation(self):
        engine = ExecutionEngine()
        var = engine.calculate_var(1000000.0)
        
        expected = 0.02 * 1000000.0
        assert var == expected

    def test_var_proportional_to_portfolio(self):
        engine = ExecutionEngine()
        
        var_1m = engine.calculate_var(1000000.0)
        var_2m = engine.calculate_var(2000000.0)
        
        assert var_2m == 2 * var_1m


class TestExecuteEntry:
    def test_execute_entry_success(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            cap_type="large",
        )
        
        result = engine.execute_entry(signal, broker)
        
        assert result.success is True
        assert result.action == "entry"
        assert signal.position_open is True
        assert signal.entry_executed_at is not None

    def test_execute_entry_rejected(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker()
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="HOLD",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
        )
        
        result = engine.execute_entry(signal, broker)
        
        assert result.success is False


class TestExecuteExit:
    def test_execute_exit_full(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="SELL",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        result = engine.execute_exit(signal, broker)
        
        assert result.success is True
        assert result.quantity == 100
        assert signal.position_open is False

    def test_execute_exit_partial(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        broker.set_price("RELIANCE", 2750.0)
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            position_open=True,
        )
        
        result = engine.execute_exit(signal, broker)
        
        assert result.success is True
        assert result.quantity == 50
        assert signal.position_open is True


class TestTrailingStopUpdate:
    def test_update_trailing_stops(self):
        engine = ExecutionEngine()
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        
        position = broker.get_position("RELIANCE")
        position.atr = 20.0
        
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            cap_type="large",
        )
        
        broker.set_price("RELIANCE", 2600.0)
        engine.update_trailing_stops(signal, broker)
        
        position = broker.get_position("RELIANCE")
        atr_multiplier = 2.5
        atr_from_broker = broker.get_atr("RELIANCE")
        expected_atr = atr_from_broker if atr_from_broker > 0 else 20.0
        expected_stop = 2600.0 - (atr_multiplier * expected_atr)
        assert position.trailing_stop_price == expected_stop


class TestSlippageAndCommission:
    def test_buy_with_slippage(self):
        broker = SimulatedBroker(
            initial_capital=1000000.0,
            slippage_pct=0.001,
        )
        broker.set_price("RELIANCE", 2500.0)
        
        result = broker.place_market_order("RELIANCE", 100, "buy")
        
        expected_price = 2500.0 * 1.001
        assert result.price == pytest.approx(expected_price, abs=0.1)

    def test_sell_with_slippage(self):
        broker = SimulatedBroker(
            initial_capital=1000000.0,
            slippage_pct=0.001,
        )
        broker.set_price("RELIANCE", 2500.0)
        
        broker.place_market_order("RELIANCE", 100, "buy")
        broker.set_price("RELIANCE", 2500.0)
        
        result = broker.place_market_order("RELIANCE", 100, "sell")
        
        expected_price = 2500.0 * 0.999
        assert result.price == pytest.approx(expected_price, abs=0.1)


class TestOrderLimit:
    def test_limit_order_filled_when_condition_met(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2450.0)
        
        result = broker.place_limit_order("RELIANCE", 100, 2500.0, "buy")
        
        assert result.status == OrderStatus.FILLED

    def test_limit_order_pending_when_condition_not_met(self):
        broker = SimulatedBroker(initial_capital=1000000.0)
        broker.set_price("RELIANCE", 2550.0)
        
        result = broker.place_limit_order("RELIANCE", 100, 2500.0, "buy")
        
        assert result.status == OrderStatus.PENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
