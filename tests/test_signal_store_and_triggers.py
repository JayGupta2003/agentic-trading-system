"""Tests for Phase 1: Signal Store & Trigger System.

Run with: python -m pytest tests -v
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.trading import (
    AgentName,
    AGENT_ROUTING,
    CapType,
    MarketContext,
    SignalStore,
    SignalType,
    TradingSignal,
    TriggerConfig,
    TriggerEngine,
    TriggerResult,
    TriggerType,
    TRIGGER_PRIORITY,
    check_price_drop_trigger,
    check_vix_trigger,
    get_last_trading_day_of_month,
    is_expiry_week,
    is_scheduled_review_day,
    trigger_decision,
)


class TestTradingSignal:
    """Tests for TradingSignal dataclass."""
    
    def test_create_signal(self):
        """Test creating a basic trading signal."""
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            thesis="Strong fundamentals",
            cap_type="large"
        )
        
        assert signal.symbol == "RELIANCE"
        assert signal.signal == "BUY"
        assert signal.conviction == 8
        assert signal.cap_type == "large"
        assert not signal.position_open
    
    def test_signal_to_dict(self):
        """Test signal serialization to dictionary."""
        signal = TradingSignal(
            symbol="TCS",
            signal="BUY",
            conviction=7,
            entry_price=3500.0,
            stop_loss=3400.0,
            target_price=3800.0,
            generated_at=datetime(2024, 1, 15, 10, 0, 0),
            valid_until=datetime(2024, 2, 15, 10, 0, 0)
        )
        
        d = signal.to_dict()
        
        assert d["symbol"] == "TCS"
        assert d["signal"] == "BUY"
        assert d["generated_at"] == "2024-01-15T10:00:00"
        assert d["valid_until"] == "2024-02-15T10:00:00"
    
    def test_signal_from_dict(self):
        """Test signal deserialization from dictionary."""
        data = {
            "symbol": "HDFCBANK",
            "signal": "HOLD",
            "conviction": 5,
            "entry_price": 1600.0,
            "stop_loss": 1550.0,
            "target_price": 1700.0,
            "trailing_stop_pct": 0.0,
            "thesis": "Wait and watch",
            "key_reasons": [],
            "risks": [],
            "generated_at": "2024-01-20T14:30:00",
            "valid_until": "2024-02-20T14:30:00",
            "last_reviewed": "2024-01-20T14:30:00",
            "review_count": 1,
            "review_type": "scheduled",
            "position_open": False,
            "entry_executed_at": None,
            "quantity": 0,
            "highest_price_since_entry": 0.0,
            "cap_type": "large",
            "atr": 25.0,
            "previous_derivatives_summary": None
        }
        
        signal = TradingSignal.from_dict(data)
        
        assert signal.symbol == "HDFCBANK"
        assert signal.signal == "HOLD"
        assert signal.generated_at == datetime(2024, 1, 20, 14, 30, 0)
    
    def test_is_expired(self):
        """Test signal expiration check."""
        signal = TradingSignal(
            symbol="SBIN",
            signal="BUY",
            conviction=6,
            entry_price=600.0,
            stop_loss=580.0,
            target_price=650.0,
            generated_at=datetime.now() - timedelta(days=35),
            valid_until=datetime.now() - timedelta(days=5)
        )
        
        assert signal.is_expired()
        
        fresh_signal = TradingSignal(
            symbol="SBIN",
            signal="BUY",
            conviction=6,
            entry_price=600.0,
            stop_loss=580.0,
            target_price=650.0,
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(days=25)
        )
        
        assert not fresh_signal.is_expired()
    
    def test_get_atr_multiplier(self):
        """Test ATR multiplier based on cap type."""
        large_cap = TradingSignal(
            symbol="RELIANCE", signal="BUY", conviction=7,
            entry_price=2500, stop_loss=2400, target_price=2750,
            cap_type="large"
        )
        assert large_cap.get_atr_multiplier() == 2.5
        
        mid_cap = TradingSignal(
            symbol="APOLLOTYRE", signal="BUY", conviction=7,
            entry_price=400, stop_loss=380, target_price=450,
            cap_type="mid"
        )
        assert mid_cap.get_atr_multiplier() == 3.0
        
        small_cap = TradingSignal(
            symbol="COMPANY", signal="BUY", conviction=7,
            entry_price=50, stop_loss=45, target_price=60,
            cap_type="small"
        )
        assert small_cap.get_atr_multiplier() == 3.5


class TestSignalStore:
    """Tests for SignalStore class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def store(self, temp_dir):
        """Create SignalStore with temporary directory."""
        return SignalStore(storage_path=temp_dir)
    
    def test_save_and_get_signal(self, store):
        """Test saving and retrieving a signal."""
        signal = TradingSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2750.0,
            cap_type="large"
        )
        
        store.save(signal)
        retrieved = store.get("RELIANCE")
        
        assert retrieved is not None
        assert retrieved.symbol == "RELIANCE"
        assert retrieved.signal == "BUY"
    
    def test_get_nonexistent_signal(self, store):
        """Test getting a signal that doesn't exist."""
        result = store.get("NONEXISTENT")
        assert result is None
    
    def test_get_all_signals(self, store):
        """Test getting all signals."""
        for i, symbol in enumerate(["RELIANCE", "TCS", "HDFCBANK"]):
            signal = TradingSignal(
                symbol=symbol,
                signal="BUY",
                conviction=7,
                entry_price=1000.0 + i * 100,
                stop_loss=900.0,
                target_price=1200.0
            )
            store.save(signal)
        
        all_signals = store.get_all()
        assert len(all_signals) == 3
    
    def test_delete_signal(self, store):
        """Test deleting a signal."""
        signal = TradingSignal(
            symbol="TODELETE",
            signal="SELL",
            conviction=3,
            entry_price=500.0,
            stop_loss=450.0,
            target_price=550.0
        )
        store.save(signal)
        assert store.get("TODELETE") is not None
        
        store.delete("TODELETE")
        assert store.get("TODELETE") is None
    
    def test_get_active_signals(self, store):
        """Test getting only active (non-expired) signals."""
        expired = TradingSignal(
            symbol="EXPIRED",
            signal="BUY",
            conviction=7,
            entry_price=1000.0,
            stop_loss=900.0,
            target_price=1200.0,
            generated_at=datetime.now() - timedelta(days=60),
            valid_until=datetime.now() - timedelta(days=30)
        )
        
        active = TradingSignal(
            symbol="ACTIVE",
            signal="BUY",
            conviction=7,
            entry_price=1000.0,
            stop_loss=900.0,
            target_price=1200.0,
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(days=30)
        )
        
        store.save(expired)
        store.save(active)
        
        active_signals = store.get_active()
        assert len(active_signals) == 1
        assert active_signals[0].symbol == "ACTIVE"
    
    def test_update_execution_state(self, store):
        """Test updating execution state."""
        signal = TradingSignal(
            symbol="UPDATE",
            signal="BUY",
            conviction=8,
            entry_price=1000.0,
            stop_loss=900.0,
            target_price=1200.0
        )
        store.save(signal)
        
        store.update_execution_state(
            "UPDATE",
            position_open=True,
            entry_executed_at=1005.0,
            quantity=100,
            highest_price=1020.0
        )
        
        updated = store.get("UPDATE")
        assert updated.position_open is True
        assert updated.entry_executed_at == 1005.0
        assert updated.quantity == 100
        assert updated.highest_price_since_entry == 1020.0
    
    def test_get_signal_count(self, store):
        """Test getting signal count summary."""
        for i, symbol in enumerate(["A", "B", "C"]):
            signal = TradingSignal(
                symbol=symbol,
                signal="BUY",
                conviction=7,
                entry_price=1000.0,
                stop_loss=900.0,
                target_price=1200.0,
                position_open=(i == 0)
            )
            store.save(signal)
        
        counts = store.get_signal_count()
        assert counts["total"] == 3
        assert counts["open_positions"] == 1


class TestTriggerTypes:
    """Tests for trigger types and agent routing."""
    
    def test_agent_routing_scheduled(self):
        """Test that scheduled review runs all 6 agents."""
        agents = AGENT_ROUTING[TriggerType.SCHEDULED]
        expected = [
            AgentName.FUNDAMENTAL, AgentName.TECHNICAL, AgentName.SENTIMENT,
            AgentName.DERIVATIVES, AgentName.RISK, AgentName.CIO
        ]
        assert agents == expected
    
    def test_agent_routing_price_drop(self):
        """Test that price drop skips fundamental and derivatives."""
        agents = AGENT_ROUTING[TriggerType.PRICE_DROP]
        assert AgentName.FUNDAMENTAL not in agents
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.CIO in agents
    
    def test_agent_routing_max_drawdown(self):
        """Test that max drawdown only runs risk and CIO."""
        agents = AGENT_ROUTING[TriggerType.MAX_DRAWDOWN]
        assert agents == [AgentName.RISK, AgentName.CIO]
    
    def test_agent_routing_expiry_week(self):
        """Test that expiry week skips derivatives."""
        agents = AGENT_ROUTING[TriggerType.EXPIRY_WEEK]
        assert AgentName.DERIVATIVES not in agents
        assert AgentName.CIO in agents
    
    def test_trigger_priority(self):
        """Test trigger priority ordering."""
        assert TRIGGER_PRIORITY[TriggerType.MAX_DRAWDOWN] > TRIGGER_PRIORITY[TriggerType.PRICE_DROP]
        assert TRIGGER_PRIORITY[TriggerType.EARNINGS] > TRIGGER_PRIORITY[TriggerType.SCHEDULED]


class TestExpiryWeek:
    """Tests for expiry week detection."""
    
    def test_expiry_week_last_thursday(self):
        """Test detection on expiry Thursday."""
        jan_2026 = datetime(2026, 1, 29)
        assert is_expiry_week(jan_2026)
    
    def test_expiry_week_friday(self):
        """Test detection on day after expiry."""
        jan_2026 = datetime(2026, 1, 30)
        assert is_expiry_week(jan_2026)
    
    def test_not_expiry_week_middle(self):
        """Test non-expiry week."""
        jan_2026 = datetime(2026, 1, 15)
        assert not is_expiry_week(jan_2026)
    
    def test_expiry_week_monday(self):
        """Test detection on Monday of expiry week."""
        jan_2026 = datetime(2026, 1, 26)
        assert is_expiry_week(jan_2026)


class TestScheduledReview:
    """Tests for scheduled review day detection."""
    
    def test_last_trading_day_of_month(self):
        """Test getting last trading day."""
        jan_2026 = datetime(2026, 1, 31)
        last_day = get_last_trading_day_of_month(jan_2026)
        
        assert last_day.month == 1
        assert last_day.year == 2026
        assert last_day.weekday() <= 4
    
    def test_scheduled_review_day(self):
        """Test scheduled review on last trading day."""
        last_day = get_last_trading_day_of_month(datetime(2026, 1, 15))
        assert is_scheduled_review_day(last_day)
    
    def test_not_scheduled_review_day(self):
        """Test non-scheduled day."""
        mid_month = datetime(2026, 1, 15)
        assert not is_scheduled_review_day(mid_month)


class TestVIXTrigger:
    """Tests for VIX spike detection."""
    
    def test_vix_spike_detected(self):
        """Test VIX spike detection with Z-score."""
        vix_history = [15.0] * 15 + [16.0, 17.0, 18.0, 19.0, 20.0]
        current_vix = 25.0
        
        assert check_vix_trigger(vix_history, current_vix)
    
    def test_vix_no_spike(self):
        """Test no spike when VIX is within normal range."""
        vix_history = [14.0, 15.0, 14.5, 15.0, 14.8, 15.2, 15.0, 14.5, 15.0, 15.5,
                       14.8, 15.0, 15.2, 14.9, 15.1, 15.0, 14.7, 15.3, 14.9, 15.0]
        current_vix = 15.5
        
        assert not check_vix_trigger(vix_history, current_vix)
    
    def test_vix_insufficient_history(self):
        """Test with insufficient history."""
        vix_history = [15.0, 16.0, 17.0]
        current_vix = 30.0
        
        assert not check_vix_trigger(vix_history, current_vix)


class TestPriceDropTrigger:
    """Tests for price drop trigger."""
    
    def test_large_cap_drop_triggered(self):
        """Test large cap 5% drop triggers."""
        config = TriggerConfig()
        assert check_price_drop_trigger(
            current_price=95.0,
            previous_price=100.0,
            cap_type="large",
            config=config
        )
    
    def test_large_cap_no_trigger(self):
        """Test large cap 3% drop doesn't trigger."""
        config = TriggerConfig()
        assert not check_price_drop_trigger(
            current_price=97.0,
            previous_price=100.0,
            cap_type="large",
            config=config
        )
    
    def test_small_cap_higher_threshold(self):
        """Test small cap has higher threshold."""
        config = TriggerConfig()
        assert not check_price_drop_trigger(
            current_price=90.0,
            previous_price=100.0,
            cap_type="small",
            config=config
        )
        assert check_price_drop_trigger(
            current_price=87.0,
            previous_price=100.0,
            cap_type="small",
            config=config
        )


class TestTriggerEngine:
    """Tests for TriggerEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create TriggerEngine instance."""
        return TriggerEngine()
    
    def test_no_triggers(self, engine):
        """Test with no trigger conditions."""
        context = MarketContext(
            current_date=datetime(2026, 1, 15),
            vix=15.0,
            vix_history=[15.0] * 20,
            stock_price=100.0,
            stock_previous_price=100.0,
            stock_cap_type="large"
        )
        
        result = engine.evaluate(context)
        
        assert not result.should_run_agents
        assert result.trigger_type is None
    
    def test_scheduled_review_trigger(self, engine):
        """Test scheduled review triggers (may also be expiry week)."""
        last_day = get_last_trading_day_of_month(datetime(2026, 5, 15))
        context = MarketContext(
            current_date=last_day,
            vix=15.0,
            vix_history=[15.0] * 20
        )
        
        result = engine.evaluate(context)
        
        assert result.should_run_agents
        assert result.trigger_type in [TriggerType.SCHEDULED, TriggerType.EXPIRY_WEEK]
        assert len(result.agents_to_run) >= 5
    
    def test_price_drop_trigger(self, engine):
        """Test price drop triggers."""
        context = MarketContext(
            current_date=datetime(2026, 1, 15),
            vix=15.0,
            vix_history=[15.0] * 20,
            stock_price=93.0,
            stock_previous_price=100.0,
            stock_cap_type="large"
        )
        
        result = engine.evaluate(context)
        
        assert result.should_run_agents
        assert result.trigger_type == TriggerType.PRICE_DROP
        assert AgentName.TECHNICAL in result.agents_to_run
        assert AgentName.FUNDAMENTAL not in result.agents_to_run
    
    def test_vix_spike_trigger(self, engine):
        """Test VIX spike triggers."""
        context = MarketContext(
            current_date=datetime(2026, 1, 15),
            vix=28.0,
            vix_history=[15.0] * 15 + [16.0, 17.0, 18.0, 19.0, 20.0]
        )
        
        result = engine.evaluate(context)
        
        assert result.should_run_agents
        assert result.trigger_type == TriggerType.VIX_SPIKE
    
    def test_max_drawdown_trigger(self, engine):
        """Test max drawdown triggers with correct agents."""
        context = MarketContext(
            current_date=datetime(2026, 1, 15),
            vix=15.0,
            vix_history=[15.0] * 20,
            portfolio_drawdown_pct=12.0
        )
        
        result = engine.evaluate(context)
        
        assert result.should_run_agents
        assert result.trigger_type == TriggerType.MAX_DRAWDOWN
        assert result.agents_to_run == [AgentName.RISK, AgentName.CIO]
    
    def test_expiry_week_trigger(self, engine):
        """Test expiry week triggers."""
        jan_29 = datetime(2026, 1, 29)
        context = MarketContext(
            current_date=jan_29,
            vix=15.0,
            vix_history=[15.0] * 20
        )
        
        result = engine.evaluate(context)
        
        assert result.should_run_agents
        assert TriggerType.EXPIRY_WEEK in [result.trigger_type]
    
    def test_priority_highest_wins(self, engine):
        """Test that highest priority trigger wins."""
        jan_29 = datetime(2026, 1, 29)
        context = MarketContext(
            current_date=jan_29,
            vix=15.0,
            vix_history=[15.0] * 20,
            stock_price=93.0,
            stock_previous_price=100.0,
            stock_cap_type="large",
            portfolio_drawdown_pct=12.0
        )
        
        result = engine.evaluate(context)
        
        assert result.trigger_type == TriggerType.MAX_DRAWDOWN


class TestTriggerDecision:
    """Tests for trigger_decision convenience function."""
    
    def test_convenience_function(self):
        """Test convenience function returns TriggerResult."""
        context = MarketContext(
            current_date=datetime(2026, 1, 15),
            vix=15.0,
            vix_history=[15.0] * 20
        )
        
        result = trigger_decision(context)
        
        assert isinstance(result, TriggerResult)
        assert hasattr(result, 'should_run_agents')
        assert hasattr(result, 'trigger_type')
        assert hasattr(result, 'agents_to_run')


class TestTriggerConfig:
    """Tests for TriggerConfig."""
    
    def test_drop_thresholds(self):
        """Test cap-type specific drop thresholds."""
        config = TriggerConfig()
        
        assert config.get_drop_threshold("large") == 5.0
        assert config.get_drop_threshold("mid") == 8.0
        assert config.get_drop_threshold("small") == 12.0
        assert config.get_drop_threshold("unknown") == 5.0
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = TriggerConfig(
            large_cap_drop_threshold=6.0,
            mid_cap_drop_threshold=10.0,
            small_cap_drop_threshold=15.0
        )
        
        assert config.get_drop_threshold("large") == 6.0
        assert config.get_drop_threshold("mid") == 10.0
        assert config.get_drop_threshold("small") == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
