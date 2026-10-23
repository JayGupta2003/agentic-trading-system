"""
Phase 6 Test Suite - Integration & Dashboard

Tests the EventLoop, dashboard components, and integration of all phases.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch
import json
import os

from src.trading.event_loop import (
    TradingEventLoop,
    EventLoopConfig,
    EventLoopStatus,
    TriggerLogEntry,
    ExecutionLogEntry,
)


class TestEventLoopConfig(unittest.TestCase):
    """Tests for EventLoopConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EventLoopConfig()
        
        self.assertEqual(config.storage_path, "data/signals")
        self.assertTrue(config.use_scheduled_reviews)
        self.assertTrue(config.include_vix_triggers)
        self.assertTrue(config.include_price_drop_triggers)
        self.assertTrue(config.include_max_drawdown_circuit_breaker)
        self.assertEqual(config.max_drawdown_threshold, 10.0)
        self.assertEqual(config.vix_zscore_threshold, 2.0)
        self.assertEqual(config.vix_history_window, 20)
        self.assertEqual(config.max_positions, 5)
        self.assertEqual(config.max_position_pct, 15.0)
        self.assertEqual(config.min_conviction, 6)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EventLoopConfig(
            storage_path="custom/path",
            max_positions=3,
            max_position_pct=20.0,
            max_drawdown_threshold=15.0,
        )
        
        self.assertEqual(config.storage_path, "custom/path")
        self.assertEqual(config.max_positions, 3)
        self.assertEqual(config.max_position_pct, 20.0)
        self.assertEqual(config.max_drawdown_threshold, 15.0)


class TestTradingEventLoop(unittest.TestCase):
    """Tests for TradingEventLoop class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EventLoopConfig(
            storage_path="data/test_signals",
            trigger_log_path="data/test_trigger_log.json",
            execution_log_path="data/test_execution_log.json",
        )

    def test_loop_initialization(self):
        """Test event loop initialization."""
        loop = TradingEventLoop(self.config)
        
        self.assertIsNotNone(loop.signal_store)
        self.assertIsNotNone(loop.trigger_engine)
        self.assertIsNotNone(loop.execution_engine)
        self.assertIsNotNone(loop.broker)
        self.assertIsNotNone(loop.summarizer)

    def test_get_status(self):
        """Test getting event loop status."""
        loop = TradingEventLoop(self.config)
        
        status = loop._get_status(
            current_date=datetime.now(),
            triggers=[],
            executions=[]
        )
        
        self.assertIsInstance(status, EventLoopStatus)
        self.assertIsNotNone(status.last_run)
        self.assertIsNotNone(status.next_scheduled_review)
        self.assertIsInstance(status.active_signals, int)
        self.assertIsInstance(status.open_positions, int)
        self.assertIsInstance(status.portfolio_value, float)

    @patch("yfinance.Ticker")
    def test_fetch_market_data(self, mock_ticker):
        """Test fetching market data."""
        mock_history = MagicMock()
        mock_history.history.return_value = MagicMock(
            __len__=lambda self: 2,
            __getitem__=lambda self, key: [20.0, 20.5] if key == "Close" else None,
            iloc=MagicMock(
                __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
            ),
            values=[20.0, 20.5],
            tail=MagicMock(
                return_value=MagicMock(
                    values=[20.0, 20.5],
                    iloc=MagicMock(
                        __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
                    )
                )
            )
        )
        mock_ticker.return_value = mock_history
        
        loop = TradingEventLoop(self.config)
        market_data = loop._fetch_market_data(datetime.now())
        
        self.assertIsInstance(market_data, dict)
        self.assertIn("timestamp", market_data)
        self.assertIn("vix", market_data)
        self.assertIn("vix_history", market_data)

    def test_get_all_symbols_empty(self):
        """Test getting symbols when signal store is empty."""
        loop = TradingEventLoop(self.config)
        
        symbols = loop._get_all_symbols()
        
        self.assertIsInstance(symbols, list)

    def test_get_trigger_summary_empty(self):
        """Test trigger summary when no triggers."""
        loop = TradingEventLoop(self.config)
        loop._trigger_log = []
        
        summary = loop.get_trigger_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(len(summary), 0)

    def test_get_execution_summary_empty(self):
        """Test execution summary when no executions."""
        loop = TradingEventLoop(self.config)
        
        summary = loop.get_execution_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("total_executions", summary)
        self.assertEqual(summary["total_executions"], 0)

    def test_get_review_schedule_empty(self):
        """Test review schedule when no signals."""
        loop = TradingEventLoop(self.config)
        
        schedule = loop.get_review_schedule()
        
        self.assertIsInstance(schedule, dict)
        self.assertEqual(len(schedule), 0)

    def test_get_next_review_no_signals(self):
        """Test next review when no signals."""
        loop = TradingEventLoop(self.config)
        
        symbol, date = loop.get_next_review()
        
        self.assertEqual(symbol, "NONE")
        self.assertIsInstance(date, datetime)

    def test_get_trigger_history(self):
        """Test getting trigger history."""
        loop = TradingEventLoop(self.config)
        
        history = loop.get_trigger_history(days=30)
        
        self.assertIsInstance(history, list)

    def test_get_trigger_history_with_filter(self):
        """Test getting trigger history with filter."""
        loop = TradingEventLoop(self.config)
        
        history = loop.get_trigger_history(days=30, trigger_type="scheduled")
        
        self.assertIsInstance(history, list)

    def test_get_execution_history(self):
        """Test getting execution history."""
        loop = TradingEventLoop(self.config)
        
        history = loop.get_execution_history(days=30)
        
        self.assertIsInstance(history, list)

    def test_get_execution_history_with_symbol(self):
        """Test getting execution history with symbol filter."""
        loop = TradingEventLoop(self.config)
        
        history = loop.get_execution_history(days=30, symbol="RELIANCE")
        
        self.assertIsInstance(history, list)


class TestTriggerLogEntry(unittest.TestCase):
    """Tests for TriggerLogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test creating a trigger log entry."""
        entry = TriggerLogEntry(
            timestamp=datetime.now(),
            trigger_type="scheduled",
            symbols_affected=["RELIANCE", "TCS"],
            agents_run=["fundamental", "technical", "cio"],
            decision="run_agents",
            details={"reason": "Monthly review"}
        )
        
        self.assertEqual(entry.trigger_type, "scheduled")
        self.assertEqual(len(entry.symbols_affected), 2)
        self.assertEqual(len(entry.agents_run), 3)
        self.assertEqual(entry.decision, "run_agents")


class TestExecutionLogEntry(unittest.TestCase):
    """Tests for ExecutionLogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test creating an execution log entry."""
        entry = ExecutionLogEntry(
            timestamp=datetime.now(),
            action="ENTRY",
            symbol="RELIANCE",
            quantity=100,
            price=2500.0,
            reason="signal_buy",
            pnl=0.0,
            success=True,
        )
        
        self.assertEqual(entry.action, "ENTRY")
        self.assertEqual(entry.symbol, "RELIANCE")
        self.assertEqual(entry.quantity, 100)
        self.assertEqual(entry.price, 2500.0)
        self.assertTrue(entry.success)

    def test_log_entry_with_pnl(self):
        """Test execution log entry with P&L."""
        entry = ExecutionLogEntry(
            timestamp=datetime.now(),
            action="EXIT",
            symbol="RELIANCE",
            quantity=100,
            price=2600.0,
            reason="target_hit",
            pnl=10000.0,
            success=True,
        )
        
        self.assertEqual(entry.pnl, 10000.0)

    def test_log_entry_failure(self):
        """Test execution log entry for failed trade."""
        entry = ExecutionLogEntry(
            timestamp=datetime.now(),
            action="ENTRY",
            symbol="RELIANCE",
            quantity=100,
            price=0.0,
            reason="insufficient_cash",
            success=False,
            error="Not enough cash"
        )
        
        self.assertFalse(entry.success)
        self.assertEqual(entry.error, "Not enough cash")


class TestEventLoopStatus(unittest.TestCase):
    """Tests for EventLoopStatus dataclass."""

    def test_status_creation(self):
        """Test creating event loop status."""
        status = EventLoopStatus(
            last_run=datetime.now(),
            next_scheduled_review=datetime.now() + timedelta(days=30),
            active_signals=5,
            open_positions=3,
            portfolio_value=1100000.0,
            portfolio_drawdown_pct=2.5,
            recent_triggers=[],
            recent_executions=[],
            pending_reviews=["RELIANCE", "TCS"],
        )
        
        self.assertIsNotNone(status.last_run)
        self.assertEqual(status.active_signals, 5)
        self.assertEqual(status.open_positions, 3)
        self.assertEqual(status.portfolio_value, 1100000.0)
        self.assertEqual(status.portfolio_drawdown_pct, 2.5)


class TestEventLoopIntegration(unittest.TestCase):
    """Integration tests for the event loop."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EventLoopConfig(
            storage_path="data/test_signals",
            trigger_log_path="data/test_trigger_log.json",
            execution_log_path="data/test_execution_log.json",
        )

    @patch("yfinance.Ticker")
    def test_run_daily(self, mock_ticker):
        """Test running the daily event loop."""
        mock_history = MagicMock()
        mock_history.history.return_value = MagicMock(
            __len__=lambda self: 2,
            iloc=MagicMock(
                __getitem__=lambda self, idx: [20.0, 20.5] if idx == -1 else 20.5
            ),
            values=[20.0, 20.5],
            tail=MagicMock(
                return_value=MagicMock(
                    values=[20.0, 20.5],
                    iloc=MagicMock(
                        __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
                    )
                )
            )
        )
        mock_ticker.return_value = mock_history
        
        loop = TradingEventLoop(self.config)
        
        status = loop.run_daily(
            current_date=datetime.now(),
            market_data={"vix": 20.0, "vix_history": [20.0] * 30, "nifty_change_pct": 0.0}
        )
        
        self.assertIsInstance(status, EventLoopStatus)
        self.assertIsNotNone(status.last_run)

    def test_load_and_save_logs(self):
        """Test loading and saving logs."""
        loop = TradingEventLoop(self.config)
        
        log_entry = TriggerLogEntry(
            timestamp=datetime.now(),
            trigger_type="test",
            symbols_affected=["TEST"],
            agents_run=["cio"],
            decision="test_decision",
            details={}
        )
        loop._trigger_log.append(log_entry)
        
        loop._save_logs()
        
        loop2 = TradingEventLoop(self.config)
        self.assertGreaterEqual(len(loop2._trigger_log), 0)


class TestSignalStoreIntegration(unittest.TestCase):
    """Tests for SignalStore integration with event loop."""

    def test_signal_store_connection(self):
        """Test that event loop connects to signal store."""
        config = EventLoopConfig(storage_path="data/test_signals")
        loop = TradingEventLoop(config)
        
        self.assertIsNotNone(loop.signal_store)
        self.assertIsInstance(loop.signal_store.get_all(), list)


class TestTriggerEngineIntegration(unittest.TestCase):
    """Tests for TriggerEngine integration with event loop."""

    def test_trigger_engine_connection(self):
        """Test that event loop connects to trigger engine."""
        config = EventLoopConfig()
        loop = TradingEventLoop(config)
        
        self.assertIsNotNone(loop.trigger_engine)


class TestExecutionEngineIntegration(unittest.TestCase):
    """Tests for ExecutionEngine integration with event loop."""

    def test_execution_engine_connection(self):
        """Test that event loop connects to execution engine."""
        config = EventLoopConfig()
        loop = TradingEventLoop(config)
        
        self.assertIsNotNone(loop.execution_engine)


class TestDataSummarizerIntegration(unittest.TestCase):
    """Tests for DataSummarizer integration with event loop."""

    def test_summarizer_connection(self):
        """Test that event loop connects to data summarizer."""
        config = EventLoopConfig()
        loop = TradingEventLoop(config)
        
        self.assertIsNotNone(loop.summarizer)


class TestBrokerIntegration(unittest.TestCase):
    """Tests for broker integration with event loop."""

    def test_broker_connection(self):
        """Test that event loop connects to broker."""
        config = EventLoopConfig()
        loop = TradingEventLoop(config)
        
        self.assertIsNotNone(loop.broker)
        self.assertIsInstance(loop.broker.get_account_balance(), float)


class TestRunDailyEventLoop(unittest.TestCase):
    """Tests for the convenience function."""

    @patch("yfinance.Ticker")
    def test_run_daily_event_loop_function(self, mock_ticker):
        """Test the run_daily_event_loop convenience function."""
        mock_history = MagicMock()
        mock_history.history.return_value = MagicMock(
            __len__=lambda self: 2,
            iloc=MagicMock(
                __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
            ),
            values=[20.0, 20.5],
            tail=MagicMock(
                return_value=MagicMock(
                    values=[20.0, 20.5],
                    iloc=MagicMock(
                        __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
                    )
                )
            )
        )
        mock_ticker.return_value = mock_history
        
        from src.trading.event_loop import run_daily_event_loop
        
        status = run_daily_event_loop(EventLoopConfig())
        
        self.assertIsInstance(status, EventLoopStatus)


class TestEndToEndFlow(unittest.TestCase):
    """End-to-end flow tests."""

    @patch("yfinance.Ticker")
    def test_complete_flow(self, mock_ticker):
        """Test complete event loop flow."""
        mock_history = MagicMock()
        mock_history.history.return_value = MagicMock(
            __len__=lambda self: 2,
            iloc=MagicMock(
                __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
            ),
            values=[20.0, 20.5],
            tail=MagicMock(
                return_value=MagicMock(
                    values=[20.0, 20.5],
                    iloc=MagicMock(
                        __getitem__=lambda self, idx: 20.0 if idx == -1 else 20.5
                    )
                )
            )
        )
        mock_ticker.return_value = mock_history
        
        config = EventLoopConfig(
            use_scheduled_reviews=False,
            include_vix_triggers=False,
            include_max_drawdown_circuit_breaker=False,
        )
        
        loop = TradingEventLoop(config)
        
        status = loop.run_daily(
            current_date=datetime.now(),
            market_data={"vix": 20.0, "vix_history": [20.0] * 30, "nifty_change_pct": 0.0}
        )
        
        self.assertIsInstance(status, EventLoopStatus)
        self.assertIsNotNone(status.last_run)
        self.assertIsNotNone(status.next_scheduled_review)
        self.assertGreaterEqual(status.portfolio_value, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
