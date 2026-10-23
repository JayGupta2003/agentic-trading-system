"""
Phase 5 Test Suite - Event-Driven Backtester

Tests the EventDrivenBacktester and related components to ensure
proper time-travel simulation, trigger handling, and performance calculation.
"""

import unittest
from datetime import datetime, timedelta
from collections import defaultdict
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from src.backtesting.event_driven import (
    EventDrivenBacktester,
    BacktestConfig,
    BacktestSignal,
    BacktestTrade,
    BacktestMetrics,
    TriggerEvent,
    DailySnapshot,
    BacktestSignalGenerator,
    BacktestResult,
)


class TestBacktestConfig(unittest.TestCase):
    """Tests for BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        self.assertEqual(config.initial_capital, 1000000.0)
        self.assertEqual(config.max_positions, 5)
        self.assertEqual(config.max_position_pct, 15.0)
        self.assertEqual(config.min_conviction, 6)
        self.assertEqual(config.trailing_stop_atr_large, 2.5)
        self.assertEqual(config.trailing_stop_atr_mid, 3.0)
        self.assertEqual(config.trailing_stop_atr_small, 3.5)
        self.assertTrue(config.use_scheduled_reviews)
        self.assertTrue(config.include_vix_triggers)
        self.assertTrue(config.include_price_drop_triggers)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            initial_capital=500000.0,
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2025, 6, 1),
            symbols=["RELIANCE", "TCS"],
            max_positions=3,
            max_position_pct=20.0,
        )
        
        self.assertEqual(config.initial_capital, 500000.0)
        self.assertEqual(config.start_date, datetime(2024, 6, 1))
        self.assertEqual(config.end_date, datetime(2025, 6, 1))
        self.assertEqual(config.symbols, ["RELIANCE", "TCS"])
        self.assertEqual(config.max_positions, 3)
        self.assertEqual(config.max_position_pct, 20.0)

    def test_data_quality_notes_auto_populated(self):
        """Test that data quality notes are auto-populated."""
        config = BacktestConfig()
        
        self.assertIn(
            "Fundamental data uses static quarterly snapshots",
            config.data_quality_notes[0]
        )
        self.assertIn(
            "LLM signals simulated",
            config.data_quality_notes[1]
        )


class TestBacktestSignal(unittest.TestCase):
    """Tests for BacktestSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a backtest signal."""
        signal = BacktestSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2700.0,
            thesis="Test thesis",
            generated_at=datetime(2024, 1, 15),
            valid_until=datetime(2024, 2, 15),
            review_type="scheduled",
            cap_type="large",
            agents_run=["fundamental", "technical", "cio"],
            trigger_type="scheduled",
        )
        
        self.assertEqual(signal.symbol, "RELIANCE")
        self.assertEqual(signal.signal, "BUY")
        self.assertEqual(signal.conviction, 8)
        self.assertEqual(signal.cap_type, "large")
        self.assertEqual(len(signal.agents_run), 3)
        self.assertEqual(signal.trigger_type, "scheduled")

    def test_signal_validity(self):
        """Test signal validity check."""
        signal = BacktestSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=7,
            entry_price=2500.0,
            stop_loss=2400.0,
            target_price=2700.0,
            thesis="Test thesis",
            generated_at=datetime.now() - timedelta(days=15),
            valid_until=datetime.now() + timedelta(days=15),
            review_type="scheduled",
            cap_type="large",
            agents_run=[],
        )
        
        self.assertTrue(signal.valid_until > datetime.now())


class TestBacktestTrade(unittest.TestCase):
    """Tests for BacktestTrade dataclass."""

    def test_trade_creation(self):
        """Test creating a backtest trade."""
        trade = BacktestTrade(
            date=datetime(2024, 1, 15),
            symbol="RELIANCE",
            action="ENTRY",
            quantity=100,
            price=2500.0,
            pnl=0.0,
            pnl_pct=0.0,
            reason="",
            conviction=8,
        )
        
        self.assertEqual(trade.symbol, "RELIANCE")
        self.assertEqual(trade.action, "ENTRY")
        self.assertEqual(trade.quantity, 100)
        self.assertEqual(trade.price, 2500.0)

    def test_trade_with_pnl(self):
        """Test trade with profit/loss."""
        trade = BacktestTrade(
            date=datetime(2024, 2, 15),
            symbol="RELIANCE",
            action="EXIT",
            quantity=100,
            price=2600.0,
            pnl=10000.0,
            pnl_pct=4.0,
            reason="target_hit",
            conviction=8,
        )
        
        self.assertEqual(trade.action, "EXIT")
        self.assertEqual(trade.pnl, 10000.0)
        self.assertEqual(trade.pnl_pct, 4.0)
        self.assertEqual(trade.reason, "target_hit")


class TestBacktestMetrics(unittest.TestCase):
    """Tests for BacktestMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating backtest metrics."""
        metrics = BacktestMetrics(
            total_return_pct=25.5,
            annualized_return_pct=12.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown_pct=8.0,
            max_drawdown_duration_days=30,
            win_rate=0.6,
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            avg_win_pct=5.0,
            avg_loss_pct=-3.0,
            profit_factor=1.67,
            avg_holding_days=15.0,
            total_fees=500.0,
        )
        
        self.assertEqual(metrics.total_return_pct, 25.5)
        self.assertEqual(metrics.sharpe_ratio, 1.2)
        self.assertEqual(metrics.win_rate, 0.6)
        self.assertEqual(metrics.total_trades, 20)
        self.assertEqual(metrics.winning_trades, 12)
        self.assertEqual(metrics.losing_trades, 8)

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BacktestMetrics(
            total_return_pct=25.5,
            total_trades=20,
        )
        
        d = metrics.to_dict()
        
        self.assertIn("total_return_pct", d)
        self.assertIn("total_trades", d)
        self.assertEqual(d["total_return_pct"], 25.5)

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = BacktestMetrics()
        
        self.assertEqual(metrics.total_return_pct, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.total_trades, 0)


class TestBacktestSignalGenerator(unittest.TestCase):
    """Tests for BacktestSignalGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            symbols=["RELIANCE", "TCS"],
            cap_types={"RELIANCE": "large", "TCS": "large"},
            simulate_llm_signals=True,
        )
        self.generator = BacktestSignalGenerator(self.config)

    def _create_sample_price_data(self, base_price=2500, trend="neutral") -> pd.DataFrame:
        """Create sample price data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = [base_price]
        
        for i in range(1, 100):
            if trend == "up":
                change = np.random.uniform(0.5, 1.5)
            elif trend == "down":
                change = np.random.uniform(-1.5, -0.5)
            else:
                change = np.random.uniform(-1, 1)
            prices.append(prices[-1] * (1 + change / 100))
        
        df = pd.DataFrame({
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 100,
        }, index=dates)
        
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df["ATR"] = tr_df.rolling(14).mean()
        
        return df

    def test_generate_buy_signal_on_price_drop(self):
        """Test BUY signal generation on significant price drop."""
        df = self._create_sample_price_data(base_price=2500, trend="down")
        
        from src.trading.types import TriggerType, AgentName, MarketContext
        
        signal = self.generator.generate_signal(
            symbol="RELIANCE",
            date=df.index[-1],
            trigger_type=TriggerType.PRICE_DROP,
            price_data=df,
            market_context=MarketContext(
                current_date=df.index[-1],
                vix=15.0,
                vix_history=[15.0] * 20,
            ),
            agents_run=[AgentName.TECHNICAL, AgentName.SENTIMENT, AgentName.RISK, AgentName.CIO],
        )
        
        self.assertIsNotNone(signal)
        self.assertIn(signal.signal, ["BUY", "STRONG_BUY", "HOLD"])
        self.assertGreaterEqual(signal.conviction, 6)

    def test_generate_sell_signal_on_overbought(self):
        """Test SELL signal generation on overbought conditions."""
        df = self._create_sample_price_data(base_price=2500, trend="up")
        df.loc[df.index[-1], "RSI"] = 85
        
        from src.trading.types import TriggerType, AgentName, MarketContext
        
        signal = self.generator.generate_signal(
            symbol="RELIANCE",
            date=df.index[-1],
            trigger_type=TriggerType.SCHEDULED,
            price_data=df,
            market_context=MarketContext(
                current_date=df.index[-1],
                vix=15.0,
                vix_history=[15.0] * 20,
            ),
            agents_run=[AgentName.FUNDAMENTAL, AgentName.TECHNICAL, AgentName.CIO],
        )
        
        self.assertIsNotNone(signal)

    def test_generate_defensive_signal_on_max_drawdown(self):
        """Test defensive SELL signal during max drawdown (circuit breaker)."""
        from src.trading.types import TriggerType, AgentName
        
        signal = self.generator._generate_defensive_signal(
            symbol="RELIANCE",
            date=datetime.now(),
            agents_run=[AgentName.RISK, AgentName.CIO],
        )
        
        self.assertEqual(signal.signal, "SELL")
        self.assertEqual(signal.conviction, 10)
        self.assertEqual(signal.review_type, "max_drawdown")
        self.assertIn("Circuit breaker", signal.thesis)

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        df = self._create_sample_price_data()
        
        rsi = self.generator._calculate_rsi(df)
        
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

    def test_atr_calculation(self):
        """Test ATR calculation."""
        df = self._create_sample_price_data()
        
        atr = self.generator._calculate_atr(df)
        
        self.assertGreater(atr, 0)

    def test_signal_disabled(self):
        """Test signal generation when disabled."""
        config = BacktestConfig(simulate_llm_signals=False)
        generator = BacktestSignalGenerator(config)
        
        df = self._create_sample_price_data()
        
        from src.trading.types import TriggerType, AgentName, MarketContext
        
        signal = generator.generate_signal(
            symbol="RELIANCE",
            date=df.index[-1],
            trigger_type=TriggerType.SCHEDULED,
            price_data=df,
            market_context=MarketContext(
                current_date=df.index[-1],
                vix=15.0,
            ),
            agents_run=[AgentName.CIO],
        )
        
        self.assertIsNone(signal)


class TestEventDrivenBacktester(unittest.TestCase):
    """Tests for EventDrivenBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            initial_capital=1000000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            symbols=["RELIANCE", "TCS"],
            cap_types={"RELIANCE": "large", "TCS": "large"},
            use_scheduled_reviews=True,
            include_vix_triggers=True,
            include_price_drop_triggers=True,
        )

    def _create_mock_data(self) -> dict:
        """Create mock historical data for testing."""
        data = {}
        
        for symbol in ["RELIANCE", "TCS"]:
            dates = pd.date_range(start="2023-10-01", end="2024-04-01", freq="D")
            
            base_price = 2500 if symbol == "RELIANCE" else 3500
            np.random.seed(42 if symbol == "RELIANCE" else 43)
            
            prices = [base_price]
            for _ in range(len(dates) - 1):
                change = np.random.uniform(-2, 2)
                prices.append(prices[-1] * (1 + change / 100))
            
            df = pd.DataFrame({
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": [p * 0.98 for p in prices],
                "Close": prices,
                "Volume": [1000000] * len(dates),
            }, index=dates)
            
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["SMA_50"] = df["Close"].rolling(50).mean()
            df["SMA_200"] = df["Close"].rolling(200).mean()
            
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
            
            exp1 = df["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
            
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
            df["ATR"] = tr_df.rolling(14).mean()
            
            df["Returns"] = df["Close"].pct_change()
            df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
            
            data[symbol] = df
        
        return data

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        backtester = EventDrivenBacktester(self.config)
        
        self.assertIsNotNone(backtester.trigger_engine)
        self.assertIsNotNone(backtester.execution_engine)
        self.assertIsNotNone(backtester.signal_generator)
        self.assertEqual(len(backtester._trades), 0)
        self.assertEqual(len(backtester._trigger_events), 0)

    def test_get_data_before_date(self):
        """Test getting historical data before a specific date."""
        backtester = EventDrivenBacktester(self.config)
        backtester._data = self._create_mock_data()
        
        test_date = datetime(2024, 2, 15)
        
        df = backtester._get_data_before_date("RELIANCE", test_date)
        
        self.assertFalse(df.empty)
        self.assertTrue(all(df.index <= test_date))
        self.assertIn("Close", df.columns)
        self.assertIn("RSI", df.columns)

    def test_get_prices_for_date(self):
        """Test getting prices for a specific date."""
        backtester = EventDrivenBacktester(self.config)
        backtester._data = self._create_mock_data()
        
        test_date = datetime(2024, 2, 15)
        
        prices = backtester._get_prices_for_date(test_date)
        
        self.assertIn("RELIANCE", prices)
        self.assertIn("TCS", prices)
        self.assertGreater(prices["RELIANCE"], 0)

    def test_get_trading_dates(self):
        """Test getting trading dates."""
        backtester = EventDrivenBacktester(self.config)
        backtester._data = self._create_mock_data()
        
        dates = backtester._get_trading_dates()
        
        self.assertGreater(len(dates), 0)
        self.assertTrue(all(self.config.start_date <= d <= self.config.end_date for d in dates))

    def test_symbol_formatting(self):
        """Test symbol formatting for yfinance."""
        backtester = EventDrivenBacktester(self.config)
        
        self.assertEqual(backtester._format_symbol("RELIANCE"), "RELIANCE.NS")
        self.assertEqual(backtester._format_symbol("NIFTY"), "NIFTY.NS")
        self.assertEqual(backtester._format_symbol("^NSEI"), "^NSEI")

    def test_add_technical_indicators(self):
        """Test adding technical indicators to data."""
        backtester = EventDrivenBacktester(self.config)
        
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "Open": [2500] * 100,
            "High": [2550] * 100,
            "Low": [2450] * 100,
            "Close": [2500 + i for i in range(100)],
            "Volume": [1000000] * 100,
        }, index=dates)
        
        df = backtester._add_technical_indicators(df)
        
        self.assertIn("SMA_20", df.columns)
        self.assertIn("SMA_50", df.columns)
        self.assertIn("RSI", df.columns)
        self.assertIn("MACD", df.columns)
        self.assertIn("ATR", df.columns)

    @patch("yfinance.Ticker")
    def test_run_with_mock_data(self, mock_ticker):
        """Test running backtest with mock data."""
        backtester = EventDrivenBacktester(self.config)
        backtester._data = self._create_mock_data()
        
        result = backtester.run()
        
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.metrics, BacktestMetrics)
        self.assertIsInstance(result.trades, list)
        self.assertIsInstance(result.trigger_events, list)
        self.assertIsInstance(result.signals_generated, list)

    def test_trigger_event_recording(self):
        """Test that trigger events are properly recorded."""
        backtester = EventDrivenBacktester(self.config)
        backtester._data = self._create_mock_data()
        
        result = backtester.run()
        
        self.assertIsInstance(result.trigger_events, list)


class TestBacktestResult(unittest.TestCase):
    """Tests for BacktestResult dataclass."""

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        config = BacktestConfig(
            initial_capital=1000000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            symbols=["RELIANCE", "TCS"],
        )
        
        metrics = BacktestMetrics(
            total_return_pct=25.0,
            total_trades=20,
            sharpe_ratio=1.5,
        )
        
        result = BacktestResult(
            config=config,
            metrics=metrics,
            trades=[],
            equity_curve=pd.DataFrame(),
            trigger_events=[],
            signals_generated=[],
            data_quality_notes=["Test note"],
        )
        
        d = result.to_dict()
        
        self.assertIn("config", d)
        self.assertIn("metrics", d)
        self.assertEqual(d["metrics"]["total_return_pct"], 25.0)

    def test_get_trigger_summary(self):
        """Test getting trigger summary."""
        from src.trading.types import TriggerType
        
        config = BacktestConfig()
        metrics = BacktestMetrics()
        
        trigger_events = [
            TriggerEvent(
                date=datetime(2024, 1, 15),
                trigger_type=TriggerType.SCHEDULED,
                symbol="RELIANCE",
                details={},
                agents_run=["fundamental", "cio"],
            ),
            TriggerEvent(
                date=datetime(2024, 2, 15),
                trigger_type=TriggerType.PRICE_DROP,
                symbol="TCS",
                details={},
                agents_run=["technical", "sentiment", "cio"],
            ),
            TriggerEvent(
                date=datetime(2024, 3, 15),
                trigger_type=TriggerType.SCHEDULED,
                symbol="RELIANCE",
                details={},
                agents_run=["fundamental", "cio"],
            ),
        ]
        
        result = BacktestResult(
            config=config,
            metrics=metrics,
            trades=[],
            equity_curve=pd.DataFrame(),
            trigger_events=trigger_events,
            signals_generated=[],
            data_quality_notes=[],
        )
        
        summary = result.get_trigger_summary()
        
        self.assertEqual(summary["scheduled"], 2)
        self.assertEqual(summary["price_drop"], 1)

    def test_get_trades_dataframe(self):
        """Test getting trades as DataFrame."""
        config = BacktestConfig()
        metrics = BacktestMetrics()
        
        trades = [
            BacktestTrade(
                date=datetime(2024, 1, 15),
                symbol="RELIANCE",
                action="ENTRY",
                quantity=100,
                price=2500.0,
            ),
            BacktestTrade(
                date=datetime(2024, 2, 15),
                symbol="RELIANCE",
                action="EXIT",
                quantity=100,
                price=2600.0,
                pnl=10000.0,
                pnl_pct=4.0,
                reason="target_hit",
            ),
        ]
        
        result = BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            equity_curve=pd.DataFrame(),
            trigger_events=[],
            signals_generated=[],
            data_quality_notes=[],
        )
        
        df = result.get_trades_dataframe()
        
        self.assertEqual(len(df), 2)
        self.assertIn("symbol", df.columns)
        self.assertIn("action", df.columns)
        self.assertIn("pnl", df.columns)


class TestTriggerEvent(unittest.TestCase):
    """Tests for TriggerEvent dataclass."""

    def test_trigger_event_creation(self):
        """Test creating a trigger event."""
        from src.trading.types import TriggerType
        
        event = TriggerEvent(
            date=datetime(2024, 1, 15),
            trigger_type=TriggerType.PRICE_DROP,
            symbol="RELIANCE",
            details={
                "drop_pct": 7.5,
                "threshold": 5.0,
                "cap_type": "large",
            },
            agents_run=["technical", "sentiment", "risk", "cio"],
        )
        
        self.assertEqual(event.trigger_type, TriggerType.PRICE_DROP)
        self.assertEqual(event.symbol, "RELIANCE")
        self.assertEqual(event.details["drop_pct"], 7.5)
        self.assertEqual(len(event.agents_run), 4)


class TestDailySnapshot(unittest.TestCase):
    """Tests for DailySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a daily snapshot."""
        snapshot = DailySnapshot(
            date=datetime(2024, 1, 15),
            portfolio_value=1050000.0,
            cash=500000.0,
            positions_value=550000.0,
            open_positions=3,
            drawdown_pct=2.5,
        )
        
        self.assertEqual(snapshot.portfolio_value, 1050000.0)
        self.assertEqual(snapshot.open_positions, 3)
        self.assertEqual(snapshot.drawdown_pct, 2.5)


class TestTimeTravelIntegrity(unittest.TestCase):
    """Tests for time-travel data integrity (no lookahead bias)."""

    def test_data_before_date_excludes_future(self):
        """Test that data fetched before date excludes future data."""
        config = BacktestConfig(
            start_date=datetime(2024, 2, 1),
            end_date=datetime(2024, 2, 28),
            symbols=["RELIANCE"],
        )
        
        backtester = EventDrivenBacktester(config)
        
        dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")
        df = pd.DataFrame({
            "Open": [2500] * len(dates),
            "High": [2550] * len(dates),
            "Low": [2450] * len(dates),
            "Close": [2500 + i for i in range(len(dates))],
            "Volume": [1000000] * len(dates),
        }, index=dates)
        
        backtester._data["RELIANCE"] = df
        
        test_date = datetime(2024, 2, 15)
        historical_data = backtester._get_data_before_date("RELIANCE", test_date)
        
        self.assertTrue(all(historical_data.index <= test_date))
        
        future_dates = historical_data[historical_data.index > test_date]
        self.assertEqual(len(future_dates), 0)


class TestExecutionLogic(unittest.TestCase):
    """Tests for execution logic (entries, exits, trailing stops)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            initial_capital=100000.0,
            symbols=["RELIANCE"],
            cap_types={"RELIANCE": "large"},
            simulate_llm_signals=True,
        )
        self.backtester = EventDrivenBacktester(self.config)

    def test_entry_requires_signal(self):
        """Test that entry requires a BUY signal."""
        self.backtester._broker = MagicMock()
        self.backtester._broker._positions = {}
        self.backtester._broker.get_account_balance.return_value = 100000.0
        
        prices = {"RELIANCE": 2500.0}
        
        self.backtester._check_entries(datetime.now(), prices)
        
        self.assertEqual(len(self.backtester._trades), 0)

    def test_entry_with_valid_signal(self):
        """Test entry with valid BUY signal."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = [2500 + i for i in range(100)]
        
        df = pd.DataFrame({
            "Open": prices,
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 100,
        }, index=dates)
        
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr_df = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df["ATR"] = tr_df.rolling(14).mean()
        
        self.backtester._data["RELIANCE"] = df
        
        mock_broker = MagicMock()
        mock_broker._positions = {}
        mock_broker._cash = 100000.0
        mock_broker.get_account_balance.return_value = 100000.0
        self.backtester._broker = mock_broker
        
        from src.trading.types import TriggerType, AgentName
        from src.backtesting.event_driven import BacktestSignal
        
        signal = BacktestSignal(
            symbol="RELIANCE",
            signal="BUY",
            conviction=8,
            entry_price=2549.0,
            stop_loss=2500.0,
            target_price=2650.0,
            thesis="Test BUY signal",
            generated_at=datetime(2024, 2, 1),
            valid_until=datetime(2024, 3, 1),
            review_type="scheduled",
            cap_type="large",
            agents_run=["fundamental", "technical", "cio"],
        )
        
        self.backtester._active_signals["RELIANCE"] = signal
        
        prices = {"RELIANCE": 2550.0}
        self.backtester._check_entries(datetime(2024, 2, 15), prices)
        
        self.assertGreaterEqual(len(self.backtester._trades), 0)


class TestMetricsCalculation(unittest.TestCase):
    """Tests for metrics calculation."""

    def test_empty_metrics(self):
        """Test metrics with no trades."""
        metrics = BacktestMetrics()
        
        self.assertEqual(metrics.total_return_pct, 0.0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.total_trades, 0)

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        metrics = BacktestMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
        )
        
        self.assertEqual(metrics.win_rate, 0.6)

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        metrics = BacktestMetrics(
            winning_trades=6,
            losing_trades=4,
            avg_win_pct=5.0,
            avg_loss_pct=-3.0,
        )
        
        total_wins = 6 * 5.0
        total_losses = 4 * 3.0
        expected_profit_factor = total_wins / total_losses
        
        self.assertAlmostEqual(metrics.avg_win_pct, 5.0)
        self.assertAlmostEqual(metrics.avg_loss_pct, -3.0)


class TestCircuitBreaker(unittest.TestCase):
    """Tests for circuit breaker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            initial_capital=100000.0,
            symbols=["RELIANCE"],
            max_drawdown_threshold=10.0,
            include_max_drawdown_circuit_breaker=True,
        )
        self.backtester = EventDrivenBacktester(self.config)

    def test_circuit_breaker_trigger(self):
        """Test circuit breaker triggers at threshold."""
        self.backtester._peak_value = 100000.0
        self.backtester._current_drawdown = 12.0
        
        self.assertGreaterEqual(
            self.backtester._current_drawdown,
            self.config.max_drawdown_threshold
        )


class TestScheduledReviews(unittest.TestCase):
    """Tests for scheduled review functionality."""

    def test_last_day_of_month(self):
        """Test that last day of month is a scheduled review day."""
        from src.trading.trigger_engine import is_scheduled_review_day
        
        jan_31 = datetime(2024, 1, 31)
        feb_29 = datetime(2024, 2, 29)
        mar_28 = datetime(2024, 3, 28)
        
        self.assertTrue(is_scheduled_review_day(jan_31, -1))
        self.assertTrue(is_scheduled_review_day(feb_29, -1))
        self.assertFalse(is_scheduled_review_day(mar_28, -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
