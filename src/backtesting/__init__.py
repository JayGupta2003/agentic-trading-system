"""
Backtesting Module - Custom Pandas-Based Backtesting Engine

This module provides a comprehensive backtesting framework for the AI trading system.
It processes LLM-generated signals (from CIO) against historical data to evaluate
strategy performance.

Modules:
    - engine: Main Backtester class
    - data: Data loading and preprocessing
    - signals: Signal mapping and processing
    - portfolio: Portfolio state management
    - metrics: Performance metrics calculation
    - risk: Risk management rules
    - strategies: Strategy configurations
    - optimization: Parameter optimization
    - reports: Visualization and reporting
    - interface: Pipeline integration
"""

from src.backtesting.engine import Backtester, BacktestConfig, BacktestResult
from src.backtesting.signals import Signal, TradeDirection, SignalProcessor
from src.backtesting.portfolio import PortfolioState, Position, Trade
from src.backtesting.metrics import MetricsCalculator
from src.backtesting.risk import RiskManager
from src.backtesting.optimizer import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    WalkForwardAnalyzer,
    MonteCarloSimulator,
    OutOfSampleValidator,
    SensitivityAnalyzer,
    StatisticalSignificanceTester,
    OverfittingDetector,
    ParameterConfigManager,
    OptimizationReportGenerator,
    generate_weekly_signals
)
from src.backtesting.reports import BacktestVisualizer, PlotConfig

__all__ = [
    "Backtester",
    "BacktestConfig", 
    "BacktestResult",
    "Signal",
    "TradeDirection",
    "SignalProcessor",
    "PortfolioState",
    "Position",
    "Trade",
    "MetricsCalculator",
    "RiskManager",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "WalkForwardAnalyzer",
    "MonteCarloSimulator",
    "OutOfSampleValidator",
    "SensitivityAnalyzer",
    "StatisticalSignificanceTester",
    "OverfittingDetector",
    "ParameterConfigManager",
    "OptimizationReportGenerator",
    "generate_weekly_signals",
    "BacktestVisualizer",
    "PlotConfig",
]
