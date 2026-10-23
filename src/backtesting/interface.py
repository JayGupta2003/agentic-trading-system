"""
Pipeline integration for backtesting.

This module provides the BacktesterInterface class that connects the agent pipeline
(orchestrator output) to the backtesting engine.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

from src.backtesting.engine import Backtester, BacktestConfig, BacktestResult
from src.backtesting.signals import Signal, TradingSignal, SignalProcessor
from src.backtesting.data import DataLoader


@dataclass
class BacktesterInterface:
    """
    Interface between pipeline orchestrator and backtesting engine.
    
    Converts agent analysis (from orchestrator) to trading signals,
    runs backtests, and formats results.
    """
    
    config: BacktestConfig = None
    min_conviction: int = 6
    
    def __post_init__(self):
        if self.config is None:
            self.config = BacktestConfig()
        self.processor = SignalProcessor(min_conviction=self.min_conviction)
        self.backtester = Backtester(self.config)
        self.data_loader = DataLoader()
    """
    Interface between pipeline orchestrator and backtesting engine.
    
    Converts agent analysis (from orchestrator) to trading signals,
    runs backtests, and formats results.
    """
    
    config: Optional[BacktestConfig] = None
    min_conviction: int = 6
    
    def __post_init__(self):
        self.config = self.config or BacktestConfig()
        self.processor = SignalProcessor(min_conviction=self.min_conviction)
        self.backtester = Backtester(self.config)
        self.data_loader = DataLoader()
    
    def convert_rankings_to_signals(
        self,
        ranked_results: List[Dict],
        signal_date: Optional[datetime] = None,
    ) -> Dict[str, List[TradingSignal]]:
        """
        Convert ranked analysis results to trading signals.
        
        Args:
            ranked_results: List of ranked stock results from orchestrator
            signal_date: Date for the signals (default: today)
            
        Returns:
            Dict mapping symbols to lists of TradingSignals
        """
        if signal_date is None:
            signal_date = datetime.now()
        
        signals = {}
        
        for result in ranked_results:
            symbol = result.get("symbol", "")
            if not symbol:
                continue
            
            verdict = result.get("verdict", "HOLD")
            conviction = result.get("conviction", 5)
            signal_type = self.processor.verdict_to_signal(verdict, conviction)
            
            position_size = self._calculate_position_size(signal_type, conviction)
            
            trading_signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                conviction=conviction,
                position_size_pct=position_size,
                timestamp=signal_date,
                reasoning=result.get("analysis", ""),
                expires_days=self.config.signal_validity_days,
            )
            
            signals[symbol] = [trading_signal]
        
        return signals
    
    def convert_single_analysis_to_signal(
        self,
        analysis: Dict,
        signal_date: Optional[datetime] = None,
    ) -> TradingSignal:
        """
        Convert single stock analysis to a trading signal.
        
        Args:
            analysis: Single stock analysis dict
            signal_date: Date for the signal
            
        Returns:
            TradingSignal object
        """
        if signal_date is None:
            signal_date = datetime.now()
        
        symbol = analysis.get("symbol", "")
        verdict = analysis.get("verdict", "HOLD")
        conviction = analysis.get("conviction", 5)
        signal_type = self.processor.verdict_to_signal(verdict, conviction)
        position_size = self._calculate_position_size(signal_type, conviction)
        
        return TradingSignal(
            symbol=symbol,
            signal=signal_type,
            conviction=conviction,
            position_size_pct=position_size,
            timestamp=signal_date,
            reasoning=analysis.get("analysis", ""),
            expires_days=self.config.signal_validity_days,
        )
    
    def _calculate_position_size(self, signal: Signal, conviction: int) -> float:
        """Calculate position size based on signal and conviction."""
        if signal == Signal.STRONG_BUY:
            return min(self.config.max_position_pct, conviction * 2.5)
        elif signal == Signal.BUY:
            return min(self.config.max_position_pct, conviction * 1.5)
        else:
            return 0.0
    
    def run_backtest(
        self,
        signals: Dict[str, List[TradingSignal]],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        benchmark_symbol: str = "^NSEI",
    ) -> BacktestResult:
        """
        Run backtest with given signals.
        
        Args:
            signals: Dict mapping symbols to lists of TradingSignals
            symbols: List of symbols to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            BacktestResult object
        """
        return self.backtester.run(
            symbols=symbols,
            signals=signals,
            start_date=start_date,
            end_date=end_date,
            benchmark_symbol=benchmark_symbol,
        )
    
    def run_backtest_from_pipeline(
        self,
        pipeline_output: Dict,
        start_date: datetime,
        end_date: datetime,
        benchmark_symbol: str = "^NSEI",
    ) -> Dict[str, Any]:
        """
        Run backtest directly from pipeline orchestrator output.
        
        Args:
            pipeline_output: Output from PipelineOrchestrator.run_full_pipeline()
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_symbol: Benchmark symbol
            
        Returns:
            Dict with backtest results and pipeline metadata
        """
        ranked_results = pipeline_output.get("ranked_results", [])
        
        if not ranked_results:
            return {
                "status": "no_signals",
                "message": "No ranked results from pipeline",
            }
        
        signals = self.convert_rankings_to_signals(ranked_results)
        symbols = [r.get("symbol") for r in ranked_results if r.get("symbol") is not None]
        
        backtest_result = self.run_backtest(
            signals=signals,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            benchmark_symbol=benchmark_symbol,
        )
        
        return {
            "status": "success",
            "pipeline_timestamp": pipeline_output.get("timestamp"),
            "backtest_result": backtest_result,
            "signals_generated": len(signals),
            "symbols": symbols,
        }
    
    def get_backtest_summary(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Get formatted summary from backtest result.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dict with formatted summary
        """
        metrics = result.metrics
        
        final_capital = 0
        if len(result.equity_curve) > 0 and "portfolio" in result.equity_curve.columns:
            final_capital = result.equity_curve["portfolio"].iloc[-1]
        
        return {
            "initial_capital": self.config.initial_capital,
            "final_capital": final_capital,
            "total_return_pct": metrics.get("total_return_pct", 0),
            "annualized_return_pct": metrics.get("annualized_return_pct", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "benchmark_return_pct": metrics.get("benchmark_return_pct", 0),
            "alpha": metrics.get("alpha", 0),
        }
    
    def generate_signal_report(
        self,
        ranked_results: List[Dict],
        signal_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate a DataFrame of signals from ranked results.
        
        Args:
            ranked_results: List of ranked stock results
            signal_date: Date for signals
            
        Returns:
            DataFrame with signal details
        """
        if signal_date is None:
            signal_date = datetime.now()
        
        rows = []
        for result in ranked_results:
            verdict = result.get("verdict", "HOLD")
            conviction = result.get("conviction", 5)
            signal_type = self.processor.verdict_to_signal(verdict, conviction)
            position_size = self._calculate_position_size(signal_type, conviction)
            
            rows.append({
                "symbol": result.get("symbol", ""),
                "verdict": verdict,
                "signal": signal_type.value,
                "conviction": conviction,
                "position_size_pct": position_size,
                "tier": result.get("tier", "HOLD"),
                "rank_score": result.get("rank_score", 0),
            })
        
        return pd.DataFrame(rows)
    
    def convert_cio_output_to_signal(
        self,
        cio_output: Dict,
        symbol: str,
        signal_date: Optional[datetime] = None,
    ) -> TradingSignal:
        """
        Convert CIO agent output to a trading signal.
        
        Handles the full CIO analysis structure from orchestrator.
        
        Args:
            cio_output: Full CIO output dict from orchestrator
            symbol: Stock symbol
            signal_date: Date for the signal
            
        Returns:
            TradingSignal object
        """
        if signal_date is None:
            signal_date = datetime.now()
        
        analysis_text = cio_output.get("analysis", "")
        
        verdict = self._extract_verdict_from_text(analysis_text)
        conviction = self._extract_conviction_from_text(analysis_text)
        
        signal_type = self.processor.verdict_to_signal(verdict, conviction)
        position_size = self._calculate_position_size(signal_type, conviction)
        
        return TradingSignal(
            symbol=symbol,
            signal=signal_type,
            conviction=conviction,
            position_size_pct=position_size,
            timestamp=signal_date,
            reasoning=analysis_text[:500],
            expires_days=self.config.signal_validity_days,
        )
    
    def convert_agent_analysis_to_signals(
        self,
        agent_outputs: Dict,
        signal_date: Optional[datetime] = None,
    ) -> Dict[str, List[TradingSignal]]:
        """
        Convert agent analysis outputs to trading signals for multiple stocks.
        
        Args:
            agent_outputs: Dict mapping symbols to their agent analysis results
            signal_date: Date for the signals
            
        Returns:
            Dict mapping symbols to lists of TradingSignals
        """
        if signal_date is None:
            signal_date = datetime.now()
        
        signals = {}
        
        for symbol, analysis in agent_outputs.items():
            if not symbol:
                continue
            
            if isinstance(analysis, dict):
                cio_data = analysis.get("agent_outputs", {}).get("cio", {})
                analysis_text = cio_data.get("analysis", "")
                
                verdict = self._extract_verdict_from_text(analysis_text)
                conviction = self._extract_conviction_from_text(analysis_text)
            else:
                verdict = "HOLD"
                conviction = 5
            
            signal_type = self.processor.verdict_to_signal(verdict, conviction)
            position_size = self._calculate_position_size(signal_type, conviction)
            
            trading_signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                conviction=conviction,
                position_size_pct=position_size,
                timestamp=signal_date,
                reasoning=f"Agent analysis for {symbol}",
                expires_days=self.config.signal_validity_days,
            )
            
            signals[symbol] = [trading_signal]
        
        return signals
    
    def _extract_verdict_from_text(self, text: str) -> str:
        """Extract verdict from CIO analysis text."""
        text_upper = text.upper()
        text_lower = text.lower()
        
        if any(x in text_lower for x in ["strong buy", "strong_buy", "very strong buy", "very bullish"]):
            return "STRONG BUY"
        elif any(x in text_lower for x in ["strong sell", "strong_sell", "very strong sell", "very bearish", "avoid"]):
            return "STRONG SELL"
        elif "BUY" in text_upper and "SELL" not in text_upper:
            return "BUY"
        elif "SELL" in text_upper:
            return "SELL"
        elif any(x in text_lower for x in ["hold", "neutral", "reduce", "watch", "bullish", "bearish"]):
            if any(x in text_lower for x in ["bullish", "positive"]):
                return "BUY"
            elif any(x in text_lower for x in ["bearish", "negative"]):
                return "SELL"
            return "HOLD"
        else:
            return "HOLD"
    
    def _extract_conviction_from_text(self, text: str) -> int:
        """Extract conviction score from CIO analysis text."""
        import re
        
        text_lower = text.lower()
        
        conv_patterns = [
            (r"conviction[:\s]+(\d+)", 1),
            (r"confidence[:\s]+(\d+)", 1),
            (r"score[:\s]+(\d+)", 1),
            (r"rating[:\s]+(\d+)", 1),
        ]
        
        for pattern, group in conv_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = int(match.group(group))
                    if 1 <= val <= 10:
                        return val
                except (ValueError, IndexError):
                    pass
        
        strong_buy_phrases = ["strong buy", "very bullish", "high conviction", "very confident"]
        buy_phrases = ["buy", "bullish", "positive", "outperform"]
        neutral_phrases = ["hold", "neutral", "fair", "equal weight"]
        sell_phrases = ["sell", "bearish", "negative", "underperform"]
        strong_sell_phrases = ["strong sell", "very bearish", "avoid", "strong negative"]
        
        for phrase in strong_sell_phrases:
            if phrase in text_lower:
                return 2
        for phrase in sell_phrases:
            if phrase in text_lower:
                return 3
        for phrase in strong_buy_phrases:
            if phrase in text_lower:
                return 9
        for phrase in buy_phrases:
            if phrase in text_lower:
                return 7
        for phrase in neutral_phrases:
            if phrase in text_lower:
                return 5
        
        return 5
    
    def run_batch_backtest(
        self,
        ranked_results: List[Dict],
        start_date: datetime,
        end_date: datetime,
        benchmark_symbol: str = "^NSEI",
    ) -> Dict[str, Any]:
        """
        Run backtest for multiple screened stocks in batch.
        
        Args:
            ranked_results: List of ranked stock results from orchestrator
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            Dict with batch backtest results and summary statistics
        """
        signals = self.convert_rankings_to_signals(ranked_results)
        symbols = [r.get("symbol") for r in ranked_results if r.get("symbol") is not None]
        
        if not symbols:
            return {
                "status": "no_symbols",
                "message": "No symbols found in ranked results",
            }
        
        backtest_result = self.run_backtest(
            signals=signals,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            benchmark_symbol=benchmark_symbol,
        )
        
        summary = self.get_backtest_summary(backtest_result)
        
        trade_df = backtest_result.trades
        
        return {
            "status": "success",
            "symbols": symbols,
            "num_signals": len(signals),
            "backtest_result": backtest_result,
            "summary": summary,
            "trade_log": trade_df.to_dict('records') if len(trade_df) > 0 else [],
            "num_trades": len(trade_df),
        }
    
    def run_multiple_regime_backtests(
        self,
        ranked_results: List[Dict],
        regimes: List[Dict],
        benchmark_symbol: str = "^NSEI",
    ) -> Dict[str, Any]:
        """
        Run backtests across multiple market regimes.
        
        Args:
            ranked_results: List of ranked stock results
            regimes: List of dicts with 'name', 'start_date', 'end_date'
            benchmark_symbol: Benchmark symbol
            
        Returns:
            Dict with regime-specific results
        """
        results = {}
        
        for regime in regimes:
            regime_name = regime.get("name", "Unknown")
            start = regime.get("start_date")
            end = regime.get("end_date")
            
            if not start or not end:
                results[regime_name] = {
                    "status": "invalid_dates",
                    "message": "Missing start or end date",
                }
                continue
            
            result = self.run_batch_backtest(
                ranked_results=ranked_results,
                start_date=start,
                end_date=end,
                benchmark_symbol=benchmark_symbol,
            )
            
            results[regime_name] = {
                "status": "success",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "summary": result.get("summary", {}),
                "num_trades": result.get("num_trades", 0),
            }
        
        return results
    
    def compare_strategies(
        self,
        ranked_results: List[Dict],
        start_date: datetime,
        end_date: datetime,
        strategy_configs: List[Dict],
        benchmark_symbol: str = "^NSEI",
    ) -> Dict[str, Any]:
        """
        Compare multiple strategy configurations.
        
        Args:
            ranked_results: List of ranked stock results
            start_date: Backtest start date
            end_date: Backtest end date
            strategy_configs: List of config dicts with 'name' and optional BacktestConfig overrides
            benchmark_symbol: Benchmark symbol
            
        Returns:
            Dict with strategy comparison results
        """
        from copy import deepcopy
        
        results = {}
        base_config = deepcopy(self.config)
        
        for config_dict in strategy_configs:
            strategy_name = config_dict.get("name", "Strategy")
            overrides = {k: v for k, v in config_dict.items() if k != "name"}
            
            test_config = deepcopy(base_config)
            for key, value in overrides.items():
                if hasattr(test_config, key):
                    setattr(test_config, key, value)
            
            original_config = self.config
            self.config = test_config
            self.backtester = Backtester(test_config)
            
            signals = self.convert_rankings_to_signals(ranked_results)
            symbols = [r.get("symbol") for r in ranked_results if r.get("symbol") is not None]
            
            backtest_result = self.run_backtest(
                signals=signals,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                benchmark_symbol=benchmark_symbol,
            )
            
            summary = self.get_backtest_summary(backtest_result)
            
            results[strategy_name] = {
                "config": overrides,
                "summary": summary,
                "num_trades": len(backtest_result.trades),
                "win_rate": summary.get("win_rate", 0),
                "total_return": summary.get("total_return_pct", 0),
                "sharpe": summary.get("sharpe_ratio", 0),
                "max_dd": summary.get("max_drawdown_pct", 0),
            }
            
            self.config = original_config
            self.backtester = Backtester(original_config)
        
        return results
    
    def generate_dashboard_summary(
        self,
        backtest_result,
        include_charts: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate dashboard-ready summary from backtest result.
        
        Args:
            backtest_result: BacktestResult object
            include_charts: Whether to include chart data
            
        Returns:
            Dict ready for Streamlit dashboard display
        """
        metrics = backtest_result.metrics
        equity_curve = backtest_result.equity_curve
        trades = backtest_result.trades
        
        summary = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "return": {
                    "total": metrics.get("total_return_pct", 0),
                    "annualized": metrics.get("annualized_return_pct", 0),
                    "benchmark": metrics.get("benchmark_return_pct", 0),
                    "alpha": metrics.get("alpha", 0),
                },
                "risk": {
                    "sharpe": metrics.get("sharpe_ratio", 0),
                    "sortino": metrics.get("sortino_ratio", 0),
                    "max_drawdown": metrics.get("max_drawdown_pct", 0),
                    "volatility": metrics.get("volatility", 0),
                    "var_95": metrics.get("var_95", 0),
                },
                "trades": {
                    "total": metrics.get("total_trades", 0),
                    "winners": metrics.get("winning_trades", 0),
                    "losers": metrics.get("losing_trades", 0),
                    "win_rate": metrics.get("win_rate", 0),
                    "avg_win": metrics.get("avg_win", 0),
                    "avg_loss": metrics.get("avg_loss", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                },
            },
            "config": {
                "initial_capital": self.config.initial_capital,
                "max_positions": self.config.max_positions,
                "min_conviction": self.min_conviction,
            },
        }
        
        if include_charts and len(equity_curve) > 0:
            summary["chart_data"] = {
                "equity_curve": {
                    "dates": equity_curve.index.strftime('%Y-%m-%d').tolist() if hasattr(equity_curve.index, 'strftime') else [],
                    "portfolio": equity_curve["portfolio"].tolist() if "portfolio" in equity_curve.columns else [],
                    "benchmark": equity_curve["benchmark"].tolist() if "benchmark" in equity_curve.columns else [],
                },
                "trades": trades.to_dict('records') if len(trades) > 0 else [],
            }
        
        return summary
    
    def generate_trade_summary(self, trades) -> Dict[str, Any]:
        """
        Generate trade summary statistics for dashboard.
        
        Args:
            trades: DataFrame of trades
            
        Returns:
            Dict with trade statistics
        """
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
            }
        
        winning_trades = trades[trades['pnl'] > 0] if 'pnl' in trades.columns else pd.DataFrame()
        losing_trades = trades[trades['pnl'] < 0] if 'pnl' in trades.columns else pd.DataFrame()
        
        return {
            "total_trades": len(trades),
            "winners": len(winning_trades),
            "losers": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0,
            "avg_win": winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            "total_pnl": trades['pnl'].sum() if 'pnl' in trades.columns else 0,
            "best_trade": trades['pnl'].max() if 'pnl' in trades.columns else 0,
            "worst_trade": trades['pnl'].min() if 'pnl' in trades.columns else 0,
            "avg_holding_days": trades['holding_days'].mean() if 'holding_days' in trades.columns else 0,
        }
    
    def generate_metrics_card(self, backtest_result) -> Dict[str, Dict]:
        """
        Generate metrics card data for Streamlit columns.
        
        Args:
            backtest_result: BacktestResult object
            
        Returns:
            Dict mapping metric names to {label, value, delta, format} dicts
        """
        metrics = backtest_result.metrics
        
        return {
            "total_return": {
                "label": "Total Return",
                "value": f"{metrics.get('total_return_pct', 0):.2f}%",
                "delta": metrics.get('alpha', 0),
                "format": "percent",
            },
            "sharpe": {
                "label": "Sharpe Ratio",
                "value": f"{metrics.get('sharpe_ratio', 0):.2f}",
                "delta": None,
                "format": "number",
            },
            "max_drawdown": {
                "label": "Max Drawdown",
                "value": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                "delta": None,
                "format": "percent",
            },
            "win_rate": {
                "label": "Win Rate",
                "value": f"{metrics.get('win_rate', 0):.1f}%",
                "delta": None,
                "format": "percent",
            },
            "total_trades": {
                "label": "Total Trades",
                "value": str(metrics.get("total_trades", 0)),
                "delta": None,
                "format": "number",
            },
            "profit_factor": {
                "label": "Profit Factor",
                "value": f"{metrics.get('profit_factor', 0):.2f}",
                "delta": None,
            "format": "number",
        },
    }
    
    def extract_risk_params_from_agent(
        self,
        agent_output: Dict,
    ) -> Dict[str, Any]:
        """
        Extract risk parameters from Agent 5 (RiskAgent) output.
        
        Args:
            agent_output: Dict from RiskAgent.analyze()
            
        Returns:
            Dict with extracted risk parameters
        """
        return {
            "position_size_pct": agent_output.get("position_size_pct", 0),
            "stop_loss": agent_output.get("stop_loss", 0),
            "max_loss_per_trade": agent_output.get("max_loss_per_trade", 0),
            "risk_reward": agent_output.get("risk_reward", 0),
            "kelly_full": agent_output.get("kelly_full", 0),
            "kelly_half": agent_output.get("kelly_half", 0),
            "verdict": self._extract_risk_verdict(agent_output.get("analysis", "")),
        }
    
    def _extract_risk_verdict(self, analysis_text: str) -> str:
        """Extract risk verdict from analysis text."""
        text_upper = analysis_text.upper()
        if "REJECT" in text_upper:
            return "REJECT"
        elif "CAUTION" in text_upper:
            return "CAUTION"
        elif "APPROVE" in text_upper:
            return "APPROVE"
        return "UNKNOWN"
    
    def apply_risk_params_to_config(
        self,
        risk_params: Dict[str, Any],
    ) -> None:
        """
        Apply risk parameters from Agent 5 to backtest config.
        
        Args:
            risk_params: Dict from extract_risk_params_from_agent()
        """
        if risk_params.get("position_size_pct", 0) > 0:
            self.config.max_position_pct = min(
                risk_params["position_size_pct"],
                self.config.max_position_pct
            )
        
        if risk_params.get("stop_loss", 0) > 0:
            self.config.default_stop_loss_atr = 2.0
    
    def create_signal_with_risk(
        self,
        base_signal: TradingSignal,
        risk_params: Dict[str, Any],
        current_price: float,
    ) -> TradingSignal:
        """
        Create enhanced signal with risk parameters from Agent 5.
        
        Args:
            base_signal: TradingSignal from convert_rankings_to_signals()
            risk_params: Dict from extract_risk_params_from_agent()
            current_price: Current stock price
            
        Returns:
            Enhanced TradingSignal with stop loss and position size
        """
        position_size = risk_params.get("position_size_pct", base_signal.position_size_pct)
        
        if risk_params.get("stop_loss", 0) > 0:
            stop_loss = risk_params["stop_loss"]
        else:
            stop_loss = current_price * 0.95
        
        if risk_params.get("max_loss_per_trade", 0) > 0:
            target_pct = self.config.default_target_pct
        else:
            target_pct = self.config.default_target_pct
        
        target_price = current_price * (1 + target_pct / 100)
        
        return TradingSignal(
            symbol=base_signal.symbol,
            signal=base_signal.signal,
            conviction=base_signal.conviction,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size_pct=position_size,
            timestamp=base_signal.timestamp,
            reasoning=f"{base_signal.reasoning}\n\nRisk Params: {risk_params.get('verdict', 'N/A')}",
            expires_days=self.config.signal_validity_days,
        )
    
    def apply_risk_limits(
        self,
        max_position_pct: float = None,
        max_drawdown_pct: float = None,
        max_daily_loss_pct: float = None,
        stop_loss_atr: float = None,
        trailing_stop_atr: float = None,
    ) -> None:
        """
        Apply risk limits from Agent 5 to backtest config.
        
        Args:
            max_position_pct: Maximum position size as % of capital
            max_drawdown_pct: Maximum drawdown before stopping
            max_daily_loss_pct: Maximum daily loss before stopping
            stop_loss_atr: ATR multiplier for stop loss
            trailing_stop_atr: ATR multiplier for trailing stop
        """
        if max_position_pct is not None:
            self.config.max_position_pct = max_position_pct
        
        if max_drawdown_pct is not None:
            self.config.max_drawdown_pct = max_drawdown_pct
        
        if max_daily_loss_pct is not None:
            self.config.max_daily_loss_pct = max_daily_loss_pct
        
        if stop_loss_atr is not None:
            self.config.default_stop_loss_atr = stop_loss_atr
        
        if trailing_stop_atr is not None:
            self.config.default_trailing_stop_atr = trailing_stop_atr
        
        self.config.__post_init__()
    
    def update_signal(
        self,
        signal: TradingSignal,
        new_conviction: int,
        new_verdict: str = None,
        timestamp: datetime = None,
    ) -> TradingSignal:
        """
        Update an existing signal with new conviction or verdict.
        
        Args:
            signal: Existing TradingSignal
            new_conviction: New conviction score (1-10)
            new_verdict: Optional new verdict string
            timestamp: Update timestamp
            
        Returns:
            Updated TradingSignal
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if new_verdict:
            signal_type = self.processor.verdict_to_signal(new_verdict, new_conviction)
        else:
            signal_type = signal.signal
        
        position_size = self._calculate_position_size(signal_type, new_conviction)
        
        signal.signal = signal_type
        signal.conviction = new_conviction
        signal.position_size_pct = position_size
        signal.timestamp = timestamp
        
        return signal
    
    def check_signal_expiration(
        self,
        signal: TradingSignal,
        current_date: datetime,
    ) -> bool:
        """
        Check if a signal has expired.
        
        Args:
            signal: TradingSignal to check
            current_date: Current date
            
        Returns:
            True if signal has expired, False otherwise
        """
        if signal.timestamp is None:
            return False
        
        days_since_signal = (current_date.date() - signal.timestamp.date()).days
        return days_since_signal > signal.expires_days
    
    def refresh_signals(
        self,
        signals: Dict[str, List[TradingSignal]],
        current_date: datetime,
    ) -> Dict[str, List[TradingSignal]]:
        """
        Refresh signals by removing expired ones and updating timestamps.
        
        Args:
            signals: Dict of symbol -> list of TradingSignals
            current_date: Current date for expiration check
            
        Returns:
            Updated signals dict with expired signals removed
        """
        refreshed = {}
        
        for symbol, sig_list in signals.items():
            valid_signals = []
            for sig in sig_list:
                if not self.check_signal_expiration(sig, current_date):
                    valid_signals.append(sig)
                else:
                    pass
            
            if valid_signals:
                refreshed[symbol] = valid_signals
        
        return refreshed
    
    def merge_signals(
        self,
        existing_signals: Dict[str, List[TradingSignal]],
        new_signals: Dict[str, List[TradingSignal]],
        current_date: datetime,
    ) -> Dict[str, List[TradingSignal]]:
        """
        Merge new signals with existing ones, updating or adding as needed.
        
        Args:
            existing_signals: Current signals dict
            new_signals: New signals to merge
            current_date: Current date
            
        Returns:
            Merged signals dict
        """
        merged = self.refresh_signals(existing_signals, current_date)
        
        for symbol, sig_list in new_signals.items():
            if symbol in merged:
                merged[symbol].extend(sig_list)
            else:
                merged[symbol] = sig_list
        
        return merged
    
    def get_active_signals(
        self,
        signals: Dict[str, List[TradingSignal]],
        current_date: datetime,
    ) -> Dict[str, List[TradingSignal]]:
        """
        Get only active (non-expired) signals.
        
        Args:
            signals: Dict of symbol -> list of TradingSignals
            current_date: Current date
            
        Returns:
            Dict with only active signals
        """
        return self.refresh_signals(signals, current_date)
    
    def get_signal_summary(
        self,
        signals: Dict[str, List[TradingSignal]],
        current_date: datetime,
    ) -> Dict[str, Any]:
        """
        Get summary of all signals.
        
        Args:
            signals: Dict of symbol -> list of TradingSignals
            current_date: Current date
            
        Returns:
            Dict with signal statistics
        """
        active = self.get_active_signals(signals, current_date)
        
        all_signals = []
        for sig_list in active.values():
            all_signals.extend(sig_list)
        
        if not all_signals:
            return {
                "total_symbols": 0,
                "total_signals": 0,
                "active_signals": 0,
                "signal_breakdown": {},
            }
        
        signal_counts = {}
        for sig in all_signals:
            sig_type = sig.signal.value
            signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1
        
        return {
            "total_symbols": len(active),
            "total_signals": len(all_signals),
            "active_signals": len(all_signals),
            "signal_breakdown": signal_counts,
            "avg_conviction": sum(s.conviction for s in all_signals) / len(all_signals),
        }
    
    def add_backtest_to_pipeline(
        self,
        orchestrator,
        start_date: datetime,
        end_date: datetime,
        run_backtest: bool = True,
    ) -> Dict[str, Any]:
        """
        Add backtest capability to orchestrator pipeline.
        
        This method enhances the orchestrator with backtest functionality
        by adding the backtester_interface and running backtest on ranked results.
        
        Args:
            orchestrator: PipelineOrchestrator instance
            start_date: Backtest start date
            end_date: Backtest end date
            run_backtest: Whether to run backtest
            
        Returns:
            Dict with enhanced orchestrator and backtest results
        """
        orchestrator.backtester_interface = self
        
        result = {
            "status": "ready",
            "backtester_interface": self,
            "backtest_config": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "run_backtest": run_backtest,
            },
        }
        
        if run_backtest and hasattr(orchestrator, 'ranked_results'):
            ranked_results = orchestrator.ranked_results
            if ranked_results:
                backtest_result = self.run_batch_backtest(
                    ranked_results=ranked_results,
                    start_date=start_date,
                    end_date=end_date,
                )
                result["backtest"] = backtest_result
                result["dashboard_summary"] = self.generate_dashboard_summary(
                    backtest_result.get("backtest_result") if isinstance(backtest_result, dict) else backtest_result
                )
        
        return result
    
    def create_pipeline_integration(self) -> Dict[str, Any]:
        """
        Create a complete pipeline integration configuration.
        
        Returns:
            Dict with integration settings and methods
        """
        return {
            "interface": self,
            "supported_methods": [
                "run_full_pipeline",
                "run_batch_backtest",
                "generate_dashboard_summary",
                "convert_rankings_to_signals",
                "run_backtest_from_pipeline",
            ],
            "signal_handlers": [
                "update_signal",
                "refresh_signals",
                "merge_signals",
            ],
            "risk_handlers": [
                "extract_risk_params_from_agent",
                "apply_risk_params_to_config",
                "create_signal_with_risk",
            ],
            "dashboard_generators": [
                "generate_dashboard_summary",
                "generate_metrics_card",
                "generate_trade_summary",
            ],
        }
    
    def run_complete_workflow(
        self,
        pipeline_output: Dict,
        start_date: datetime,
        end_date: datetime,
        include_risk: bool = True,
        include_visualization: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete backtesting workflow from pipeline to dashboard.
        
        Args:
            pipeline_output: Output from PipelineOrchestrator
            start_date: Backtest start date
            end_date: Backtest end date
            include_risk: Include risk parameter integration
            include_visualization: Generate dashboard summaries
            
        Returns:
            Complete workflow results
        """
        ranked_results = pipeline_output.get("ranked_results", [])
        
        if not ranked_results:
            return {
                "status": "no_signals",
                "message": "No ranked results from pipeline",
            }
        
        signals = self.convert_rankings_to_signals(ranked_results)
        symbols = [r.get("symbol") for r in ranked_results if r.get("symbol") is not None]
        
        backtest_result = self.run_backtest(
            signals=signals,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        
        result = {
            "status": "success",
            "pipeline_timestamp": pipeline_output.get("timestamp"),
            "symbols": symbols,
            "num_signals": len(signals),
            "backtest_result": backtest_result,
        }
        
        if include_visualization:
            result["dashboard_summary"] = self.generate_dashboard_summary(backtest_result)
            result["metrics_card"] = self.generate_metrics_card(backtest_result)
            result["trade_summary"] = self.generate_trade_summary(backtest_result.trades)
        
        return result
    
    def generate_pdf_report(
        self,
        backtest_result,
        output_path: str = "backtest_report.pdf",
        title: str = "Backtest Report",
        include_charts: bool = False,
    ) -> str:
        """
        Generate PDF backtest report.
        
        Args:
            backtest_result: BacktestResult object
            output_path: Path to save PDF
            title: Report title
            include_charts: Whether to include charts (requires matplotlib figures)
            
        Returns:
            Path to saved PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        except ImportError:
            raise ImportError("reportlab required for PDF generation. Install with: pip install reportlab")
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
        )
        
        story = []
        
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.25 * inch))
        
        metrics = backtest_result.metrics
        config = backtest_result.config
        
        story.append(Paragraph("Performance Summary", heading_style))
        
        summary_data = [
            ["Metric", "Value"],
            ["Initial Capital", f"Rs {config.initial_capital:,.0f}"],
            ["Total Return", f"{metrics.get('total_return_pct', 0):.2f}%"],
            ["Annualized Return", f"{metrics.get('annualized_return_pct', 0):.2f}%"],
            ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ["Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"],
            ["Win Rate", f"{metrics.get('win_rate', 0):.1f}%"],
            ["Total Trades", str(metrics.get('total_trades', 0))],
        ]
        
        summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F0FE')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5 * inch))
        
        story.append(Paragraph("Risk Metrics", heading_style))
        
        risk_data = [
            ["Metric", "Value"],
            ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"],
            ["Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}"],
            ["Volatility", f"{metrics.get('volatility', 0):.2f}%"],
            ["Value at Risk (95%)", f"{metrics.get('var_95', 0):.2f}%"],
            ["Conditional VaR", f"{metrics.get('cvar_95', 0):.2f}%"],
            ["Benchmark Return", f"{metrics.get('benchmark_return_pct', 0):.2f}%"],
            ["Alpha", f"{metrics.get('alpha', 0):.2f}%"],
        ]
        
        risk_table = Table(risk_data, colWidths=[3 * inch, 2 * inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC3545')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFF3CD')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.5 * inch))
        
        trades = backtest_result.trades
        if len(trades) > 0:
            story.append(Paragraph("Trade Summary", heading_style))
            
            trade_data = [["Symbol", "P&L", "Holding Days", "Return %"]]
            for _, trade in trades.head(10).iterrows():
                trade_data.append([
                    str(trade.get('symbol', 'N/A'))[:10],
                    f"Rs {trade.get('pnl', 0):.2f}",
                    str(trade.get('holding_days', 0)),
                    f"{trade.get('pnl_pct', 0):.2f}%" if 'pnl_pct' in trade else "N/A"
                ])
            
            trade_table = Table(trade_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28A745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(trade_table)
        
        doc.build(story)
        
        return output_path
    
    def generate_json_report(
        self,
        backtest_result,
        output_path: str = "backtest_report.json",
    ) -> str:
        """
        Generate JSON backtest report.
        
        Args:
            backtest_result: BacktestResult object
            output_path: Path to save JSON
            
        Returns:
            Path to saved JSON
        """
        import json
        from datetime import datetime
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "initial_capital": backtest_result.config.initial_capital,
                "max_positions": backtest_result.config.max_positions,
                "min_conviction": backtest_result.config.min_conviction,
                "risk_free_rate": backtest_result.config.risk_free_rate,
            },
            "metrics": backtest_result.metrics,
            "trades": self._serialize_trades(backtest_result.trades) if len(backtest_result.trades) > 0 else [],
            "equity_curve_summary": {
                "start_value": float(backtest_result.equity_curve["portfolio"].iloc[0]) if len(backtest_result.equity_curve) > 0 else 0,
                "end_value": float(backtest_result.equity_curve["portfolio"].iloc[-1]) if len(backtest_result.equity_curve) > 0 else 0,
                "trading_days": len(backtest_result.equity_curve) if len(backtest_result.equity_curve) > 0 else 0,
            },
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def _serialize_trades(self, trades_df: pd.DataFrame) -> list:
        """Serialize trades DataFrame to JSON-compatible format."""
        records = []
        for _, row in trades_df.iterrows():
            record = {}
            for col, value in row.items():
                if hasattr(value, 'isoformat'):
                    record[col] = value.isoformat()
                elif pd.isna(value):
                    record[col] = None
                else:
                    record[col] = value
            records.append(record)
        return records
    
    def save_backtest_result(
        self,
        backtest_result,
        output_dir: str = "backtest_results",
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Save complete BacktestResult to JSON with metadata.
        
        Args:
            backtest_result: BacktestResult object to save
            output_dir: Directory to save results
            name: Optional name for the backtest (auto-generated if None)
            tags: Optional list of tags for categorization
            
        Returns:
            Path to saved JSON file
        """
        import json
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        if name is None:
            name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        metadata = {
            "version": "1.0",
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "tags": tags or [],
        }
        
        trades_records = self._serialize_trades(backtest_result.trades) if len(backtest_result.trades) > 0 else []
        
        equity_curve_records = []
        for _, row in backtest_result.equity_curve.iterrows():
            record = {}
            for col, value in row.items():
                if hasattr(value, 'isoformat'):
                    record[col] = value.isoformat()
                elif pd.isna(value):
                    record[col] = None
                else:
                    record[col] = value
            equity_curve_records.append(record)
        
        config_dict = {}
        for field_name in ["initial_capital", "brokerage_rate", "stt_rate", 
                          "exchange_txn_charge", "gst_rate", "max_positions",
                          "max_position_pct", "min_conviction", "default_stop_loss_atr",
                          "default_trailing_stop_atr", "default_target_pct", 
                          "warmup_days", "signal_validity_days", "risk_free_rate",
                          "max_drawdown_pct", "max_daily_loss_pct", "entry_type",
                          "exit_type", "allow_short", "allow_partial_exit",
                          "benchmark_symbol"]:
            if hasattr(backtest_result.config, field_name):
                config_dict[field_name] = getattr(backtest_result.config, field_name)
        
        full_result = {
            "metadata": metadata,
            "config": config_dict,
            "metrics": backtest_result.metrics,
            "summary": {
                "total_trades": len(trades_records),
                "total_return_pct": backtest_result.metrics.get("total_return_pct", 0),
                "sharpe_ratio": backtest_result.metrics.get("sharpe_ratio", 0),
                "max_drawdown_pct": backtest_result.metrics.get("max_drawdown_pct", 0),
                "win_rate": backtest_result.metrics.get("win_rate", 0),
                "execution_time_seconds": backtest_result.execution_time_seconds,
            },
            "trades": trades_records,
            "trade_log": [
                {k: (v.isoformat() if hasattr(v, 'isoformat') else v) 
                 for k, v in log.items()} 
                for log in backtest_result.trade_log
            ],
            "equity_curve": equity_curve_records,
            "errors": backtest_result.errors,
        }
        
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, 'w') as f:
            json.dump(full_result, f, indent=2)
        
        return output_path
    
    def load_backtest_result(self, file_path: str) -> Dict[str, Any]:
        """
        Load BacktestResult from JSON file.
        
        Args:
            file_path: Path to JSON file created by save_backtest_result
            
        Returns:
            Dictionary with loaded data (metadata, config, metrics, trades, equity_curve)
        """
        import json
        from src.backtesting.engine import BacktestConfig
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        config_data = data.get("config", {})
        config = BacktestConfig(
            initial_capital=config_data.get("initial_capital", 1000000.0),
            max_positions=config_data.get("max_positions", 5),
            min_conviction=config_data.get("min_conviction", 6),
            risk_free_rate=config_data.get("risk_free_rate", 0.06),
            max_drawdown_pct=config_data.get("max_drawdown_pct", 15.0),
            max_daily_loss_pct=config_data.get("max_daily_loss_pct", 5.0),
            entry_type=config_data.get("entry_type", "eod"),
            exit_type=config_data.get("exit_type", "eod"),
            allow_short=config_data.get("allow_short", False),
            allow_partial_exit=config_data.get("allow_partial_exit", True),
            benchmark_symbol=config_data.get("benchmark_symbol", "^NSEI"),
        )
        
        trades_df = pd.DataFrame(data.get("trades", []))
        equity_curve_df = pd.DataFrame(data.get("equity_curve", []))
        
        return {
            "metadata": data.get("metadata", {}),
            "config": config,
            "metrics": data.get("metrics", {}),
            "summary": data.get("summary", {}),
            "trades": trades_df,
            "trade_log": data.get("trade_log", []),
            "equity_curve": equity_curve_df,
            "errors": data.get("errors", []),
        }
    
    def list_backtest_results(self, output_dir: str = "backtest_results") -> List[Dict[str, Any]]:
        """
        List all saved backtest results in directory.
        
        Args:
            output_dir: Directory to search for backtest results
            
        Returns:
            List of metadata summaries for each backtest
        """
        import json
        import os
        from datetime import datetime
        
        results = []
        
        if not os.path.exists(output_dir):
            return results
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(output_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    metadata = data.get("metadata", {})
                    summary = data.get("summary", {})
                    
                    results.append({
                        "filename": filename,
                        "file_path": file_path,
                        "name": metadata.get("name", filename.replace('.json', '')),
                        "saved_at": metadata.get("saved_at", ""),
                        "tags": metadata.get("tags", []),
                        "total_trades": summary.get("total_trades", 0),
                        "total_return_pct": summary.get("total_return_pct", 0),
                        "sharpe_ratio": summary.get("sharpe_ratio", 0),
                        "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
                        "win_rate": summary.get("win_rate", 0),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        
        results.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return results
    
    def check_data_sufficiency(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_days: int = 200,
    ) -> Dict[str, Any]:
        """
        Check if data is sufficient for backtest.
        
        Args:
            symbols: List of stock symbols to check
            start_date: Start date for backtest
            end_date: End date for backtest
            min_days: Minimum trading days required
            
        Returns:
            Dictionary with sufficiency report
        """
        data = self.data_loader.load_data(symbols, start_date, end_date, "3y")
        
        report = {
            "is_sufficient": True,
            "total_symbols": len(symbols),
            "loaded_symbols": len(data),
            "missing_symbols": [],
            "insufficient_symbols": [],
            "symbols_detail": {},
        }
        
        for symbol in symbols:
            symbol_data = data.get(symbol)
            
            if symbol_data is None or symbol_data.empty:
                report["missing_symbols"].append(symbol)
                report["symbols_detail"][symbol] = {
                    "status": "missing",
                    "data_points": 0,
                }
                continue
            
            days_count = len(symbol_data)
            has_indicators = all(
                col in symbol_data.columns 
                for col in ['SMA_50', 'SMA_200', 'RSI', 'ATR']
            )
            
            if not has_indicators:
                symbol_data = self.data_loader.add_technical_indicators(symbol_data)
            
            detail = {
                "status": "ok",
                "data_points": days_count,
                "min_days_met": days_count >= min_days,
                "date_range": {
                    "start": symbol_data.index[0].isoformat() if len(symbol_data) > 0 else None,
                    "end": symbol_data.index[-1].isoformat() if len(symbol_data) > 0 else None,
                },
            }
            
            if days_count < min_days:
                detail["status"] = "insufficient"
                detail["shortage_days"] = min_days - days_count
                report["insufficient_symbols"].append(symbol)
            
            report["symbols_detail"][symbol] = detail
        
        if report["missing_symbols"] or report["insufficient_symbols"]:
            report["is_sufficient"] = False
        
        return report
    
    def validate_backtest_prerequisites(
        self,
        symbols: List[str],
        signals: Dict[str, List[TradingSignal]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Validate all prerequisites for running backtest.
        
        Args:
            symbols: List of stock symbols
            signals: Trading signals dictionary
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Validation result with status and issues
        """
        issues = []
        warnings = []
        recommendations = []
        
        if not symbols:
            issues.append("No symbols provided for backtest")
        
        if not signals:
            issues.append("No signals provided for backtest")
        
        if start_date and end_date:
            if start_date >= end_date:
                issues.append("Start date must be before end date")
            
            trading_days_estimate = (end_date - start_date).days * 0.65
            if trading_days_estimate < 100:
                warnings.append(
                    f"Date range may be too short ({trading_days_estimate:.0f} estimated trading days)"
                )
        
        data_sufficiency = self.check_data_sufficiency(
            symbols, start_date, end_date, 
            min_days=self.config.warmup_days + 50
        )
        
        if not data_sufficiency["is_sufficient"]:
            for symbol in data_sufficiency["missing_symbols"]:
                issues.append(f"Missing data for symbol: {symbol}")
                recommendations.append(
                    f"Verify {symbol} is a valid NSE/BSE stock symbol"
                )
            
            for symbol in data_sufficiency["insufficient_symbols"]:
                shortage = data_sufficiency["symbols_detail"][symbol].get("shortage_days", 0)
                warnings.append(
                    f"{symbol}: Insufficient data ({shortage} days short of minimum)"
                )
                recommendations.append(
                    f"Extend date range for {symbol} or reduce warmup_days requirement"
                )
        
        for symbol, signal_list in signals.items():
            if not signal_list:
                warnings.append(f"No signals for {symbol}")
            else:
                for sig in signal_list:
                    if sig.timestamp and start_date and sig.timestamp < start_date:
                        warnings.append(
                            f"Signal for {symbol} predates backtest start ({sig.timestamp.date()})"
                        )
        
        return {
            "is_valid": len(issues) == 0,
            "can_proceed": len([i for i in issues if "missing" in i.lower()]) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "data_sufficiency": data_sufficiency,
        }
    
    def adjust_dates_for_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Adjust start/end dates based on available data.
        
        Args:
            symbols: List of stock symbols
            start_date: Desired start date
            end_date: Desired end date
            
        Returns:
            Adjusted dates and data availability report
        """
        data = self.data_loader.load_data(symbols, start_date, end_date, "5y")
        
        if not data:
            return {
                "success": False,
                "error": "No data available for any symbols",
                "adjusted_start": None,
                "adjusted_end": None,
            }
        
        actual_starts = []
        actual_ends = []
        
        for symbol, df in data.items():
            if len(df) > 0:
                actual_starts.append(df.index[0])
                actual_ends.append(df.index[-1])
        
        if not actual_starts or not actual_ends:
            return {
                "success": False,
                "error": "Data exists but is empty",
                "adjusted_start": None,
                "adjusted_end": None,
            }
        
        actual_start = max(actual_starts)
        actual_end = min(actual_ends)
        
        adjusted_start = start_date if (start_date and start_date >= actual_start) else actual_start
        adjusted_end = end_date if (end_date and end_date <= actual_end) else actual_end
        
        if adjusted_start >= adjusted_end:
            adjusted_start = actual_start
            adjusted_end = actual_end
            warnings = ["Requested dates adjusted to common data range"]
        else:
            warnings = []
        
        if start_date and start_date < actual_start:
            warnings.append(f"Start date adjusted from {start_date.date()} to {actual_start.date()}")
        
        if end_date and end_date > actual_end:
            warnings.append(f"End date adjusted from {end_date.date()} to {actual_end.date()}")
        
        return {
            "success": True,
            "original_start": start_date,
            "original_end": end_date,
            "adjusted_start": adjusted_start,
            "adjusted_end": adjusted_end,
            "warnings": warnings,
            "common_data_range": {
                "start": actual_start.isoformat(),
                "end": actual_end.isoformat(),
            },
            "symbols_covered": list(data.keys()),
            "symbols_missing": [s for s in symbols if s not in data],
        }
    
    def handle_insufficient_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: str = "adjust",
        min_days: int = 200,
    ) -> Dict[str, Any]:
        """
        Handle insufficient data for backtest with various strategies.
        
        Args:
            symbols: List of stock symbols
            start_date: Desired start date
            end_date: Desired end date
            strategy: How to handle - "error", "adjust", "skip", "extend"
            min_days: Minimum trading days required
            
        Returns:
            Result with status and recommended action
        """
        data_check = self.check_data_sufficiency(symbols, start_date, end_date, min_days)
        
        result = {
            "strategy": strategy,
            "data_check": data_check,
            "action_taken": None,
            "modified_symbols": symbols.copy(),
            "can_run_backtest": True,
            "backtest_config": None,
        }
        
        if data_check["is_sufficient"]:
            result["action_taken"] = "no_action"
            result["backtest_config"] = {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
            }
            return result
        
        if strategy == "error":
            issues = []
            if data_check["missing_symbols"]:
                issues.append(f"Missing symbols: {', '.join(data_check['missing_symbols'])}")
            if data_check["insufficient_symbols"]:
                issues.append(
                    f"Insufficient data for: {', '.join(data_check['insufficient_symbols'])}"
                )
            result["can_run_backtest"] = False
            result["action_taken"] = "blocked"
            result["error"] = "; ".join(issues)
            return result
        
        elif strategy == "skip":
            valid_symbols = [
                s for s in symbols 
                if s not in data_check["missing_symbols"] 
                and s not in data_check["insufficient_symbols"]
            ]
            result["modified_symbols"] = valid_symbols
            result["action_taken"] = "skipped_invalid"
            result["skipped_symbols"] = (
                data_check["missing_symbols"] + data_check["insufficient_symbols"]
            )
            result["backtest_config"] = {
                "symbols": valid_symbols,
                "start_date": start_date,
                "end_date": end_date,
            }
            
            if not valid_symbols:
                result["can_run_backtest"] = False
                result["error"] = "All symbols have insufficient or missing data"
            
            return result
        
        elif strategy == "extend":
            data = self.data_loader.load_data(symbols, None, None, "10y")
            
            for symbol in data_check["insufficient_symbols"]:
                if symbol in data:
                    days_count = len(data[symbol])
                    if days_count >= min_days:
                        result["modified_symbols"] = symbols.copy()
                        result["action_taken"] = "extended_date_range"
                        result["backtest_config"] = {
                            "symbols": symbols,
                            "start_date": None,
                            "end_date": None,
                            "period": "10y",
                        }
                        return result
            
            result["can_run_backtest"] = False
            result["action_taken"] = "extension_insufficient"
            result["error"] = "Extending date range does not provide sufficient data"
            return result
        
        elif strategy == "adjust":
            date_adjustment = self.adjust_dates_for_data(symbols, start_date, end_date)
            
            if date_adjustment["success"]:
                result["modified_symbols"] = date_adjustment["symbols_covered"]
                result["action_taken"] = "adjusted_dates"
                result["date_adjustments"] = date_adjustment
                result["backtest_config"] = {
                    "symbols": date_adjustment["symbols_covered"],
                    "start_date": date_adjustment["adjusted_start"],
                    "end_date": date_adjustment["adjusted_end"],
                }
                
                recheck = self.check_data_sufficiency(
                    result["modified_symbols"],
                    date_adjustment["adjusted_start"],
                    date_adjustment["adjusted_end"],
                    min_days
                )
                
                if not recheck["is_sufficient"]:
                    result["warnings"] = [
                        "Some symbols still have insufficient data after adjustment"
                    ]
                
                return result
            
            result["can_run_backtest"] = False
            result["action_taken"] = "adjustment_failed"
            result["error"] = date_adjustment.get("error", "Failed to adjust dates")
            return result
        
        result["can_run_backtest"] = False
        result["action_taken"] = "unknown_strategy"
        result["error"] = f"Unknown strategy: {strategy}"
        return result
    
    def detect_data_gaps(
        self,
        data: Dict[str, pd.DataFrame],
        max_gap_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Detect gaps in price data for each symbol.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            max_gap_days: Maximum acceptable gap in trading days
            
        Returns:
            Report with detected gaps per symbol
        """
        import pandas as pd
        
        report = {
            "symbols_analyzed": len(data),
            "symbols_with_gaps": [],
            "symbols_without_gaps": [],
            "total_gaps": 0,
            "critical_gaps": 0,
            "gaps_detail": {},
        }
        
        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            df_index = pd.to_datetime(df.index)
            date_diffs = df_index.diff()
            
            gap_threshold = pd.Timedelta(days=max_gap_days + 1)
            
            gap_mask = date_diffs > gap_threshold
            gap_indices = df_index[gap_mask].tolist() if len(df_index) > 1 else []
            
            gaps = []
            critical_gaps = []
            
            for gap_date in gap_indices:
                if isinstance(gap_date, pd.Timestamp):
                    gap_idx = df_index.get_indexer([gap_date])[0]
                else:
                    gap_idx = df_index.get_loc(gap_date)
                prev_idx = max(0, gap_idx - 1)
                
                gap_start = df_index[prev_idx]
                gap_end = gap_date
                gap_days = (gap_end - gap_start).days
                
                gap_info = {
                    "gap_start": pd.Timestamp(gap_start).isoformat() if hasattr(gap_start, 'isoformat') else str(gap_start),
                    "gap_end": pd.Timestamp(gap_end).isoformat() if hasattr(gap_end, 'isoformat') else str(gap_end),
                    "gap_days": gap_days,
                    "is_critical": gap_days > max_gap_days * 2,
                    "prev_price": float(df.iloc[prev_idx]['Close']),
                    "next_price": float(df.iloc[gap_idx]['Close']),
                }
                
                if gap_info["is_critical"]:
                    critical_gaps.append(gap_info)
                
                gaps.append(gap_info)
            
            symbol_detail = {
                "total_gaps": len(gaps),
                "critical_gaps": len(critical_gaps),
                "gaps": gaps,
            }
            
            report["gaps_detail"][symbol] = symbol_detail
            
            if len(gaps) > 0:
                report["symbols_with_gaps"].append(symbol)
            else:
                report["symbols_without_gaps"].append(symbol)
            
            report["total_gaps"] += len(gaps)
            report["critical_gaps"] += len(critical_gaps)
        
        return report
    
    def fill_data_gaps(
        self,
        data: Dict[str, pd.DataFrame],
        method: str = "forward",
        max_fill_days: int = 3,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fill gaps in price data using specified method.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            method: Fill method - "forward", "backward", "interpolate", "ffill_then_bfill"
            max_fill_days: Maximum days to fill (gaps larger are left as-is)
            
        Returns:
            Data with gaps filled
        """
        import numpy as np
        
        filled_data = {}
        
        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            df_filled = df.copy()
            original_len = len(df_filled)
            
            df_index = pd.to_datetime(df_filled.index)
            date_diffs = df_index.diff()
            gap_threshold = pd.Timedelta(days=max_fill_days)
            
            gap_mask = date_diffs > gap_threshold
            
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if method == "forward":
                for col in price_columns:
                    if col in df_filled.columns:
                        df_filled[col] = df_filled[col].ffill()
            
            elif method == "backward":
                for col in price_columns:
                    if col in df_filled.columns:
                        df_filled[col] = df_filled[col].bfill()
            
            elif method == "ffill_then_bfill":
                for col in price_columns:
                    if col in df_filled.columns:
                        df_filled[col] = df_filled[col].ffill().bfill()
            
            elif method == "interpolate":
                for col in price_columns:
                    if col in df_filled.columns:
                        df_filled[col] = df_filled[col].interpolate(method='time')
                
                for col in price_columns:
                    if col in df_filled.columns:
                        df_filled[col] = df_filled[col].ffill().bfill()
            
            remaining_gaps = df_filled.isnull().sum().sum()
            original_gaps = df.isnull().sum().sum()
            gaps_filled = original_gaps - remaining_gaps
            
            if len(df_filled) > 0:
                df_filled = df_filled.dropna(subset=['Close'])
            
            filled_data[symbol] = df_filled
        
        return filled_data
    
    def handle_data_gaps(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: str = "interpolate",
        max_gap_days: int = 5,
        max_fill_days: int = 3,
    ) -> Dict[str, Any]:
        """
        Handle data gaps with various strategies.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            strategy: How to handle - "error", "skip", "fill", "warn"
            max_gap_days: Maximum acceptable gap before flagging as critical
            max_fill_days: Maximum days to fill
            
        Returns:
            Result with status and action taken
        """
        gap_report = self.detect_data_gaps(data, max_gap_days)
        
        result = {
            "strategy": strategy,
            "gap_report": gap_report,
            "data_modified": False,
            "modified_data": None,
            "can_proceed": True,
            "warnings": [],
            "errors": [],
        }
        
        if gap_report["total_gaps"] == 0:
            result["action_taken"] = "no_gaps_found"
            result["modified_data"] = data
            return result
        
        result["warnings"].append(
            f"Found {gap_report['total_gaps']} gap(s) in {len(gap_report['symbols_with_gaps'])} symbol(s)"
        )
        
        if gap_report["critical_gaps"] > 0:
            result["warnings"].append(
                f"Critical gaps (> {max_gap_days * 2} days) found: {gap_report['critical_gaps']}"
            )
        
        if strategy == "error":
            if gap_report["critical_gaps"] > 0:
                result["can_proceed"] = False
                result["action_taken"] = "blocked_critical_gaps"
                result["errors"].append(
                    f"Cannot proceed with {gap_report['critical_gaps']} critical data gaps"
                )
            else:
                result["action_taken"] = "gaps_acceptable"
                result["modified_data"] = data
            return result
        
        elif strategy == "skip":
            valid_symbols = gap_report["symbols_without_gaps"]
            
            if len(valid_symbols) < len(data):
                result["warnings"].append(
                    f"Skipped {len(data) - len(valid_symbols)} symbol(s) with gaps"
                )
            
            valid_data = {s: data[s] for s in valid_symbols if s in data}
            result["modified_data"] = valid_data
            result["symbols_skipped"] = list(set(data.keys()) - set(valid_symbols))
            result["action_taken"] = "skipped_gapped_symbols"
            
            if len(valid_data) == 0:
                result["can_proceed"] = False
                result["errors"].append("All symbols have data gaps")
            
            return result
        
        elif strategy in ["fill", "interpolate", "forward", "backward", "ffill_then_bfill"]:
            fill_method = strategy if strategy != "fill" else "interpolate"
            
            filled_data = self.fill_data_gaps(data, method=fill_method, max_fill_days=max_fill_days)
            
            result["modified_data"] = filled_data
            result["data_modified"] = True
            result["action_taken"] = f"filled_using_{fill_method}"
            result["filled_symbols"] = list(filled_data.keys())
            
            return result
        
        elif strategy == "warn":
            result["action_taken"] = "warnings_only"
            result["modified_data"] = data
            return result
        
        result["action_taken"] = "unknown_strategy"
        result["errors"].append(f"Unknown strategy: {strategy}")
        return result
    
    def validate_data_continuity(
        self,
        data: Dict[str, pd.DataFrame],
        signals: Dict[str, List[TradingSignal]],
        max_gap_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Validate data continuity and signal alignment.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            signals: Dictionary of symbol -> TradingSignals
            max_gap_days: Maximum acceptable gap
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        gap_report = self.detect_data_gaps(data, max_gap_days)
        
        for symbol, gap_detail in gap_report["gaps_detail"].items():
            for gap in gap_detail["gaps"]:
                warnings.append(
                    f"{symbol}: Data gap from {gap['gap_start'][:10]} to "
                    f"{gap['gap_end'][:10]} ({gap['gap_days']} days)"
                )
        
        for symbol, signal_list in signals.items():
            if symbol not in data:
                issues.append(f"Signal for {symbol} but no data available")
                continue
            
            df = data[symbol]
            df_dates = set(pd.to_datetime(df.index).date)
            
            for sig in signal_list:
                if sig.timestamp is None:
                    continue
                
                sig_date = pd.Timestamp(sig.timestamp).date()
                
                if sig_date not in df_dates:
                    warnings.append(
                        f"Signal for {symbol} on {sig_date} falls in data gap or is outside range"
                    )
                
                if sig.signal.value in ["BUY", "STRONG_BUY"]:
                    if df is not None and len(df) > 0:
                        recent_closes = df['Close'].tail(5)
                        if len(recent_closes) > 0 and recent_closes.iloc[-1] == 0:
                            issues.append(f"{symbol}: Latest price is 0 - data may be stale")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "gap_summary": {
                "total_gaps": gap_report["total_gaps"],
                "critical_gaps": gap_report["critical_gaps"],
                "affected_symbols": gap_report["symbols_with_gaps"],
            },
        }
    
    def detect_zero_volume_days(
        self,
        data: Dict[str, pd.DataFrame],
        min_volume_threshold: float = 1000,
    ) -> Dict[str, Any]:
        """
        Detect days with zero or very low volume.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            min_volume_threshold: Minimum acceptable volume
            
        Returns:
            Report with zero volume days per symbol
        """
        report = {
            "symbols_analyzed": len(data),
            "symbols_with_zero_volume": [],
            "symbols_without_issues": [],
            "total_zero_volume_days": 0,
            "total_low_volume_days": 0,
            "detail": {},
        }
        
        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            if 'Volume' not in df.columns:
                continue
            
            zero_mask = df['Volume'] == 0
            low_mask = (df['Volume'] > 0) & (df['Volume'] < min_volume_threshold)
            
            zero_indices = df.index[zero_mask].tolist()
            low_indices = df.index[low_mask].tolist()
            
            detail = {
                "zero_volume_days": len(zero_indices),
                "low_volume_days": len(low_indices),
                "zero_volume_dates": [str(d)[:10] for d in zero_indices],
                "low_volume_dates": [str(d)[:10] for d in low_indices[:10]],
                "avg_volume": float(df['Volume'].mean()),
                "median_volume": float(df['Volume'].median()),
            }
            
            report["detail"][symbol] = detail
            
            if len(zero_indices) > 0:
                report["symbols_with_zero_volume"].append(symbol)
            else:
                report["symbols_without_issues"].append(symbol)
            
            report["total_zero_volume_days"] += len(zero_indices)
            report["total_low_volume_days"] += len(low_indices)
        
        return report
    
    def handle_zero_volume_days(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: str = "filter",
        min_volume_threshold: float = 1000,
    ) -> Dict[str, Any]:
        """
        Handle days with zero or very low volume.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            strategy: How to handle - "filter", "skip", "warn", "interpolate"
            min_volume_threshold: Minimum acceptable volume
            
        Returns:
            Result with status and action taken
        """
        detection = self.detect_zero_volume_days(data, min_volume_threshold)
        
        result = {
            "strategy": strategy,
            "detection": detection,
            "data_modified": False,
            "modified_data": None,
            "can_proceed": True,
            "warnings": [],
            "errors": [],
        }
        
        if detection["total_zero_volume_days"] == 0:
            result["action_taken"] = "no_issues_found"
            result["modified_data"] = data
            return result
        
        result["warnings"].append(
            f"Found {detection['total_zero_volume_days']} zero-volume day(s) "
            f"in {len(detection['symbols_with_zero_volume'])} symbol(s)"
        )
        
        if detection["total_low_volume_days"] > 0:
            result["warnings"].append(
                f"Found {detection['total_low_volume_days']} low-volume day(s)"
            )
        
        if strategy == "filter":
            filtered_data = {}
            
            for symbol, df in data.items():
                if df is None or len(df) == 0:
                    continue
                
                if 'Volume' not in df.columns:
                    filtered_data[symbol] = df
                    continue
                
                mask = df['Volume'] >= min_volume_threshold
                df_filtered = df[mask].copy()
                
                if len(df_filtered) < len(df):
                    result["warnings"].append(
                        f"{symbol}: Filtered {len(df) - len(df_filtered)} low/zero volume days"
                    )
                
                filtered_data[symbol] = df_filtered
            
            result["modified_data"] = filtered_data
            result["data_modified"] = True
            result["action_taken"] = "filtered_zero_volume_days"
            
            return result
        
        elif strategy == "skip":
            valid_symbols = detection["symbols_without_issues"]
            
            if len(valid_symbols) < len(data):
                result["warnings"].append(
                    f"Skipped {len(data) - len(valid_symbols)} symbol(s) with zero volume days"
                )
            
            valid_data = {s: data[s] for s in valid_symbols if s in data}
            result["modified_data"] = valid_data
            result["symbols_skipped"] = list(set(data.keys()) - set(valid_symbols))
            result["action_taken"] = "skipped_zero_volume_symbols"
            
            if len(valid_data) == 0:
                result["can_proceed"] = False
                result["errors"].append("All symbols have zero volume days")
            
            return result
        
        elif strategy == "interpolate":
            interpolated_data = {}
            
            for symbol, df in data.items():
                if df is None or len(df) == 0:
                    continue
                
                df_interp = df.copy()
                
                if 'Volume' in df_interp.columns:
                    df_interp['Volume'] = df_interp['Volume'].astype(float).replace(0, float('nan'))
                    df_interp['Volume'] = df_interp['Volume'].interpolate(method='linear')
                    df_interp['Volume'] = df_interp['Volume'].fillna(df_interp['Volume'].median())
                
                interpolated_data[symbol] = df_interp
            
            result["modified_data"] = interpolated_data
            result["data_modified"] = True
            result["action_taken"] = "interpolated_volume"
            
            return result
        
        elif strategy == "warn":
            result["action_taken"] = "warnings_only"
            result["modified_data"] = data
            return result
        
        result["action_taken"] = "unknown_strategy"
        result["errors"].append(f"Unknown strategy: {strategy}")
        return result
    
    def validate_volume_quality(
        self,
        data: Dict[str, pd.DataFrame],
        min_volume_threshold: float = 1000,
        min_avg_volume: float = 10000,
    ) -> Dict[str, Any]:
        """
        Validate overall volume quality for backtesting.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            min_volume_threshold: Minimum per-day volume
            min_avg_volume: Minimum average volume
            
        Returns:
            Quality validation report
        """
        issues = []
        warnings = []
        
        detection = self.detect_zero_volume_days(data, min_volume_threshold)
        
        if detection["total_zero_volume_days"] > 0:
            warnings.append(
                f"{detection['total_zero_volume_days']} zero-volume day(s) found"
            )
        
        for symbol, detail in detection["detail"].items():
            if detail["zero_volume_days"] > len(data[symbol]) * 0.05:
                warnings.append(
                    f"{symbol}: {detail['zero_volume_days']} zero-volume days (>5% of data)"
                )
            
            if detail["avg_volume"] < min_avg_volume:
                warnings.append(
                    f"{symbol}: Low average volume ({detail['avg_volume']:.0f} < {min_avg_volume})"
                )
            
            if detail["median_volume"] == 0:
                issues.append(
                    f"{symbol}: Median volume is 0 - data may be unreliable"
                )
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "volume_summary": {
                "symbols_analyzed": detection["symbols_analyzed"],
                "symbols_with_issues": len(detection["symbols_with_zero_volume"]),
                "total_zero_volume_days": detection["total_zero_volume_days"],
                "total_low_volume_days": detection["total_low_volume_days"],
            },
        }
    
    def detect_delist_or_suspension(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        min_data_points: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect stocks that may have been delisted or suspended.
        
        Args:
            symbols: List of stock symbols
            data: Dictionary of symbol -> DataFrame
            min_data_points: Minimum expected data points for the period
            
        Returns:
            Report with potentially delisted/suspended stocks
        """
        report = {
            "symbols_checked": len(symbols),
            "potentially_delisted": [],
            "potentially_suspended": [],
            "active_symbols": [],
            "detail": {},
        }
        
        for symbol in symbols:
            df = data.get(symbol)
            
            detail = {
                "status": "active",
                "data_points": 0,
                "date_range": None,
                "issues": [],
            }
            
            if df is None or len(df) == 0:
                detail["status"] = "delisted"
                detail["issues"].append("No data available")
                report["potentially_delisted"].append(symbol)
                report["detail"][symbol] = detail
                continue
            
            data_points = len(df)
            detail["data_points"] = data_points
            
            if data_points < min_data_points:
                detail["issues"].append(f"Only {data_points} data points (expected ~{min_data_points})")
            
            if data_points > 0:
                start_date = df.index[0]
                end_date = df.index[-1]
                detail["date_range"] = {
                    "start": str(start_date)[:10],
                    "end": str(end_date)[:10],
                    "days": (end_date - start_date).days,
                }
                
                expected_days = (end_date - start_date).days
                if expected_days > 0:
                    trading_day_ratio = data_points / expected_days
                    if trading_day_ratio < 0.5:
                        detail["issues"].append(
                            f"Low trading day ratio: {trading_day_ratio:.1%} "
                            f"(expected ~65% for Indian markets)"
                        )
                        detail["status"] = "suspicious"
                        report["potentially_suspended"].append(symbol)
            
            missing_columns = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c not in df.columns]
            if missing_columns:
                detail["issues"].append(f"Missing columns: {missing_columns}")
            
            recent_volume = df['Volume'].tail(20).mean() if 'Volume' in df.columns else 0
            if recent_volume == 0 and len(df) > 0:
                detail["issues"].append("Recent volume is 0 - may be suspended")
                detail["status"] = "suspended"
                report["potentially_suspended"].append(symbol)
            
            if detail["status"] == "active":
                report["active_symbols"].append(symbol)
            
            report["detail"][symbol] = detail
        
        return report
    
    def handle_delist_or_suspend(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        strategy: str = "skip",
        min_data_points: int = 100,
    ) -> Dict[str, Any]:
        """
        Handle stocks that may be delisted or suspended.
        
        Args:
            symbols: List of stock symbols
            data: Dictionary of symbol -> DataFrame
            strategy: How to handle - "error", "skip", "warn", "partial"
            min_data_points: Minimum expected data points
            
        Returns:
            Result with status and action taken
        """
        detection = self.detect_delist_or_suspension(symbols, data, min_data_points)
        
        result = {
            "strategy": strategy,
            "detection": detection,
            "data_modified": False,
            "modified_data": {},
            "can_proceed": True,
            "warnings": [],
            "errors": [],
        }
        
        if len(detection["potentially_delisted"]) == 0 and len(detection["potentially_suspended"]) == 0:
            result["action_taken"] = "no_issues_found"
            result["modified_data"] = data
            return result
        
        if detection["potentially_delisted"]:
            result["warnings"].append(
                f"Potentially delisted: {', '.join(detection['potentially_delisted'])}"
            )
        
        if detection["potentially_suspended"]:
            result["warnings"].append(
                f"Potentially suspended: {', '.join(detection['potentially_suspended'])}"
            )
        
        if strategy == "error":
            if detection["potentially_delisted"]:
                result["can_proceed"] = False
                result["action_taken"] = "blocked_delisted"
                result["errors"].append(
                    f"Cannot proceed with delisted symbols: {', '.join(detection['potentially_delisted'])}"
                )
            elif detection["potentially_suspended"]:
                result["can_proceed"] = False
                result["action_taken"] = "blocked_suspended"
                result["errors"].append(
                    f"Cannot proceed with suspended symbols: {', '.join(detection['potentially_suspended'])}"
                )
            return result
        
        elif strategy == "skip":
            valid_symbols = detection["active_symbols"]
            
            skipped = list(set(symbols) - set(valid_symbols))
            if skipped:
                result["warnings"].append(f"Skipped {len(skipped)} symbol(s) due to delist/suspension")
            
            result["modified_data"] = {s: data[s] for s in valid_symbols if s in data}
            result["symbols_skipped"] = skipped
            result["action_taken"] = "skipped_delisted_suspended"
            
            if len(valid_symbols) == 0:
                result["can_proceed"] = False
                result["errors"].append("All symbols are delisted or suspended")
            
            return result
        
        elif strategy == "partial":
            partial_data = {}
            
            for symbol in symbols:
                df = data.get(symbol)
                if df is None:
                    continue
                
                detail = detection["detail"].get(symbol, {})
                
                if detail.get("status") == "active":
                    partial_data[symbol] = df
                elif detail.get("status") == "suspicious":
                    cutoff_date = detail.get("date_range", {}).get("end")
                    if cutoff_date:
                        df_trimmed = df[df.index <= pd.Timestamp(cutoff_date)]
                        if len(df_trimmed) >= min_data_points // 2:
                            partial_data[symbol] = df_trimmed
                            result["warnings"].append(
                                f"{symbol}: Using partial data up to {str(cutoff_date)[:10]}"
                            )
            
            result["modified_data"] = partial_data
            result["data_modified"] = True
            result["action_taken"] = "used_partial_data"
            
            if len(partial_data) == 0:
                result["can_proceed"] = False
                result["errors"].append("No usable partial data available")
            
            return result
        
        elif strategy == "warn":
            result["action_taken"] = "warnings_only"
            result["modified_data"] = data
            return result
        
        result["action_taken"] = "unknown_strategy"
        result["errors"].append(f"Unknown strategy: {strategy}")
        return result
    
    def validate_symbols_for_backtest(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        min_data_points: int = 100,
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of symbols for backtesting.
        
        Args:
            symbols: List of stock symbols
            data: Dictionary of symbol -> DataFrame
            min_data_points: Minimum required data points
            
        Returns:
            Validation report with all issues
        """
        issues = []
        warnings = []
        
        detection = self.detect_delist_or_suspension(symbols, data, min_data_points)
        
        for symbol, detail in detection["detail"].items():
            if detail["status"] == "delisted":
                issues.append(f"{symbol}: Delisted or no data available")
            elif detail["status"] == "suspended":
                warnings.append(f"{symbol}: Recently suspended (zero volume)")
            
            for issue in detail.get("issues", []):
                if detail["status"] == "suspicious":
                    warnings.append(f"{symbol}: {issue}")
        
        valid_count = len(detection["active_symbols"])
        total_count = len(symbols)
        
        if valid_count < total_count:
            warnings.append(
                f"Only {valid_count}/{total_count} symbols are fully active"
            )
        
        if valid_count == 0:
            issues.append("No valid symbols available for backtest")
        
        return {
            "is_valid": len(issues) == 0,
            "can_proceed": valid_count > 0,
            "issues": issues,
            "warnings": warnings,
            "summary": {
                "total_symbols": total_count,
                "active_symbols": valid_count,
                "potentially_delisted": len(detection["potentially_delisted"]),
                "potentially_suspended": len(detection["potentially_suspended"]),
            },
            "detail": detection["detail"],
        }
    
    def detect_circuit_limit_hits(
        self,
        data: Dict[str, pd.DataFrame],
        upper_limit_pct: float = 20.0,
        lower_limit_pct: float = 20.0,
        lookback_days: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect potential circuit limit hits (upper/lower freeze).
        
        Args:
            data: Dictionary of symbol -> DataFrame
            upper_limit_pct: Upper circuit limit percentage (default 20% for NSE/BSE)
            lower_limit_pct: Lower circuit limit percentage
            lookback_days: Number of days to check
            
        Returns:
            Report with potential circuit limit hits
        """
        report = {
            "symbols_analyzed": len(data),
            "symbols_with_circuit_hits": [],
            "symbols_without_issues": [],
            "total_upper_hits": 0,
            "total_lower_hits": 0,
            "detail": {},
        }
        
        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            if 'Close' not in df.columns or 'Open' not in df.columns:
                continue
            
            df_recent = df.tail(lookback_days).copy()
            
            df_recent['Prev_Close'] = df_recent['Close'].shift(1)
            df_recent['Change_Pct'] = ((df_recent['Close'] - df_recent['Prev_Close']) / df_recent['Prev_Close']) * 100
            
            upper_hit_mask = df_recent['Change_Pct'] >= upper_limit_pct
            lower_hit_mask = df_recent['Change_Pct'] <= -lower_limit_pct
            
            upper_hit_dates = df_recent.index[upper_hit_mask].tolist()
            lower_hit_dates = df_recent.index[lower_hit_mask].tolist()
            
            upper_hits = []
            lower_hits = []
            
            for hit_date in upper_hit_dates:
                row = df_recent.loc[hit_date]
                upper_hits.append({
                    "date": str(hit_date)[:10],
                    "change_pct": float(row['Change_Pct']),
                    "close": float(row['Close']),
                    "prev_close": float(row['Prev_Close']),
                })
            
            for hit_date in lower_hit_dates:
                row = df_recent.loc[hit_date]
                lower_hits.append({
                    "date": str(hit_date)[:10],
                    "change_pct": float(row['Change_Pct']),
                    "close": float(row['Close']),
                    "prev_close": float(row['Prev_Close']),
                })
            
            detail = {
                "upper_circuit_hits": len(upper_hits),
                "lower_circuit_hits": len(lower_hits),
                "upper_hit_dates": upper_hits,
                "lower_hit_dates": lower_hits,
                "has_circuit_hits": len(upper_hits) > 0 or len(lower_hits) > 0,
            }
            
            report["detail"][symbol] = detail
            
            if detail["has_circuit_hits"]:
                report["symbols_with_circuit_hits"].append(symbol)
                report["total_upper_hits"] += len(upper_hits)
                report["total_lower_hits"] += len(lower_hits)
            else:
                report["symbols_without_issues"].append(symbol)
        
        return report
    
    def handle_circuit_limit_hits(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: str = "warn",
        upper_limit_pct: float = 20.0,
        lower_limit_pct: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Handle potential circuit limit hits.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            strategy: How to handle - "error", "skip", "warn", "adjust"
            upper_limit_pct: Upper circuit limit percentage
            lower_limit_pct: Lower circuit limit percentage
            
        Returns:
            Result with status and action taken
        """
        detection = self.detect_circuit_limit_hits(
            data, upper_limit_pct, lower_limit_pct
        )
        
        result = {
            "strategy": strategy,
            "detection": detection,
            "data_modified": False,
            "modified_data": None,
            "can_proceed": True,
            "warnings": [],
            "errors": [],
        }
        
        if detection["total_upper_hits"] == 0 and detection["total_lower_hits"] == 0:
            result["action_taken"] = "no_circuit_hits"
            result["modified_data"] = data
            return result
        
        result["warnings"].append(
            f"Found {detection['total_upper_hits']} upper circuit hit(s) "
            f"and {detection['total_lower_hits']} lower circuit hit(s) "
            f"in {len(detection['symbols_with_circuit_hits'])} symbol(s)"
        )
        
        if strategy == "error":
            if detection["total_upper_hits"] > 0:
                result["can_proceed"] = False
                result["action_taken"] = "blocked_upper_circuit_hits"
                result["errors"].append(
                    f"Cannot backtest: Upper circuit hits detected in "
                    f"{detection['symbols_with_circuit_hits']}"
                )
            else:
                result["action_taken"] = "lower_circuit_hits_acceptable"
                result["modified_data"] = data
            return result
        
        elif strategy == "skip":
            valid_symbols = detection["symbols_without_issues"]
            
            if len(valid_symbols) < len(data):
                result["warnings"].append(
                    f"Skipped {len(data) - len(valid_symbols)} symbol(s) with circuit hits"
                )
            
            result["modified_data"] = {s: data[s] for s in valid_symbols if s in data}
            result["symbols_skipped"] = list(set(data.keys()) - set(valid_symbols))
            result["action_taken"] = "skipped_circuit_hit_symbols"
            
            if len(valid_symbols) == 0:
                result["can_proceed"] = False
                result["errors"].append("All symbols have circuit limit hits")
            
            return result
        
        elif strategy == "adjust":
            adjusted_data = {}
            
            for symbol, df in data.items():
                if df is None or len(df) == 0:
                    continue
                
                df_adjusted = df.copy()
                detail = detection["detail"].get(symbol, {})
                
                for hit in detail.get("upper_hit_dates", []) + detail.get("lower_hit_dates", []):
                    hit_date = hit["date"]
                    try:
                        hit_idx = pd.Timestamp(hit_date)
                        if hit_idx in df_adjusted.index:
                            pct_change = hit["change_pct"] / 100
                            adjusted_price = hit["prev_close"] * (1 + pct_change * 0.5)
                            
                            df_adjusted.loc[hit_idx, 'Close'] = adjusted_price
                            df_adjusted.loc[hit_idx, 'High'] = max(df_adjusted.loc[hit_idx, 'High'], adjusted_price)
                            df_adjusted.loc[hit_idx, 'Low'] = min(df_adjusted.loc[hit_idx, 'Low'], adjusted_price)
                    except (KeyError, ValueError):
                        continue
                
                adjusted_data[symbol] = df_adjusted
            
            result["modified_data"] = adjusted_data
            result["data_modified"] = True
            result["action_taken"] = "adjusted_circuit_hit_prices"
            
            return result
        
        elif strategy == "warn":
            result["action_taken"] = "warnings_only"
            result["modified_data"] = data
            return result
        
        result["action_taken"] = "unknown_strategy"
        result["errors"].append(f"Unknown strategy: {strategy}")
        return result
    
    def validate_market_conditions(
        self,
        data: Dict[str, pd.DataFrame],
        max_circuit_hits_per_symbol: int = 2,
        max_total_circuit_hits: int = 5,
    ) -> Dict[str, Any]:
        """
        Validate overall market conditions for backtesting.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            max_circuit_hits_per_symbol: Maximum acceptable circuit hits per symbol
            max_total_circuit_hits: Maximum acceptable total circuit hits
            
        Returns:
            Market conditions validation report
        """
        issues = []
        warnings = []
        
        circuit_detection = self.detect_circuit_limit_hits(data)
        
        for symbol, detail in circuit_detection["detail"].items():
            total_hits = detail["upper_circuit_hits"] + detail["lower_circuit_hits"]
            if total_hits > max_circuit_hits_per_symbol:
                warnings.append(
                    f"{symbol}: {total_hits} circuit hits (max: {max_circuit_hits_per_symbol})"
                )
        
        if circuit_detection["total_upper_hits"] > max_total_circuit_hits:
            issues.append(
                f"Too many upper circuit hits: {circuit_detection['total_upper_hits']}"
            )
        
        if circuit_detection["total_lower_hits"] > max_total_circuit_hits:
            warnings.append(
                f"Many lower circuit hits: {circuit_detection['total_lower_hits']}"
            )
        
        market_status = "NORMAL"
        if circuit_detection["total_upper_hits"] + circuit_detection["total_lower_hits"] > max_total_circuit_hits:
            market_status = "VOLATILE"
        if circuit_detection["total_upper_hits"] > max_total_circuit_hits:
            market_status = "EXTREME_UPPER"
        if circuit_detection["total_lower_hits"] > max_total_circuit_hits:
            market_status = "EXTREME_LOWER"
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "market_status": market_status,
            "circuit_summary": {
                "symbols_analyzed": circuit_detection["symbols_analyzed"],
                "symbols_with_hits": len(circuit_detection["symbols_with_circuit_hits"]),
                "total_upper_hits": circuit_detection["total_upper_hits"],
                "total_lower_hits": circuit_detection["total_lower_hits"],
            },
        }
    
    def get_nse_holidays(self, year: int) -> List[Dict[str, Any]]:
        """
        Get list of NSE (India) market holidays for a given year.
        
        Args:
            year: Year to get holidays for
            
        Returns:
            List of dictionaries with holiday info
        """
        all_holidays = {
            2024: [
                {"date": "2024-01-26", "name": "Republic Day"},
                {"date": "2024-03-25", "name": "Holi"},
                {"date": "2024-03-29", "name": "Good Friday"},
                {"date": "2024-04-11", "name": "Id-ul-Fitr (Ramzan Eid)"},
                {"date": "2024-04-17", "name": "Mahavir Jayanti"},
                {"date": "2024-05-01", "name": "Maharashtra Day"},
                {"date": "2024-08-15", "name": "Independence Day"},
                {"date": "2024-10-02", "name": "Gandhi Jayanti"},
                {"date": "2024-11-01", "name": "Karnataka Rajyotsava"},
                {"date": "2024-11-15", "name": "Bhau Bij"},
                {"date": "2024-12-25", "name": "Christmas Day"},
            ],
            2025: [
                {"date": "2025-01-26", "name": "Republic Day"},
                {"date": "2025-03-14", "name": "Holi"},
                {"date": "2025-04-18", "name": "Good Friday"},
                {"date": "2025-03-31", "name": "Id-ul-Fitr (Ramzan Eid)"},
                {"date": "2025-04-10", "name": "Mahavir Jayanti"},
                {"date": "2025-05-01", "name": "Maharashtra Day"},
                {"date": "2025-08-15", "name": "Independence Day"},
                {"date": "2025-10-02", "name": "Gandhi Jayanti"},
                {"date": "2025-11-03", "name": "Diwali (Balipratipada)"},
                {"date": "2025-11-05", "name": "Bhau Bij"},
                {"date": "2025-12-25", "name": "Christmas Day"},
            ],
            2026: [
                {"date": "2026-01-26", "name": "Republic Day"},
                {"date": "2026-02-03", "name": "Vasant Panchami"},
                {"date": "2026-03-06", "name": "Mahashivratri"},
                {"date": "2026-04-03", "name": "Good Friday"},
                {"date": "2026-04-09", "name": "Id-ul-Fitr (Ramzan Eid)"},
                {"date": "2026-04-13", "name": "Mahavir Jayanti"},
                {"date": "2026-05-01", "name": "Maharashtra Day"},
                {"date": "2026-08-15", "name": "Independence Day"},
                {"date": "2026-10-02", "name": "Gandhi Jayanti"},
                {"date": "2026-10-20", "name": "Diwali (Balipratipada)"},
                {"date": "2026-11-05", "name": "Bhau Bij"},
                {"date": "2026-12-25", "name": "Christmas Day"},
            ],
            2027: [
                {"date": "2027-01-26", "name": "Republic Day"},
                {"date": "2027-02-22", "name": "Vasant Panchami"},
                {"date": "2027-03-22", "name": "Holi"},
                {"date": "2027-03-26", "name": "Good Friday"},
                {"date": "2027-04-10", "name": "Id-ul-Fitr (Ramzan Eid)"},
                {"date": "2027-05-01", "name": "Maharashtra Day"},
                {"date": "2027-08-15", "name": "Independence Day"},
                {"date": "2027-10-02", "name": "Gandhi Jayanti"},
                {"date": "2027-11-09", "name": "Diwali (Balipratipada)"},
                {"date": "2027-11-05", "name": "Bhau Bij"},
                {"date": "2027-12-25", "name": "Christmas Day"},
            ],
        }
        
        if year in all_holidays:
            return all_holidays[year]
        
        return []
    
    def detect_weekends_and_holidays(
        self,
        data: Dict[str, pd.DataFrame],
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Detect weekends and holidays in the data.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            years: Years to check for holidays (auto-detected if None)
            
        Returns:
            Detection report with weekends and holidays found
        """
        report = {
            "symbols_analyzed": 0,
            "total_weekdays": 0,
            "total_weekend_days": 0,
            "total_holidays": 0,
            "trading_days": 0,
            "detail": {},
            "weekend_dates": [],
            "holiday_dates": [],
        }
        
        if not data:
            return report
        
        if years is None:
            all_dates = set()
            for df in data.values():
                if df is not None and len(df) > 0:
                    all_dates.update(df.index)
            if all_dates:
                years = list(set(d.year for d in all_dates))
            else:
                years = [datetime.now().year]
        
        all_holidays = set()
        for year in years:
            for holiday in self.get_nse_holidays(year):
                all_holidays.add(pd.Timestamp(holiday["date"]))
        
        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            report["symbols_analyzed"] += 1
            
            weekend_mask = df.index.dayofweek.isin([5, 6])
            weekend_dates = df.index[weekend_mask].tolist()
            
            holiday_mask = df.index.isin(all_holidays)
            holiday_dates = df.index[holiday_mask].tolist()
            
            weekday_mask = ~weekend_mask & ~holiday_mask
            weekday_dates = df.index[weekday_mask].tolist()
            
            report["detail"][symbol] = {
                "total_days": len(df),
                "weekend_dates": [str(d)[:10] for d in weekend_dates],
                "holiday_dates": [str(d)[:10] for d in holiday_dates],
                "weekday_dates": [str(d)[:10] for d in weekday_dates],
                "weekend_count": len(weekend_dates),
                "holiday_count": len(holiday_dates),
                "weekday_count": len(weekday_dates),
            }
            
            report["total_weekend_days"] += len(weekend_dates)
            report["total_holidays"] += len(holiday_dates)
            report["total_weekdays"] += len(weekday_dates)
            report["weekend_dates"].extend([str(d)[:10] for d in weekend_dates])
            report["holiday_dates"].extend([str(d)[:10] for d in holiday_dates])
        
        report["trading_days"] = report["total_weekdays"]
        return report
    
    def handle_weekends_and_holidays(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: str = "warn",
    ) -> Dict[str, Any]:
        """
        Handle weekends and holidays in the data.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            strategy: How to handle - "error", "skip", "warn"
            
        Returns:
            Result with status and modified data
        """
        detection = self.detect_weekends_and_holidays(data)
        
        result = {
            "strategy": strategy,
            "detection": detection,
            "data_modified": False,
            "modified_data": None,
            "can_proceed": True,
            "warnings": [],
            "errors": [],
            "days_removed": 0,
        }
        
        has_issues = detection["total_weekend_days"] > 0 or detection["total_holidays"] > 0
        
        if not has_issues:
            result["action_taken"] = "no_weekends_or_holidays"
            result["modified_data"] = data
            return result
        
        if detection["total_weekend_days"] > 0:
            result["warnings"].append(
                f"Found {detection['total_weekend_days']} weekend day(s) in data"
            )
        
        if detection["total_holidays"] > 0:
            result["warnings"].append(
                f"Found {detection['total_holidays']} holiday(s) in data"
            )
        
        if strategy == "error":
            result["can_proceed"] = False
            result["action_taken"] = "blocked_weekends_holidays"
            result["errors"].append(
                f"Cannot backtest: Found {detection['total_weekend_days']} weekend(s) "
                f"and {detection['total_holidays']} holiday(s)"
            )
            return result
        
        elif strategy == "skip" or strategy == "filter":
            filtered_data = {}
            total_removed = 0
            
            for symbol, df in data.items():
                if df is None or len(df) == 0:
                    continue
                
                detail = detection["detail"].get(symbol, {})
                weekend_dates_str = set(detail.get("weekend_dates", []))
                holiday_dates_str = set(detail.get("holiday_dates", []))
                remove_dates = weekend_dates_str.union(holiday_dates_str)
                
                remove_indices = [
                    idx for idx in df.index
                    if str(idx)[:10] in remove_dates
                ]
                
                df_filtered = df.drop(remove_indices)
                total_removed += len(remove_indices)
                filtered_data[symbol] = df_filtered
            
            result["modified_data"] = filtered_data
            result["data_modified"] = True
            result["days_removed"] = total_removed
            result["action_taken"] = "removed_weekends_holidays"
            return result
        
        elif strategy == "warn":
            result["action_taken"] = "warnings_only"
            result["modified_data"] = data
            return result
        
        result["action_taken"] = "unknown_strategy"
        result["errors"].append(f"Unknown strategy: {strategy}")
        return result
    
    def validate_trading_calendar(
        self,
        data: Dict[str, pd.DataFrame],
        max_consecutive_holidays: int = 5,
    ) -> Dict[str, Any]:
        """
        Validate trading calendar for backtesting.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            max_consecutive_holidays: Maximum acceptable consecutive non-trading days
            
        Returns:
            Trading calendar validation report
        """
        issues = []
        warnings = []
        
        detection = self.detect_weekends_and_holidays(data)
        
        all_dates = set()
        for df in data.values():
            if df is not None and len(df) > 0:
                all_dates.update(df.index)
        
        if not all_dates:
            issues.append("No trading dates found in data")
            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "market_status": "UNKNOWN",
                "calendar_summary": {
                    "trading_days": 0,
                    "non_trading_days": 0,
                    "total_dates": 0,
                },
            }
        
        sorted_dates = sorted(all_dates)
        min_date = min(sorted_dates)
        max_date = max(sorted_dates)
        
        expected_trading_days = len([
            d for d in pd.date_range(min_date, max_date)
            if d.dayofweek < 5
        ])
        
        actual_trading_days = detection["trading_days"]
        non_trading_days = detection["total_weekend_days"] + detection["total_holidays"]
        
        if non_trading_days > expected_trading_days * 0.3:
            warnings.append(
                f"High ratio of non-trading days: {non_trading_days} of "
                f"{expected_trading_days} expected ({non_trading_days/expected_trading_days*100:.1f}%)"
            )
        
        consecutive_count = 0
        max_consecutive = 0
        last_was_non_trading = False
        
        for d in pd.date_range(min_date, max_date):
            is_non_trading = d.dayofweek >= 5 or d in {
                pd.Timestamp(h["date"]) 
                for year in set(d.year for d in all_dates)
                for h in self.get_nse_holidays(d.year)
            }
            
            if is_non_trading:
                if last_was_non_trading:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                last_was_non_trading = True
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                last_was_non_trading = False
        
        if max_consecutive > max_consecutive_holidays:
            warnings.append(
                f"Found {max_consecutive} consecutive non-trading days "
                f"(max allowed: {max_consecutive_holidays})"
            )
        
        market_status = "NORMAL"
        if max_consecutive > max_consecutive_holidays:
            market_status = "IRREGULAR"
        if non_trading_days > expected_trading_days * 0.5:
            market_status = "SPARSE"
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "market_status": market_status,
            "calendar_summary": {
                "trading_days": actual_trading_days,
                "non_trading_days": non_trading_days,
                "total_dates": len(sorted_dates),
                "date_range": {
                    "start": str(min_date)[:10],
                    "end": str(max_date)[:10],
                },
                "consecutive_non_trading_days": max_consecutive,
            },
        }
    
    def detect_concurrent_order_conflicts(
        self,
        signals: List[TradingSignal],
    ) -> Dict[str, Any]:
        """
        Detect concurrent order conflicts in a list of trading signals.
        
        Args:
            signals: List of TradingSignal objects
            
        Returns:
            Detection report with conflicts found
        """
        report = {
            "total_signals": len(signals),
            "conflicts_found": 0,
            "conflicting_symbols": [],
            "detail": {},
            "resolution_actions": [],
        }
        
        if not signals:
            return report
        
        symbol_signals: Dict[str, List[TradingSignal]] = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        for symbol, sig_list in symbol_signals.items():
            if len(sig_list) < 2:
                continue
            
            conflicts = []
            
            signals_by_direction: Dict[str, List[TradingSignal]] = {}
            for s in sig_list:
                direction = self.processor.signal_to_direction(s.signal).value
                if direction not in signals_by_direction:
                    signals_by_direction[direction] = []
                signals_by_direction[direction].append(s)
            
            has_buy = "LONG" in signals_by_direction
            has_sell = "SHORT" in signals_by_direction
            has_exit = "EXIT" in signals_by_direction
            has_hold = "HOLD" in signals_by_direction
            
            has_original_buy = any(s.signal in [Signal.BUY, Signal.STRONG_BUY] for s in sig_list)
            has_original_sell = any(s.signal in [Signal.SELL, Signal.STRONG_SELL] for s in sig_list)
            
            conflict_type = None
            if (has_buy and has_sell) or (has_original_buy and has_original_sell):
                conflict_type = "opposite_signals"
            elif has_exit and has_buy:
                conflict_type = "exit_with_entry"
            elif has_hold and has_buy:
                conflict_type = "hold_with_entry"
            elif has_hold and has_sell:
                conflict_type = "hold_with_short"
            
            if conflict_type:
                report["conflicts_found"] += 1
                report["conflicting_symbols"].append(symbol)
                
                report["detail"][symbol] = {
                    "conflict_type": conflict_type,
                    "signals": [
                        {
                            "signal": s.signal.value,
                            "conviction": s.conviction,
                            "direction": self.processor.signal_to_direction(s.signal).value,
                        }
                        for s in sig_list
                    ],
                    "signal_count": len(sig_list),
                }
                
                if conflict_type == "opposite_signals":
                    report["resolution_actions"].append(
                        f"{symbol}: Opposite signals (BUY/SELL or LONG/SHORT) - needs resolution"
                    )
                elif conflict_type == "exit_with_entry":
                    report["resolution_actions"].append(
                        f"{symbol}: EXIT signal while position entry pending"
                    )
        
        return report
    
    def handle_concurrent_order_conflicts(
        self,
        signals: List[TradingSignal],
        strategy: str = "prioritize_conviction",
    ) -> Dict[str, Any]:
        """
        Handle concurrent order conflicts with specified strategy.
        
        Args:
            signals: List of TradingSignal objects
            strategy: Resolution strategy - "prioritize_conviction", "skip", "error", "keep_first"
            
        Returns:
            Result with resolved signals
        """
        detection = self.detect_concurrent_order_conflicts(signals)
        
        result = {
            "strategy": strategy,
            "detection": detection,
            "resolved_signals": [],
            "can_proceed": True,
            "warnings": [],
            "errors": [],
            "action_taken": None,
        }
        
        if detection["conflicts_found"] == 0:
            result["action_taken"] = "no_conflicts"
            result["resolved_signals"] = signals
            return result
        
        symbol_signals: Dict[str, List[TradingSignal]] = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        if strategy == "error":
            result["can_proceed"] = False
            result["action_taken"] = "blocked_conflicts"
            result["errors"].append(
                f"Cannot proceed: {detection['conflicts_found']} concurrent order conflict(s) "
                f"in {detection['conflicting_symbols']}"
            )
            return result
        
        resolved = []
        
        for symbol, sig_list in symbol_signals.items():
            if len(sig_list) < 2:
                resolved.extend(sig_list)
                continue
            
            if strategy == "skip":
                if symbol in detection["conflicting_symbols"]:
                    result["warnings"].append(f"Skipped conflicting signals for {symbol}")
                    continue
                resolved.extend(sig_list)
            
            elif strategy == "keep_first":
                resolved.append(sig_list[0])
                if len(sig_list) > 1:
                    result["warnings"].append(
                        f"Kept first signal for {symbol}, skipped {len(sig_list) - 1}"
                    )
            
            elif strategy == "prioritize_conviction":
                best_signal = max(sig_list, key=lambda s: (s.conviction, self._get_signal_priority(s.signal)))
                resolved.append(best_signal)
                if len(sig_list) > 1:
                    result["warnings"].append(
                        f"Selected highest conviction ({best_signal.conviction}) for {symbol}"
                    )
            
            else:
                result["errors"].append(f"Unknown strategy: {strategy}")
                result["can_proceed"] = False
                return result
        
        result["resolved_signals"] = resolved
        
        if strategy == "skip":
            result["action_taken"] = "skipped_conflicting_signals"
        elif strategy == "keep_first":
            result["action_taken"] = "kept_first_signal"
        elif strategy == "prioritize_conviction":
            result["action_taken"] = "kept_highest_conviction"
        
        return result
    
    def validate_order_consistency(
        self,
        signals: List[TradingSignal],
    ) -> Dict[str, Any]:
        """
        Validate order consistency for a list of trading signals.
        
        Args:
            signals: List of TradingSignal objects
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        detection = self.detect_concurrent_order_conflicts(signals)
        
        if detection["conflicts_found"] > 0:
            for symbol in detection["conflicting_symbols"]:
                issues.append(f"Concurrent order conflict for {symbol}")
        
        symbol_counts: Dict[str, int] = {}
        for signal in signals:
            symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1
        
        for symbol, count in symbol_counts.items():
            if count > 3:
                warnings.append(f"High signal count for {symbol}: {count} signals")
        
        low_conviction = [s for s in signals if s.conviction < 5]
        if low_conviction:
            warnings.append(f"{len(low_conviction)} signal(s) with conviction < 5")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "conflict_detection": detection,
            "signal_summary": {
                "total_signals": len(signals),
                "unique_symbols": len(symbol_counts),
                "signals_per_symbol": symbol_counts,
            },
        }
    
    def resolve_order_priority(
        self,
        conflicting_signals: List[TradingSignal],
    ) -> Dict[str, Any]:
        """
        Resolve order priority from conflicting signals.
        
        Args:
            conflicting_signals: List of TradingSignal objects for the same symbol
            
        Returns:
            Resolution result with best signal
        """
        if not conflicting_signals:
            return {
                "signal": None,
                "conviction": 0,
                "basis": "no_signals",
            }
        
        if len(conflicting_signals) == 1:
            return {
                "signal": conflicting_signals[0].signal,
                "conviction": conflicting_signals[0].conviction,
                "basis": "single_signal",
            }
        
        signal_priorities = {
            Signal.STRONG_SELL: 5,
            Signal.SELL: 4,
            Signal.STRONG_BUY: 3,
            Signal.BUY: 2,
            Signal.HOLD: 1,
            Signal.NO_SIGNAL: 0,
        }
        
        best = max(
            conflicting_signals,
            key=lambda s: (s.conviction, signal_priorities.get(s.signal, 0))
        )
        
        return {
            "signal": best.signal,
            "conviction": best.conviction,
            "reasoning": best.reasoning,
            "basis": "conviction_and_priority",
            "alternatives": [
                {"signal": s.signal.value, "conviction": s.conviction}
                for s in conflicting_signals if s != best
            ],
        }
    
    def _get_signal_priority(self, signal: Signal) -> int:
        """Get priority for a signal type."""
        priorities = {
            Signal.STRONG_SELL: 5,
            Signal.SELL: 4,
            Signal.STRONG_BUY: 3,
            Signal.BUY: 2,
            Signal.HOLD: 1,
            Signal.NO_SIGNAL: 0,
        }
        return priorities.get(signal, 0)
    
    def validate_backtest_config(
        self,
        config: BacktestConfig,
    ) -> Dict[str, Any]:
        """
        Validate backtest configuration parameters.
        
        Args:
            config: BacktestConfig object to validate
            
        Returns:
            Validation report with issues found
        """
        issues = []
        warnings = []
        
        if config.initial_capital <= 0:
            issues.append(f"initial_capital must be positive, got {config.initial_capital}")
        elif config.initial_capital < 10000:
            warnings.append(f"initial_capital is very low: {config.initial_capital}")
        
        if config.max_positions <= 0:
            issues.append(f"max_positions must be positive, got {config.max_positions}")
        elif config.max_positions > 20:
            warnings.append(f"max_positions is very high: {config.max_positions}")
        
        if config.max_position_pct <= 0:
            issues.append(f"max_position_pct must be positive, got {config.max_position_pct}")
        elif config.max_position_pct > 100:
            issues.append(f"max_position_pct cannot exceed 100, got {config.max_position_pct}")
        
        if config.min_conviction < 1 or config.min_conviction > 10:
            issues.append(f"min_conviction must be 1-10, got {config.min_conviction}")
        
        if config.default_stop_loss_atr <= 0:
            issues.append(f"default_stop_loss_atr must be positive, got {config.default_stop_loss_atr}")
        
        if config.default_target_pct <= 0:
            issues.append(f"default_target_pct must be positive, got {config.default_target_pct}")
        
        if config.warmup_days < 0:
            issues.append(f"warmup_days cannot be negative, got {config.warmup_days}")
        
        if config.max_drawdown_pct <= 0:
            issues.append(f"max_drawdown_pct must be positive, got {config.max_drawdown_pct}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
    
    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Validate date range for backtesting.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        try:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
        except Exception as e:
            issues.append(f"Invalid date format: {e}")
            return {"is_valid": False, "issues": issues, "warnings": warnings}
        
        if start >= end:
            issues.append(f"start_date ({start_date}) must be before end_date ({end_date})")
        
        today = pd.Timestamp.now()
        if start > today:
            warnings.append(f"start_date ({start_date}) is in the future")
        if end > today:
            warnings.append(f"end_date ({end_date}) is in the future")
        
        date_range = (end - start).days
        if date_range < 30:
            warnings.append(f"Date range is very short: {date_range} days")
        elif date_range > 3650:
            warnings.append(f"Date range is very long: {date_range} days (max recommended: 3 years)")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "date_range_days": date_range,
        }
    
    def validate_symbols(
        self,
        symbols: List[str],
    ) -> Dict[str, Any]:
        """
        Validate symbol formats.
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        valid_patterns = [
            r"^[A-Z]+\.[A-Z]{2}$",
            r"^\^[A-Z]+$",
        ]
        
        import re
        
        for i, symbol in enumerate(symbols):
            if not symbol or not symbol.strip():
                issues.append(f"Symbol at index {i} is empty")
                continue
            
            symbol = symbol.strip()
            
            if len(symbol) < 2:
                issues.append(f"Symbol '{symbol}' is too short (min 2 chars)")
            
            if len(symbol) > 20:
                issues.append(f"Symbol '{symbol}' is too long (max 20 chars)")
            
            is_valid_format = any(re.match(pattern, symbol) for pattern in valid_patterns)
            if not is_valid_format:
                warnings.append(f"Symbol '{symbol}' may have non-standard format")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "symbol_count": len(symbols),
        }
    
    def validate_risk_parameters(
        self,
        risk_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate risk parameters.
        
        Args:
            risk_params: Dictionary of risk parameters
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        limits = {
            "max_position_pct": (0, 100),
            "max_drawdown_pct": (0, 100),
            "max_daily_loss_pct": (0, 50),
            "stop_loss_atr": (0.1, 10),
            "trailing_stop_atr": (0.1, 10),
        }
        
        for param, (min_val, max_val) in limits.items():
            if param in risk_params:
                value = risk_params[param]
                if not isinstance(value, (int, float)):
                    issues.append(f"{param} must be numeric, got {type(value).__name__}")
                elif value < min_val or value > max_val:
                    issues.append(f"{param} must be {min_val}-{max_val}, got {value}")
        
        if "max_positions" in risk_params:
            value = risk_params["max_positions"]
            if not isinstance(value, int):
                issues.append(f"max_positions must be int, got {type(value).__name__}")
            elif value < 1 or value > 50:
                issues.append(f"max_positions must be 1-50, got {value}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }
    
    def validate_signal_parameters(
        self,
        signals: List[TradingSignal],
    ) -> Dict[str, Any]:
        """
        Validate trading signal parameters.
        
        Args:
            signals: List of TradingSignal objects
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        for i, signal in enumerate(signals):
            if not signal.symbol or not signal.symbol.strip():
                issues.append(f"Signal at index {i} has empty symbol")
            
            if signal.conviction < 1 or signal.conviction > 10:
                issues.append(
                    f"Signal for {signal.symbol} has invalid conviction {signal.conviction} (must be 1-10)"
                )
            
            if signal.reasoning and len(signal.reasoning.strip()) < 5:
                warnings.append(f"Signal for {signal.symbol} has very short reasoning")
            
            if signal.signal == Signal.NO_SIGNAL:
                warnings.append(f"Signal for {signal.symbol} is NO_SIGNAL")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "signal_count": len(signals),
        }
    
    def validate_all_parameters(
        self,
        config: Optional[BacktestConfig] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of all input parameters.
        
        Args:
            config: BacktestConfig object
            start_date: Start date string
            end_date: End date string
            symbols: List of symbols
            risk_parameters: Risk parameters dictionary
            
        Returns:
            Combined validation report
        """
        results = {
            "is_valid": True,
            "validation_results": {},
            "all_issues": [],
            "all_warnings": [],
        }
        
        if config is not None:
            config_result = self.validate_backtest_config(config)
            results["validation_results"]["config_valid"] = config_result["is_valid"]
            results["all_issues"].extend(config_result["issues"])
            results["all_warnings"].extend(config_result["warnings"])
        else:
            results["validation_results"]["config_valid"] = None
        
        if start_date and end_date:
            date_result = self.validate_date_range(start_date, end_date)
            results["validation_results"]["dates_valid"] = date_result["is_valid"]
            results["all_issues"].extend(date_result["issues"])
            results["all_warnings"].extend(date_result["warnings"])
        else:
            results["validation_results"]["dates_valid"] = None
        
        if symbols:
            symbols_result = self.validate_symbols(symbols)
            results["validation_results"]["symbols_valid"] = symbols_result["is_valid"]
            results["all_issues"].extend(symbols_result["issues"])
            results["all_warnings"].extend(symbols_result["warnings"])
        else:
            results["validation_results"]["symbols_valid"] = None
        
        if risk_parameters:
            risk_result = self.validate_risk_parameters(risk_parameters)
            results["validation_results"]["risk_valid"] = risk_result["is_valid"]
            results["all_issues"].extend(risk_result["issues"])
            results["all_warnings"].extend(risk_result["warnings"])
        else:
            results["validation_results"]["risk_valid"] = None
        
        results["is_valid"] = all(
            v is None or v for v in results["validation_results"].values()
        )
        
        return results
    
    def _init_logging(self):
        """Initialize logging storage if not already done."""
        if not hasattr(self, '_error_log'):
            self._error_log: List[Dict[str, Any]] = []
        if not hasattr(self, '_warning_log'):
            self._warning_log: List[Dict[str, Any]] = []
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
    
    def log_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log an error.
        
        Args:
            error_type: Type/category of error
            message: Error message
            context: Additional context data
            
        Returns:
            Log result with error_id
        """
        self._init_logging()
        self._log_counter += 1
        
        error_entry = {
            "error_id": f"ERR_{self._log_counter:05d}",
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "context": context or {},
        }
        
        self._error_log.append(error_entry)
        
        return {
            "logged": True,
            "error_id": error_entry["error_id"],
            "entry": error_entry,
        }
    
    def log_warning(
        self,
        warning_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a warning.
        
        Args:
            warning_type: Type/category of warning
            message: Warning message
            context: Additional context data
            
        Returns:
            Log result with warning_id
        """
        self._init_logging()
        self._log_counter += 1
        
        warning_entry = {
            "warning_id": f"WARN_{self._log_counter:05d}",
            "timestamp": datetime.now().isoformat(),
            "type": warning_type,
            "message": message,
            "context": context or {},
        }
        
        self._warning_log.append(warning_entry)
        
        return {
            "logged": True,
            "warning_id": warning_entry["warning_id"],
            "entry": warning_entry,
        }
    
    def get_error_log(
        self,
        limit: int = 100,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get error log entries.
        
        Args:
            limit: Maximum number of entries to return
            error_type: Filter by error type (optional)
            
        Returns:
            Error log with count and entries
        """
        self._init_logging()
        
        entries = self._error_log
        if error_type:
            entries = [e for e in entries if e['type'] == error_type]
        
        return {
            "count": len(entries),
            "recent": entries[-limit:],
        }
    
    def get_warning_log(
        self,
        limit: int = 100,
        warning_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get warning log entries.
        
        Args:
            limit: Maximum number of entries to return
            warning_type: Filter by warning type (optional)
            
        Returns:
            Warning log with count and entries
        """
        self._init_logging()
        
        entries = self._warning_log
        if warning_type:
            entries = [e for e in entries if e['type'] == warning_type]
        
        return {
            "count": len(entries),
            "recent": entries[-limit:],
        }
    
    def export_logs(
        self,
        output_path: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """
        Export logs to file.
        
        Args:
            output_path: Path to output file
            format: Export format (json or csv)
            
        Returns:
            Export result
        """
        import json
        import csv
        
        self._init_logging()
        
        try:
            if format == "json":
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "errors": {
                        "count": len(self._error_log),
                        "entries": self._error_log,
                    },
                    "warnings": {
                        "count": len(self._warning_log),
                        "entries": self._warning_log,
                    },
                }
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == "csv":
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'type', 'message', 'id'])
                    for entry in self._error_log:
                        writer.writerow([entry['timestamp'], 'ERROR', entry['message'], entry['error_id']])
                    for entry in self._warning_log:
                        writer.writerow([entry['timestamp'], 'WARNING', entry['message'], entry['warning_id']])
            else:
                return {"success": False, "error": f"Unknown format: {format}"}
            
            return {
                "success": True,
                "file_path": output_path,
                "errors_exported": len(self._error_log),
                "warnings_exported": len(self._warning_log),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clear_logs(
        self,
        log_type: str = "all",
    ) -> Dict[str, Any]:
        """
        Clear logs.
        
        Args:
            log_type: Type to clear - "all", "errors", "warnings"
            
        Returns:
            Clear result
        """
        self._init_logging()
        
        cleared = 0
        if log_type in ["all", "errors"]:
            cleared = len(self._error_log)
            self._error_log = []
        if log_type in ["all", "warnings"]:
            cleared += len(self._warning_log)
            self._warning_log = []
        
        return {
            "success": True,
            "cleared_count": cleared,
            "log_type": log_type,
        }
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary of all logs.
        
        Returns:
            Log summary with counts and types
        """
        self._init_logging()
        
        error_types: Dict[str, int] = {}
        for entry in self._error_log:
            error_types[entry['type']] = error_types.get(entry['type'], 0) + 1
        
        warning_types: Dict[str, int] = {}
        for entry in self._warning_log:
            warning_types[entry['type']] = warning_types.get(entry['type'], 0) + 1
        
        return {
            "total_errors": len(self._error_log),
            "total_warnings": len(self._warning_log),
            "error_types": error_types,
            "warning_types": warning_types,
        }
    
    def _init_retry_stats(self):
        """Initialize retry stats storage."""
        if not hasattr(self, '_retry_stats'):
            self._retry_stats = {
                'total_retries': 0,
                'successful_retries': 0,
                'failed_retries': 0,
                'retry_history': [],
            }
    
    def fetch_with_retry(
        self,
        fetch_func,
        max_retries: int = 3,
        delay_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Fetch data with retry logic.
        
        Args:
            fetch_func: Function to call that may fail
            max_retries: Maximum number of retry attempts
            delay_seconds: Delay between retries
            
        Returns:
            Result with success status and data
        """
        self._init_retry_stats()
        import time
        
        for attempt in range(1, max_retries + 1):
            self._retry_stats['total_retries'] += 1
            
            try:
                result = fetch_func()
                self._retry_stats['successful_retries'] += 1
                self._retry_stats['retry_history'].append({
                    'attempt': attempt,
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                })
                
                return {
                    'success': True,
                    'data': result,
                    'attempts': attempt,
                }
            except Exception as e:
                error_msg = str(e)
                self._retry_stats['retry_history'].append({
                    'attempt': attempt,
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat(),
                })
                
                if attempt < max_retries:
                    time.sleep(delay_seconds)
        
        self._retry_stats['failed_retries'] += 1
        
        return {
            'success': False,
            'data': None,
            'attempts': max_retries,
            'error': error_msg,
        }
    
    def retry_with_exponential_backoff(
        self,
        func,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Retry with exponential backoff.
        
        Args:
            func: Function to call
            max_retries: Maximum retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap
            
        Returns:
            Result with success status
        """
        self._init_retry_stats()
        import time
        
        for attempt in range(1, max_retries + 1):
            self._retry_stats['total_retries'] += 1
            
            try:
                result = func()
                self._retry_stats['successful_retries'] += 1
                
                return {
                    'success': True,
                    'data': result,
                    'attempts': attempt,
                }
            except Exception as e:
                error_msg = str(e)
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                
                if attempt < max_retries:
                    time.sleep(delay)
        
        self._retry_stats['failed_retries'] += 1
        
        return {
            'success': False,
            'data': None,
            'attempts': max_retries,
            'error': error_msg,
        }
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get retry statistics.
        
        Returns:
            Retry statistics
        """
        self._init_retry_stats()
        return self._retry_stats.copy()
    
    def clear_retry_stats(self) -> Dict[str, Any]:
        """
        Clear retry statistics.
        
        Returns:
            Clear result
        """
        self._init_retry_stats()
        
        self._retry_stats = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'retry_history': [],
        }
        
        return {
            'success': True,
            'message': 'Retry stats cleared',
        }
