"""
Performance metrics calculation for backtesting.

Calculates comprehensive performance metrics from trade log and equity curve.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.backtesting.portfolio import Trade, PortfolioState


class MetricsCalculator:
    """
    Calculates performance metrics from portfolio state.
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(
        self,
        portfolio: PortfolioState,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        trades_df = portfolio.get_trade_log_df()
        equity_df = portfolio.get_equity_curve_df()
        
        if trades_df.empty:
            return self._empty_metrics()
        
        metrics.update(self._return_metrics(portfolio))
        metrics.update(self._risk_metrics(trades_df, equity_df))
        metrics.update(self._trade_metrics(trades_df))
        
        if not equity_df.empty:
            metrics.update(self._rolling_metrics(equity_df))
            
        if benchmark_returns is not None and not equity_df.empty:
            metrics.update(self._benchmark_metrics(equity_df, benchmark_returns))
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics when no trades."""
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }
    
    def _return_metrics(self, portfolio: PortfolioState) -> Dict[str, float]:
        """Calculate return-based metrics."""
        equity_df = portfolio.get_equity_curve_df()
        
        if equity_df.empty:
            return {}
        
        total_return = portfolio.total_return
        total_return_pct = portfolio.total_return_pct
        
        start_date = equity_df["date"].iloc[0]
        end_date = equity_df["date"].iloc[-1]
        years = (end_date - start_date).days / 365.25
        
        if years > 0:
            annualized_return = ((portfolio.portfolio_value / portfolio.initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0
        
        return {
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "annualized_return": annualized_return,
            "initial_capital": portfolio.initial_capital,
            "final_capital": portfolio.portfolio_value,
        }
    
    def _risk_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        metrics = {}
        
        if not equity_df.empty:
            equity_df = equity_df.copy()
            equity_df["drawdown"] = equity_df["total_value"].cummax() - equity_df["total_value"]
            equity_df["drawdown_pct"] = equity_df["drawdown"] / equity_df["total_value"].cummax() * 100
            
            max_drawdown = equity_df["drawdown"].max()
            max_drawdown_pct = equity_df["drawdown_pct"].max()
            
            metrics["max_drawdown"] = max_drawdown
            metrics["max_drawdown_pct"] = max_drawdown_pct
            
            in_drawdown = equity_df["drawdown"] > 0
            if in_drawdown.any():
                max_dd_duration = 0
                current_dd_duration = 0
                for is_dd in in_drawdown:
                    if is_dd:
                        current_dd_duration += 1
                        max_dd_duration = max(max_dd_duration, current_dd_duration)
                    else:
                        current_dd_duration = 0
                metrics["max_drawdown_duration"] = max_dd_duration
            else:
                metrics["max_drawdown_duration"] = 0
            
            if "total_return_pct" in equity_df.columns:
                returns = equity_df["total_return_pct"].pct_change().dropna()
                
                if len(returns) > 0:
                    annual_vol = returns.std() * np.sqrt(252) * 100
                    metrics["annualized_volatility"] = annual_vol
                    
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std() * np.sqrt(252) * 100
                        metrics["downside_deviation"] = downside_std
                    else:
                        metrics["downside_deviation"] = 0.0
                else:
                    metrics["annualized_volatility"] = 0.0
                    metrics["downside_deviation"] = 0.0
            else:
                metrics["annualized_volatility"] = 0.0
                metrics["downside_deviation"] = 0.0
        else:
            metrics["max_drawdown"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["max_drawdown_duration"] = 0
            metrics["annualized_volatility"] = 0.0
            metrics["downside_deviation"] = 0.0
        
        if len(trades_df) > 0:
            daily_returns = trades_df["pnl_pct"] / 100
            
            if len(daily_returns) > 0:
                annual_vol = daily_returns.std() * np.sqrt(252) * 100 if daily_returns.std() > 0 else 0.0
                
                if annual_vol > 0:
                    sharpe = ((daily_returns.mean() * 252) - self.risk_free_rate) / annual_vol
                    sortino = ((daily_returns.mean() * 252) - self.risk_free_rate) / (metrics["downside_deviation"] if metrics["downside_deviation"] > 0 else 0.01)
                else:
                    sharpe = 0.0
                    sortino = 0.0
                
                metrics["sharpe_ratio"] = sharpe
                metrics["sortino_ratio"] = sortino
                
                if daily_returns.mean() > 0 and metrics["max_drawdown_pct"] > 0:
                    metrics["calmar_ratio"] = (daily_returns.mean() * 252) / (metrics["max_drawdown_pct"] / 100)
                else:
                    metrics["calmar_ratio"] = 0.0
            else:
                metrics["sharpe_ratio"] = 0.0
                metrics["sortino_ratio"] = 0.0
                metrics["calmar_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0
            metrics["calmar_ratio"] = 0.0
        
        return metrics
    
    def _trade_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        if trades_df.empty:
            return {}
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        winning_pnl = trades_df[trades_df["pnl"] > 0]["pnl_pct"]
        losing_pnl = trades_df[trades_df["pnl"] <= 0]["pnl_pct"]
        
        avg_win_pct = winning_pnl.mean() if len(winning_pnl) > 0 else 0.0
        avg_loss_pct = abs(losing_pnl.mean()) if len(losing_pnl) > 0 else 0.0
        
        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0.0
        avg_loss = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].mean()) if losing_trades > 0 else 0.0
        
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        expectancy = (winning_trades / total_trades * avg_win_pct) - (losing_trades / total_trades * avg_loss_pct) if total_trades > 0 else 0.0
        
        avg_duration = trades_df["duration_days"].mean() if "duration_days" in trades_df.columns else 0.0
        
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for _, trade in trades_df.iterrows():
            if trade["pnl"] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            "total_trades": total_trades,
            "profitable_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "profit_factor": profit_factor,
            "expectancy_pct": expectancy,
            "avg_duration_days": avg_duration,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
        }
    
    def _rolling_metrics(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate rolling metrics."""
        if "total_return_pct" not in equity_df.columns or len(equity_df) < 30:
            return {}
        
        returns = equity_df["total_return_pct"].pct_change().dropna()
        
        if len(returns) >= 30:
            rolling_mean = returns.rolling(30).mean()
            rolling_std = returns.rolling(30).std()
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
            max_rolling_sharpe = rolling_sharpe.max()
        else:
            max_rolling_sharpe = 0.0
        
        return {
            "max_rolling_sharpe": max_rolling_sharpe,
        }
    
    def _benchmark_metrics(
        self,
        equity_df: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        if "total_return_pct" not in equity_df.columns:
            return {}
        
        strategy_returns = equity_df["total_return_pct"].pct_change().dropna()
        
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}
        
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        excess_returns = strategy_returns - benchmark_returns
        
        if excess_returns.std() > 0:
            information_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
        else:
            information_ratio = 0.0
        
        beta = 0.0
        if benchmark_returns.std() > 0:
            covariance = strategy_returns.cov(benchmark_returns)
            beta = covariance / benchmark_returns.var()
        
        alpha = (strategy_returns.mean() * 252) - self.risk_free_rate - beta * ((benchmark_returns.mean() * 252) - self.risk_free_rate)
        
        tracking_error = excess_returns.std() * np.sqrt(252) * 100
        
        return {
            "information_ratio": information_ratio,
            "beta": beta,
            "alpha": alpha * 100,
            "tracking_error": tracking_error,
        }
    
    def get_summary_dataframe(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """Convert metrics dict to formatted DataFrame."""
        rows = []
        for key, value in metrics.items():
            if isinstance(value, float):
                rows.append({"Metric": key, "Value": f"{value:.2f}"})
            else:
                rows.append({"Metric": key, "Value": value})
        return pd.DataFrame(rows)
