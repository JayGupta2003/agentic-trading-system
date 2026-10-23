"""
Visualization and reporting for backtesting.

Provides plotting functionality for equity curves, drawdowns, trades, and performance metrics.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figsize: Tuple[int, int] = (12, 6)
    style: str = 'seaborn-v0_8-darkgrid'
    title_fontsize: int = 14
    label_fontsize: int = 11
    line_width: float = 1.5
    color_equity: str = '#2E86AB'
    color_benchmark: str = '#A23B72'
    color_positive: str = '#28A745'
    color_negative: str = '#DC3545'
    save_dpi: int = 150
    save_format: str = 'png'


class BacktestVisualizer:
    """
    Visualization tools for backtest results.
    
    Generates equity curves, drawdown charts, trade markers, and performance visualizations.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Plot configuration
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        self.config = config or PlotConfig()
        plt.style.use(self.config.style)
        
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot equity curve with optional benchmark comparison.
        
        Args:
            equity_curve: Series with dates as index and portfolio values
            benchmark: Optional benchmark series for comparison
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(equity_curve.index, equity_curve.values, 
                label='Strategy', color=self.config.color_equity, 
                linewidth=self.config.line_width)
        
        if benchmark is not None:
            aligned_benchmark = benchmark.reindex(equity_curve.index, method='ffill')
            ax.plot(aligned_benchmark.index, aligned_benchmark.values,
                    label='Benchmark', color=self.config.color_benchmark,
                    linewidth=self.config.line_width, alpha=0.7)
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Date', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Portfolio Value', fontsize=self.config.label_fontsize)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def get_equity_stats(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate statistics for equity curve display.
        
        Args:
            equity_curve: Series with portfolio values
            
        Returns:
            Dict of statistics
        """
        returns = equity_curve.pct_change().dropna()
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        max_value = equity_curve.max()
        min_value = equity_curve.min()
        
        return {
            'start_value': equity_curve.iloc[0],
            'end_value': equity_curve.iloc[-1],
            'total_return_pct': total_return,
            'max_value': max_value,
            'min_value': min_value,
            'peak_value': equity_curve.cummax().iloc[-1],
            'n_days': len(equity_curve),
        }
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown series from equity curve.
        
        Args:
            equity_curve: Series with portfolio values
            
        Returns:
            Series with drawdown values (negative percentages)
        """
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak * 100
        return drawdown
    
    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = "Drawdown Chart",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot drawdown chart.
        
        Args:
            equity_curve: Series with dates as index and portfolio values
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        drawdown = self.calculate_drawdown(equity_curve)
        
        ax.fill_between(drawdown.index, drawdown.values, 0,
                       color=self.config.color_negative, alpha=0.5)
        ax.plot(drawdown.index, drawdown.values,
                color=self.config.color_negative, linewidth=self.config.line_width)
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.axhline(y=max_dd, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(f'Max DD: {max_dd:.2f}%',
                   xy=(max_dd_date, max_dd),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, color='red',
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Date', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Drawdown (%)', fontsize=self.config.label_fontsize)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def get_drawdown_stats(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Calculate drawdown statistics.
        
        Args:
            equity_curve: Series with portfolio values
            
        Returns:
            Dict of drawdown statistics
        """
        drawdown = self.calculate_drawdown(equity_curve)
        
        peaks = equity_curve.cummax()
        in_drawdown = equity_curve < peaks
        
        drawdown_periods = []
        current_dd_start = None
        
        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and current_dd_start is None:
                current_dd_start = date
            elif not is_dd and current_dd_start is not None:
                dd_end = equity_curve.index[i - 1] if i > 0 else date
                dd_value = (equity_curve.iloc[i - 1] - peaks.iloc[i - 1]) / peaks.iloc[i - 1] * 100
                drawdown_periods.append({
                    'start': current_dd_start,
                    'end': dd_end,
                    'depth': dd_value
                })
                current_dd_start = None
        
        return {
            'max_drawdown_pct': drawdown.min(),
            'max_drawdown_date': drawdown.idxmin(),
            'avg_drawdown_pct': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'current_drawdown_pct': drawdown.iloc[-1],
            'n_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration_days': np.mean([(p['end'] - p['start']).days for p in drawdown_periods]) if drawdown_periods else 0,
        }
    
    def plot_entry_exit_markers(
        self,
        price_data: pd.DataFrame,
        trades: List[Any],
        title: str = "Entry/Exit Markers",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot price chart with entry/exit markers.
        
        Args:
            price_data: DataFrame with 'close' column and datetime index
            trades: List of trade objects with entry/exit info
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.plot(price_data.index, price_data['close'].values,
                color=self.config.color_equity, linewidth=self.config.line_width, label='Close Price')
        
        entry_dates = []
        entry_prices = []
        exit_dates = []
        exit_prices = []
        
        for trade in trades:
            if hasattr(trade, 'entry_date') and hasattr(trade, 'entry_price'):
                entry_dates.append(trade.entry_date)
                entry_prices.append(trade.entry_price)
            if hasattr(trade, 'exit_date') and hasattr(trade, 'exit_price'):
                exit_dates.append(trade.exit_date)
                exit_prices.append(trade.exit_price)
        
        if entry_dates:
            ax.scatter(entry_dates, entry_prices, color='green', marker='^', 
                      s=100, label='Entry', zorder=5)
        
        if exit_dates:
            ax.scatter(exit_dates, exit_prices, color='red', marker='v',
                      s=100, label='Exit', zorder=5)
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Date', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Price', fontsize=self.config.label_fontsize)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate monthly returns in a pivot table format.
        
        Args:
            equity_curve: Series with portfolio values
            
        Returns:
            DataFrame with years as rows and months as columns
        """
        returns = equity_curve.pct_change().dropna()
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return pivot * 100
    
    def plot_monthly_returns_heatmap(
        self,
        equity_curve: pd.Series,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot monthly returns as a heatmap.
        
        Args:
            equity_curve: Series with dates as index and portfolio values
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        monthly_returns = self.calculate_monthly_returns(equity_curve)
        
        im = ax.imshow(monthly_returns.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        ax.set_xticks(range(len(monthly_returns.columns)))
        ax.set_xticklabels(monthly_returns.columns)
        ax.set_yticks(range(len(monthly_returns.index)))
        ax.set_yticklabels(monthly_returns.index)
        
        for i in range(len(monthly_returns.index)):
            for j in range(len(monthly_returns.columns)):
                value = monthly_returns.values[i, j]
                if not np.isnan(value):
                    color = 'white' if abs(value) > 5 else 'black'
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color=color, fontsize=9)
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Month', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Year', fontsize=self.config.label_fontsize)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', fontsize=self.config.label_fontsize)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def trades_to_dataframe(self, trades: List[Any]) -> pd.DataFrame:
        """
        Convert list of trades to a formatted DataFrame.
        
        Args:
            trades: List of trade objects
            
        Returns:
            DataFrame with trade information
        """
        trade_data = []
        
        for trade in trades:
            row = {
                'symbol': getattr(trade, 'symbol', 'N/A'),
                'entry_date': getattr(trade, 'entry_date', None),
                'entry_price': getattr(trade, 'entry_price', 0),
                'exit_date': getattr(trade, 'exit_date', None),
                'exit_price': getattr(trade, 'exit_price', 0),
                'quantity': getattr(trade, 'quantity', 0),
                'pnl': getattr(trade, 'pnl', 0),
                'pnl_pct': getattr(trade, 'pnl_pct', 0),
                'holding_days': getattr(trade, 'holding_days', 0),
                'exit_reason': getattr(trade, 'exit_reason', 'N/A'),
            }
            
            if row['entry_price'] > 0:
                row['pnl_pct'] = (row['exit_price'] - row['entry_price']) / row['entry_price'] * 100
            
            trade_data.append(row)
        
        return pd.DataFrame(trade_data)
    
    def plot_trade_log_table(
        self,
        trades: List[Any],
        title: str = "Trade Log",
        max_rows: int = 20,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot trade log as a formatted table.
        
        Args:
            trades: List of trade objects
            title: Plot title
            max_rows: Maximum number of rows to display
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        trade_df = self.trades_to_dataframe(trades)
        
        if len(trade_df) > max_rows:
            trade_df = trade_df.tail(max_rows)
        
        fig, ax = plt.subplots(figsize=(14, min(8, 2 + len(trade_df) * 0.4)))
        ax.axis('off')
        
        table = ax.table(
            cellText=trade_df.values,
            colLabels=trade_df.columns,
            cellLoc='center',
            loc='center',
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        for i, col in enumerate(trade_df.columns):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(trade_df) + 1):
            pnl = trade_df.iloc[i-1]['pnl']
            if pnl > 0:
                table[(i, 6)].set_facecolor('#C6EFCE')
            elif pnl < 0:
                table[(i, 6)].set_facecolor('#FFC7CE')
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold', pad=20)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_performance_summary(
        self,
        metrics: Dict[str, float],
        trades: Optional[List[Any]] = None,
        title: str = "Performance Summary",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot performance summary as a formatted text box.
        
        Args:
            metrics: Dict of performance metrics
            trades: Optional list of trades for additional stats
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append(f"{title:^50}")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        summary_lines.append("RETURN METRICS")
        summary_lines.append("-" * 50)
        for key in ['total_return_pct', 'annualized_return_pct', 'benchmark_return_pct']:
            if key in metrics:
                value = metrics[key]
                summary_lines.append(f"  {key.replace('_', ' ').title()}: {value:>10.2f}%")
        
        summary_lines.append("")
        summary_lines.append("RISK METRICS")
        summary_lines.append("-" * 50)
        for key in ['max_drawdown_pct', 'volatility_pct', 'var_95_pct', 'cvar_95_pct']:
            if key in metrics:
                value = metrics[key]
                summary_lines.append(f"  {key.replace('_', ' ').title()}: {value:>10.2f}%")
        
        summary_lines.append("")
        summary_lines.append("RISK-ADJUSTED METRICS")
        summary_lines.append("-" * 50)
        for key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']:
            if key in metrics:
                value = metrics[key]
                summary_lines.append(f"  {key.replace('_', ' ').title()}: {value:>10.4f}")
        
        if trades:
            trade_df = self.trades_to_dataframe(trades)
            summary_lines.append("")
            summary_lines.append("TRADE STATISTICS")
            summary_lines.append("-" * 50)
            summary_lines.append(f"  Total Trades: {len(trades)}")
            if 'win_rate' in metrics:
                summary_lines.append(f"  Win Rate: {metrics['win_rate']:>10.2f}%")
            if 'avg_win_pct' in metrics:
                summary_lines.append(f"  Avg Win: {metrics['avg_win_pct']:>10.2f}%")
            if 'avg_loss_pct' in metrics:
                summary_lines.append(f"  Avg Loss: {metrics['avg_loss_pct']:>10.2f}%")
            if 'profit_factor' in metrics:
                summary_lines.append(f"  Profit Factor: {metrics['profit_factor']:>10.4f}")
        
        summary_lines.append("")
        summary_lines.append("=" * 50)
        
        ax.text(0.05, 0.95, '\n'.join(summary_lines),
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_strategy_vs_benchmark(
        self,
        equity_curve: pd.Series,
        benchmark: pd.Series,
        title: str = "Strategy vs Benchmark Comparison",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot strategy vs benchmark comparison chart.
        
        Args:
            equity_curve: Strategy equity curve
            benchmark: Benchmark equity curve
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize, height_ratios=[2, 1])
        
        aligned_benchmark = benchmark.reindex(equity_curve.index, method='ffill')
        
        strategy_normalized = equity_curve / equity_curve.iloc[0] * 100
        benchmark_normalized = aligned_benchmark / aligned_benchmark.iloc[0] * 100
        
        ax1.plot(strategy_normalized.index, strategy_normalized.values,
                 label='Strategy', color=self.config.color_equity, linewidth=self.config.line_width)
        ax1.plot(benchmark_normalized.index, benchmark_normalized.values,
                 label='Benchmark', color=self.config.color_benchmark, linewidth=self.config.line_width)
        
        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax1.set_ylabel('Normalized Value (100 = Start)', fontsize=self.config.label_fontsize)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        excess_returns = (strategy_normalized / benchmark_normalized - 1) * 100
        ax2.fill_between(excess_returns.index, excess_returns.values, 0,
                        where=excess_returns.values >= 0, color=self.config.color_positive, alpha=0.5)
        ax2.fill_between(excess_returns.index, excess_returns.values, 0,
                        where=excess_returns.values < 0, color=self.config.color_negative, alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Excess Return (%)', fontsize=self.config.label_fontsize)
        ax2.set_xlabel('Date', fontsize=self.config.label_fontsize)
        ax2.grid(True, alpha=0.3)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def calculate_rolling_sharpe(
        self,
        equity_curve: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.0,
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            equity_curve: Series with portfolio values
            window: Rolling window size (default 63 = ~3 months)
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Series with rolling Sharpe ratios
        """
        returns = equity_curve.pct_change().dropna()
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        daily_rf = risk_free_rate / 252
        rolling_sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(252)
        return rolling_sharpe
    
    def plot_rolling_sharpe(
        self,
        equity_curve: pd.Series,
        window: int = 63,
        title: str = "Rolling Sharpe Ratio",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot rolling Sharpe ratio chart.
        
        Args:
            equity_curve: Series with dates as index and portfolio values
            window: Rolling window size (default 63 = ~3 months)
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        rolling_sharpe = self.calculate_rolling_sharpe(equity_curve, window=window)
        
        ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                       where=rolling_sharpe.values >= 0, color=self.config.color_positive, alpha=0.5)
        ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                       where=rolling_sharpe.values < 0, color=self.config.color_negative, alpha=0.5)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                color='black', linewidth=self.config.line_width, alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Sharpe = -1')
        
        ax.set_title(f"{title} ({window}-day window)", fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Date', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Sharpe Ratio', fontsize=self.config.label_fontsize)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_trade_distribution(
        self,
        trades: List[Any],
        title: str = "Trade Returns Distribution",
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Any:
        """
        Plot trade returns distribution histogram.
        
        Args:
            trades: List of trade objects
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        trade_df = self.trades_to_dataframe(trades)
        returns = trade_df['pnl_pct'].values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(returns, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax1.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax1.set_title('Trade Returns Distribution', fontsize=self.config.title_fontsize, fontweight='bold')
        ax1.set_xlabel('Return (%)', fontsize=self.config.label_fontsize)
        ax1.set_ylabel('Frequency', fontsize=self.config.label_fontsize)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.boxplot(returns, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7))
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_title('Returns Box Plot', fontsize=self.config.title_fontsize, fontweight='bold')
        ax2.set_ylabel('Return (%)', fontsize=self.config.label_fontsize)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, fontweight='bold')
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.save_dpi, format=self.config.save_format)
        
        if show:
            plt.show()
        
        return fig
    
    def export_trades_to_csv(
        self,
        trades: List[Any],
        filepath: str,
        include_stats: bool = True,
    ) -> str:
        """
        Export trade log to CSV file.
        
        Args:
            trades: List of trade objects
            filepath: Path to save the CSV file
            include_stats: Whether to include summary statistics row
            
        Returns:
            Path to the saved file
        """
        trade_df = self.trades_to_dataframe(trades)
        
        if include_stats:
            total_pnl = trade_df['pnl'].sum()
            winning_trades = trade_df[trade_df['pnl'] > 0]
            losing_trades = trade_df[trade_df['pnl'] < 0]
            
            stats_row = pd.DataFrame({
                'symbol': ['SUMMARY'],
                'entry_date': [''],
                'entry_price': [''],
                'exit_date': [''],
                'exit_price': [''],
                'quantity': [len(trade_df)],
                'pnl': [total_pnl],
                'pnl_pct': [total_pnl / len(trade_df) if len(trade_df) > 0 else 0],
                'holding_days': [trade_df['holding_days'].mean() if len(trade_df) > 0 else 0],
                'exit_reason': [f"W:{len(winning_trades)} L:{len(losing_trades)} WR:{len(winning_trades)/len(trade_df)*100:.1f}%" if len(trade_df) > 0 else 'N/A'],
            })
            
            trade_df = pd.concat([trade_df, stats_row], ignore_index=True)
        
        trade_df.to_csv(filepath, index=False)
        return filepath
    
    def generate_summary_dataframe(
        self,
        equity_curve: pd.Series,
        trades: List[Any],
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.06,
    ) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics DataFrame.
        
        Args:
            equity_curve: Series with portfolio values
            trades: List of trade objects
            benchmark: Optional benchmark series for comparison
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            
        Returns:
            DataFrame with summary statistics organized by category
        """
        trade_df = self.trades_to_dataframe(trades)
        
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annual_return = total_return * 252 / len(returns) if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) * 100
        
        daily_rf = risk_free_rate / 252
        sharpe = (returns.mean() - daily_rf) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return / 100 - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        drawdown = self.calculate_drawdown(equity_curve)
        max_dd = drawdown.min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        winning_trades = trade_df[trade_df['pnl'] > 0]
        losing_trades = trade_df[trade_df['pnl'] < 0]
        win_rate = len(winning_trades) / len(trade_df) * 100 if len(trade_df) > 0 else 0
        
        stats_data = {
            'Metric': [
                'Portfolio Value (Start)',
                'Portfolio Value (End)',
                'Total Return (%)',
                'Annualized Return (%)',
                'Annualized Volatility (%)',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Calmar Ratio',
                'Max Drawdown (%)',
                'Total Trades',
                'Winning Trades',
                'Losing Trades',
                'Win Rate (%)',
                'Avg Win (Rs)',
                'Avg Loss (Rs)',
                'Profit Factor',
                'Avg Holding Days',
                'Total P&L (Rs)',
            ],
            'Value': [
                f"{equity_curve.iloc[0]:,.2f}",
                f"{equity_curve.iloc[-1]:,.2f}",
                f"{total_return:.2f}",
                f"{annual_return:.2f}",
                f"{volatility:.2f}",
                f"{sharpe:.4f}",
                f"{sortino:.4f}",
                f"{calmar:.4f}",
                f"{max_dd:.2f}",
                len(trade_df),
                len(winning_trades),
                len(losing_trades),
                f"{win_rate:.2f}",
                f"{winning_trades['pnl'].mean():.2f}" if len(winning_trades) > 0 else "N/A",
                f"{losing_trades['pnl'].mean():.2f}" if len(losing_trades) > 0 else "N/A",
                f"{abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.4f}" if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else "N/A",
                f"{trade_df['holding_days'].mean():.1f}" if len(trade_df) > 0 else "0",
                f"{trade_df['pnl'].sum():,.2f}",
            ]
        }
        
        summary_df = pd.DataFrame(stats_data)
        
        if benchmark is not None:
            aligned_benchmark = benchmark.reindex(equity_curve.index, method='ffill')
            bm_return = (aligned_benchmark.iloc[-1] / aligned_benchmark.iloc[0] - 1) * 100
            alpha = total_return - bm_return
            
            benchmark_stats = pd.DataFrame({
                'Metric': ['Benchmark Return (%)', 'Alpha (%)'],
                'Value': [f"{bm_return:.2f}", f"{alpha:.2f}"]
            })
            summary_df = pd.concat([summary_df, benchmark_stats], ignore_index=True)
        
        return summary_df
    
    def plot_interactive_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot interactive equity curve using Plotly.
        
        Args:
            equity_curve: Series with dates as index and portfolio values
            benchmark: Optional benchmark series for comparison
            title: Plot title
            save_path: Path to save the HTML file
            
        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("plotly is required for interactive charts. Install with: pip install plotly")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color=self.config.color_equity, width=2)
        ))
        
        if benchmark is not None:
            aligned_benchmark = benchmark.reindex(equity_curve.index, method='ffill')
            fig.add_trace(go.Scatter(
                x=aligned_benchmark.index,
                y=aligned_benchmark.values,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.config.color_benchmark, width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_interactive_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = "Drawdown Chart",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot interactive drawdown chart using Plotly.
        
        Args:
            equity_curve: Series with portfolio values
            title: Plot title
            save_path: Path to save the HTML file
            
        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for interactive charts. Install with: pip install plotly")
        
        drawdown = self.calculate_drawdown(equity_curve)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=self.config.color_negative, width=1.5),
            fillcolor='rgba(220, 53, 69, 0.3)'
        ))
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        fig.add_trace(go.Scatter(
            x=[max_dd_date],
            y=[max_dd],
            mode='markers+text',
            name='Max Drawdown',
            marker=dict(color='red', size=10),
            text=[f'Max DD: {max_dd:.2f}%'],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_interactive_trades(
        self,
        equity_curve: pd.Series,
        trades: List[Any],
        title: str = "Trade Entry/Exit Points",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot interactive equity curve with trade markers using Plotly.
        
        Args:
            equity_curve: Series with portfolio values
            trades: List of trade objects
            title: Plot title
            save_path: Path to save the HTML file
            
        Returns:
            Plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for interactive charts. Install with: pip install plotly")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color=self.config.color_equity, width=2)
        ))
        
        entry_dates = []
        entry_values = []
        exit_dates = []
        exit_values = []
        
        for trade in trades:
            if hasattr(trade, 'entry_date') and hasattr(trade, 'pnl'):
                entry_idx = equity_curve.index.get_indexer([trade.entry_date], method='ffill')[0]
                if entry_idx >= 0 and entry_idx < len(equity_curve):
                    entry_dates.append(equity_curve.index[entry_idx])
                    entry_values.append(equity_curve.iloc[entry_idx])
            
            if hasattr(trade, 'exit_date') and hasattr(trade, 'pnl'):
                exit_idx = equity_curve.index.get_indexer([trade.exit_date], method='ffill')[0]
                if exit_idx >= 0 and exit_idx < len(equity_curve):
                    exit_dates.append(equity_curve.index[exit_idx])
                    exit_values.append(equity_curve.iloc[exit_idx])
        
        if entry_dates:
            fig.add_trace(go.Scatter(
                x=entry_dates,
                y=entry_values,
                mode='markers',
                name='Entry',
                marker=dict(color='green', size=12, symbol='triangle-up')
            ))
        
        if exit_dates:
            colors = ['green' if pnl >= 0 else 'red' for pnl in 
                     [t.pnl if hasattr(t, 'pnl') else 0 for t in trades[:len(exit_dates)]]]
            fig.add_trace(go.Scatter(
                x=exit_dates,
                y=exit_values,
                mode='markers',
                name='Exit',
                marker=dict(color=colors, size=12, symbol='triangle-down')
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
