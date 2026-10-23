"""
Pandas-based Optimizers for backtesting.

Implements grid search and random search optimization using pandas DataFrames.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import random

from src.backtesting.engine import Backtester, BacktestConfig
from src.backtesting.signals import TradingSignal, Signal


@dataclass
class GridSearchResult:
    """Results from a single grid search run."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    num_trades: int
    error: Optional[str] = None


class GridSearchOptimizer:
    """
    Pandas-based grid search optimizer for backtest parameters.
    
    Efficiently searches parameter space and returns results as DataFrames.
    """
    
    def __init__(self):
        self.results_df: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        
    def define_grid(self, param_grid: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Define parameter grid and create all combinations.
        
        Args:
            param_grid: Dict mapping parameter names to lists of values to test
            
        Returns:
            DataFrame with all parameter combinations
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = [[]]
        for value_list in values:
            combinations = [combo + [v] for combo in combinations for v in value_list]
        
        grid_df = pd.DataFrame(combinations, columns=keys)
        grid_df['_run_id'] = range(len(grid_df))
        
        return grid_df
    
    def _create_config_from_params(self, params: Dict[str, Any]) -> BacktestConfig:
        """Create BacktestConfig from parameters."""
        config_params = {
            'initial_capital': params.get('initial_capital', 1000000),
            'warmup_days': params.get('warmup_days', 15),
            'max_positions': int(params.get('max_positions', 3)),
            'max_position_pct': float(params.get('max_position_pct', 15.0)),
            'default_target_pct': float(params.get('target_pct', 5.0)),
            'default_stop_loss_atr': float(params.get('stop_loss_atr', 1.5)),
            'min_conviction': int(params.get('min_conviction', 6)),
        }
        return BacktestConfig(**config_params)
    
    def run_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1y",
        metric: str = 'total_return_pct',
        maximize: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run grid search across all parameter combinations.
        
        Args:
            param_grid: Parameter grid to search
            symbol: Stock symbol to test
            signals: Trading signals dict
            start_date: Backtest start date
            end_date: Backtest end date  
            period: Data period if dates not provided
            metric: Metric to optimize
            maximize: If True, maximize metric; if False, minimize
            verbose: Print progress
            
        Returns:
            DataFrame with results for all combinations
        """
        grid_df = self.define_grid(param_grid)
        param_names = [k for k in param_grid.keys()]
        
        results = []
        
        for idx, row in grid_df.iterrows():
            params = {k: row[k] for k in param_names}
            
            if verbose:
                print(f"  [{idx + 1}/{len(grid_df)}] {params}", end=' ')
            
            try:
                config = self._create_config_from_params(params)
                
                backtester = Backtester(config)
                result = backtester.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=start_date,
                    end_date=end_date,
                    period=period if not start_date else None,
                )
                
                metric_value = result.metrics.get(metric, 0)
                
                results.append({
                    '_run_id': idx,
                    '_metric_value': metric_value,
                    **{k: row[k] for k in param_names},
                    'total_return_pct': result.metrics.get('total_return_pct', 0),
                    'max_drawdown_pct': result.metrics.get('max_drawdown_pct', 0),
                    'win_rate': result.metrics.get('win_rate', 0),
                    'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                    'num_trades': len(result.trades),
                    '_error': None,
                })
                
                if verbose:
                    print(f"-> {metric_value:.2f}%")
                
            except Exception as e:
                results.append({
                    '_run_id': idx,
                    '_metric_value': -999 if maximize else 999,
                    **{k: row[k] for k in param_names},
                    'total_return_pct': -999,
                    'max_drawdown_pct': 0,
                    'win_rate': 0,
                    'sharpe_ratio': 0,
                    'num_trades': 0,
                    '_error': str(e),
                })
                if verbose:
                    print(f"-> ERROR: {e}")
        
        self.results_df = pd.DataFrame(results)
        
        if maximize:
            self.results_df = self.results_df.sort_values('_metric_value', ascending=False)
        else:
            self.results_df = self.results_df.sort_values('_metric_value', ascending=True)
        
        best_idx = self.results_df.iloc[0]['_run_id']
        best_row = self.results_df[self.results_df['_run_id'] == best_idx].iloc[0]
        
        self.best_params = {k: best_row[k] for k in param_names}
        self.best_score = best_row['_metric_value']
        self.best_metrics = {
            'total_return_pct': best_row['total_return_pct'],
            'max_drawdown_pct': best_row['max_drawdown_pct'],
            'win_rate': best_row['win_rate'],
            'sharpe_ratio': best_row['sharpe_ratio'],
            'num_trades': best_row['num_trades'],
        }
        
        return self.results_df
    
    def get_top_n(self, n: int = 5) -> pd.DataFrame:
        """Get top N parameter combinations sorted by metric."""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df.head(n).copy()
    
    def get_worst_n(self, n: int = 5) -> pd.DataFrame:
        """Get worst N parameter combinations."""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df.tail(n).copy()
    
    def get_results_df(self) -> pd.DataFrame:
        """Get full results DataFrame."""
        return self.results_df.copy() if self.results_df is not None else pd.DataFrame()
    
    def summary(self) -> str:
        """Generate summary report."""
        if self.results_df is None:
            return "No results yet. Run grid_search first."
        
        output = []
        output.append("=" * 70)
        output.append("GRID SEARCH RESULTS SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nTotal combinations tested: {len(self.results_df)}")
        output.append(f"Best {self.best_score:.4f}")
        output.append(f"Best parameters: {self.best_params}")
        
        output.append("\n" + "-" * 70)
        output.append("TOP 5 PARAMETER COMBINATIONS")
        output.append("-" * 70)
        
        for i, row in self.get_top_n(5).iterrows():
            rank = self.results_df.index.get_loc(i) + 1
            output.append(f"\nRank {rank}:")
            for key, value in self.best_params.items():
                output.append(f"  {key}: {row[key]}")
            output.append(f"  Return: {row['total_return_pct']:.2f}%")
            output.append(f"  Max DD: {row['max_drawdown_pct']:.2f}%")
            output.append(f"  Win Rate: {row['win_rate']:.1f}%")
            output.append(f"  Trades: {row['num_trades']}")
        
        return "\n".join(output)
    
    def sensitivity_analysis(self, param_name: str) -> pd.DataFrame:
        """
        Analyze impact of a single parameter on the metric.
        
        Args:
            param_name: Parameter to analyze
            
        Returns:
            DataFrame with average metric by parameter value
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        sensitivity = self.results_df.groupby(param_name).agg({
            '_metric_value': ['mean', 'std', 'min', 'max'],
            'total_return_pct': 'mean',
            'max_drawdown_pct': 'mean',
            'num_trades': 'mean',
        }).round(4)
        
        return sensitivity


def generate_weekly_signals(symbol: str, start: datetime, end: datetime,
                           conviction: int = 6, position_size: float = 8.0) -> list:
    """Generate weekly trading signals for testing."""
    signals = []
    current = start
    week_count = 0
    
    while current <= end:
        if current.weekday() == 0:
            week_count += 1
            if week_count % 2 == 1:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal=Signal.BUY,
                    conviction=conviction,
                    timestamp=current,
                    reasoning=f"Weekly signal week {week_count}",
                    position_size_pct=position_size,
                ))
            else:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal=Signal.SELL,
                    conviction=5,
                    timestamp=current,
                    reasoning="Take profit",
                ))
        current += __import__('datetime').timedelta(days=1)
    
    return signals


@dataclass
class RandomSearchResult:
    """Results from a single random search run."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    num_trades: int
    error: Optional[str] = None


class RandomSearchOptimizer:
    """
    Random search optimizer for backtest parameters.
    
    More efficient than grid search for large parameter spaces.
    Uses random sampling to explore the parameter space.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random search optimizer.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.results_df: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        
    def _sample_params(
        self, 
        param_bounds: Dict[str, Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """
        Sample a single parameter combination from bounds.
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) tuples
            
        Returns:
            Dict of sampled parameters
        """
        params = {}
        for name, (min_val, max_val) in param_bounds.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[name] = random.randint(min_val, max_val)
            elif isinstance(min_val, float) or isinstance(max_val, float):
                params[name] = random.uniform(min_val, max_val)
            else:
                choices = list(range(min_val, max_val + 1)) if isinstance(min_val, int) else [min_val, max_val]
                params[name] = random.choice(choices)
        return params
    
    def _create_config_from_params(self, params: Dict[str, Any]) -> BacktestConfig:
        """Create BacktestConfig from parameters."""
        config_params = {
            'initial_capital': params.get('initial_capital', 1000000),
            'warmup_days': int(params.get('warmup_days', 15)),
            'max_positions': int(params.get('max_positions', 3)),
            'max_position_pct': float(params.get('max_position_pct', 15.0)),
            'default_target_pct': float(params.get('target_pct', 5.0)),
            'default_stop_loss_atr': float(params.get('stop_loss_atr', 1.5)),
            'min_conviction': int(params.get('min_conviction', 6)),
        }
        return BacktestConfig(**config_params)
    
    def run_random_search(
        self,
        param_bounds: Dict[str, Tuple[Any, Any]],
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        n_iterations: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1y",
        metric: str = 'total_return_pct',
        maximize: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run random search across parameter space.
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) tuples
            symbol: Stock symbol to test
            signals: Trading signals dict
            n_iterations: Number of random samples to test
            start_date: Backtest start date
            end_date: Backtest end date
            period: Data period if dates not provided
            metric: Metric to optimize
            maximize: If True, maximize metric; if False, minimize
            verbose: Print progress
            
        Returns:
            DataFrame with results for all samples
        """
        param_names = list(param_bounds.keys())
        results = []
        
        for i in range(n_iterations):
            params = self._sample_params(param_bounds)
            
            if verbose:
                print(f"  [{i + 1}/{n_iterations}] {params}", end=' ')
            
            try:
                config = self._create_config_from_params(params)
                
                backtester = Backtester(config)
                result = backtester.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=start_date,
                    end_date=end_date,
                    period=period if not start_date else None,
                )
                
                metric_value = result.metrics.get(metric, 0)
                
                result_row = {
                    '_iteration': i,
                    '_metric_value': metric_value,
                    **{k: params[k] for k in param_names},
                    'total_return_pct': result.metrics.get('total_return_pct', 0),
                    'max_drawdown_pct': result.metrics.get('max_drawdown_pct', 0),
                    'win_rate': result.metrics.get('win_rate', 0),
                    'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                    'num_trades': len(result.trades),
                    '_error': None,
                }
                
                results.append(result_row)
                
                if verbose:
                    print(f"-> {metric_value:.2f}%")
                
            except Exception as e:
                results.append({
                    '_iteration': i,
                    '_metric_value': -999 if maximize else 999,
                    **{k: params[k] for k in param_names},
                    'total_return_pct': -999,
                    'max_drawdown_pct': 0,
                    'win_rate': 0,
                    'sharpe_ratio': 0,
                    'num_trades': 0,
                    '_error': str(e),
                })
                if verbose:
                    print(f"-> ERROR: {e}")
        
        self.results_df = pd.DataFrame(results)
        
        if maximize:
            self.results_df = self.results_df.sort_values('_metric_value', ascending=False)
        else:
            self.results_df = self.results_df.sort_values('_metric_value', ascending=True)
        
        best_idx = self.results_df.iloc[0]['_iteration']
        best_row = self.results_df[self.results_df['_iteration'] == best_idx].iloc[0]
        
        self.best_params = {k: best_row[k] for k in param_names}
        self.best_score = best_row['_metric_value']
        self.best_metrics = {
            'total_return_pct': best_row['total_return_pct'],
            'max_drawdown_pct': best_row['max_drawdown_pct'],
            'win_rate': best_row['win_rate'],
            'sharpe_ratio': best_row['sharpe_ratio'],
            'num_trades': best_row['num_trades'],
        }
        
        return self.results_df
    
    def get_top_n(self, n: int = 5) -> pd.DataFrame:
        """Get top N parameter combinations."""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df.head(n).copy()
    
    def get_results_df(self) -> pd.DataFrame:
        """Get full results DataFrame."""
        return self.results_df.copy() if self.results_df is not None else pd.DataFrame()
    
    def summary(self) -> str:
        """Generate summary report."""
        if self.results_df is None:
            return "No results yet. Run random_search first."
        
        output = []
        output.append("=" * 70)
        output.append("RANDOM SEARCH RESULTS SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nTotal iterations: {len(self.results_df)}")
        output.append(f"Best metric value: {self.best_score:.4f}")
        output.append(f"Best parameters: {self.best_params}")
        
        output.append("\n" + "-" * 70)
        output.append("TOP 5 PARAMETER COMBINATIONS")
        output.append("-" * 70)
        
        for _, row in self.get_top_n(5).iterrows():
            output.append(f"\nIteration {int(row['_iteration'])}:")
            for key in self.best_params.keys():
                output.append(f"  {key}: {row[key]}")
            output.append(f"  Return: {row['total_return_pct']:.2f}%")
            output.append(f"  Max DD: {row['max_drawdown_pct']:.2f}%")
            output.append(f"  Win Rate: {row['win_rate']:.1f}%")
            output.append(f"  Trades: {row['num_trades']}")
        
        return "\n".join(output)
    
    def sensitivity_analysis(self, param_name: str) -> pd.DataFrame:
        """
        Analyze impact of a single parameter on the metric.
        
        Args:
            param_name: Parameter to analyze
            
        Returns:
            DataFrame with correlation and statistics
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        param_data = self.results_df[[param_name, '_metric_value']].dropna()
        
        if len(param_data) < 2:
            return pd.DataFrame()
        
        correlation = param_data[param_name].corr(param_data['_metric_value'])
        
        analysis = pd.DataFrame({
            'parameter': [param_name],
            'correlation': [correlation],
            'param_mean': [param_data[param_name].mean()],
            'param_std': [param_data[param_name].std()],
            'metric_mean': [param_data['_metric_value'].mean()],
            'metric_std': [param_data['_metric_value'].std()],
        })
        
        return analysis


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for robust parameter optimization.
    
    Trains on in-sample periods and validates on out-of-sample periods
    to reduce overfitting and provide more realistic performance estimates.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.6,
        n_folds: int = 3,
        step_size_days: int = 30,
    ):
        """
        Initialize WalkForwardAnalyzer.
        
        Args:
            train_ratio: Ratio of training period to total window
            n_folds: Number of walk-forward windows
            step_size_days: Days to step between windows
        """
        self.train_ratio = train_ratio
        self.n_folds = n_folds
        self.step_size_days = step_size_days
        self.results_df: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_oos_score: Optional[float] = None
        
    def _create_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create walk-forward windows.
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        total_days = (end_date - start_date).days
        window_days = total_days // self.n_folds
        
        train_days = int(window_days * self.train_ratio)
        test_days = window_days - train_days
        
        windows = []
        current = start_date
        
        for _ in range(self.n_folds):
            train_start = current
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)
            
            if test_end <= end_date:
                windows.append((train_start, train_end, test_start, test_end))
            
            current = current + timedelta(days=self.step_size_days)
        
        return windows
    
    def _optimize_on_train(
        self,
        param_grid: Dict[str, List[Any]],
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        train_start: datetime,
        train_end: datetime,
    ) -> Dict[str, Any]:
        """
        Find best parameters on training period.
        
        Returns:
            Best parameters dict
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = [[]]
        for value_list in values:
            combinations = [combo + [v] for combo in combinations for v in value_list]
        
        best_score = float('-inf')
        best_params = {k: v[0] for k, v in param_grid.items()}
        
        for combo in combinations[:20]:
            params = dict(zip(keys, combo))
            
            config_params = {
                'initial_capital': params.get('initial_capital', 1000000),
                'warmup_days': 10,
                'max_positions': int(params.get('max_positions', 3)),
                'max_position_pct': float(params.get('max_position_pct', 15.0)),
                'default_target_pct': float(params.get('target_pct', 5.0)),
                'default_stop_loss_atr': float(params.get('stop_loss_atr', 1.5)),
                'min_conviction': int(params.get('min_conviction', 6)),
            }
            
            try:
                config = BacktestConfig(**config_params)
                backtester = Backtester(config)
                result = backtester.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=train_start,
                    end_date=train_end,
                )
                
                score = result.metrics.get('total_return_pct', -999)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception:
                continue
        
        return best_params
    
    def run_walk_forward(
        self,
        param_grid: Dict[str, List[Any]],
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        start_date: datetime,
        end_date: datetime,
        metric: str = 'total_return_pct',
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis.
        
        Args:
            param_grid: Parameter grid to search
            symbol: Stock symbol
            signals: Trading signals
            start_date: Analysis start date
            end_date: Analysis end date
            metric: Metric to optimize
            verbose: Print progress
            
        Returns:
            DataFrame with walk-forward results
        """
        windows = self._create_windows(start_date, end_date)
        
        if verbose:
            print(f"Walk-Forward Analysis: {len(windows)} windows")
        
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if verbose:
                print(f"\n  Window {i+1}/{len(windows)}:")
                print(f"    Train: {train_start.date()} to {train_end.date()}")
                print(f"    Test:  {test_start.date()} to {test_end.date()}")
            
            best_params = self._optimize_on_train(
                param_grid, symbol, signals, train_start, train_end
            )
            
            if verbose:
                print(f"    Best train params: {best_params}")
            
            config_params = {
                'initial_capital': 1000000,
                'warmup_days': 10,
                'max_positions': int(best_params.get('max_positions', 3)),
                'max_position_pct': float(best_params.get('max_position_pct', 15.0)),
                'default_target_pct': float(best_params.get('target_pct', 5.0)),
                'default_stop_loss_atr': float(best_params.get('stop_loss_atr', 1.5)),
                'min_conviction': int(best_params.get('min_conviction', 6)),
            }
            
            try:
                config = BacktestConfig(**config_params)
                backtester = Backtester(config)
                test_result = backtester.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=test_start,
                    end_date=test_end,
                )
                
                train_result = None
                train_config = BacktestConfig(**config_params)
                train_bt = Backtester(train_config)
                train_result = train_bt.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=train_start,
                    end_date=train_end,
                )
                
                oos_metric = test_result.metrics.get(metric, 0)
                train_metric = train_result.metrics.get(metric, 0) if train_result else 0
                
                results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_return': train_metric,
                    'test_return': oos_metric,
                    'train_test_diff': oos_metric - train_metric,
                    **{f'param_{k}': v for k, v in best_params.items()},
                })
                
                if verbose:
                    print(f"    Train return: {train_metric:.2f}%")
                    print(f"    Test return:  {oos_metric:.2f}%")
                    print(f"    OOS Gap:     {oos_metric - train_metric:.2f}%")
                    
            except Exception as e:
                if verbose:
                    print(f"    ERROR: {e}")
                results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_return': 0,
                    'test_return': -999,
                    'train_test_diff': -999,
                    **{f'param_{k}': v for k, v in best_params.items()},
                })
        
        self.results_df = pd.DataFrame(results)
        
        avg_oos = self.results_df['test_return'].mean()
        avg_train = self.results_df['train_return'].mean()
        
        self.best_oos_score = avg_oos
        
        self.best_params = {}
        for col in self.results_df.columns:
            if col.startswith('param_'):
                param_name = col.replace('param_', '')
                self.best_params[param_name] = self.results_df[col].mode().iloc[0]
        
        return self.results_df
    
    def summary(self) -> str:
        """Generate walk-forward summary report."""
        if self.results_df is None:
            return "No results yet. Run walk_forward first."
        
        output = []
        output.append("=" * 70)
        output.append("WALK-FORWARD ANALYSIS SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nWindows: {len(self.results_df)}")
        output.append(f"Average Train Return: {self.results_df['train_return'].mean():.2f}%")
        output.append(f"Average Test Return:  {self.results_df['test_return'].mean():.2f}%")
        output.append(f"Average OOS Gap:      {(self.results_df['test_return'] - self.results_df['train_return']).mean():.2f}%")
        
        output.append("\n" + "-" * 70)
        output.append("BY WINDOW RESULTS")
        output.append("-" * 70)
        
        for _, row in self.results_df.iterrows():
            output.append(f"\nWindow {int(row['window'])}:")
            output.append(f"  Train: {row['train_return']:.2f}%")
            output.append(f"  Test:  {row['test_return']:.2f}%")
            output.append(f"  Gap:   {row['train_test_diff']:.2f}%")
        
        return "\n".join(output)


class SensitivityAnalyzer:
    """
    Sensitivity Analysis for identifying critical strategy parameters.
    
    Analyzes how parameter variations impact strategy performance to identify
    which parameters have the greatest influence on returns and risk.
    """
    
    def __init__(self, n_steps: int = 5):
        """
        Initialize sensitivity analyzer.
        
        Args:
            n_steps: Number of steps for parameter variation
        """
        self.n_steps = n_steps
        self.results_df: Optional[pd.DataFrame] = None
        self.impact_scores: Optional[Dict[str, float]] = None
        
    def run_analysis(
        self,
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        base_params: Dict[str, Any],
        param_ranges: Dict[str, Tuple[Any, Any]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1y",
        metric: str = 'total_return_pct',
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis across parameter ranges.
        
        Args:
            symbol: Stock symbol
            signals: Trading signals
            base_params: Base parameter values
            param_ranges: Dict mapping params to (min, max) tuples
            start_date: Backtest start date
            end_date: Backtest end date
            period: Data period if dates not provided
            metric: Metric to evaluate
            verbose: Print progress
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for param_name, (min_val, max_val) in param_ranges.items():
            if verbose:
                print(f"  Analyzing {param_name}...")
            
            if isinstance(min_val, int) and isinstance(max_val, int):
                step = max(1, (max_val - min_val) // self.n_steps)
                values = list(range(min_val, max_val + 1, step))
            else:
                step = (max_val - min_val) / self.n_steps
                values = [min_val + i * step for i in range(self.n_steps + 1)]
            
            for value in values:
                test_params = base_params.copy()
                test_params[param_name] = value
                
                try:
                    config_params = {
                        'initial_capital': test_params.get('initial_capital', 1000000),
                        'warmup_days': 10,
                        'max_positions': int(test_params.get('max_positions', 3)),
                        'max_position_pct': float(test_params.get('max_position_pct', 15.0)),
                        'default_target_pct': float(test_params.get('target_pct', 5.0)),
                        'default_stop_loss_atr': float(test_params.get('stop_loss_atr', 1.5)),
                        'min_conviction': int(test_params.get('min_conviction', 6)),
                    }
                    
                    config = BacktestConfig(**config_params)
                    backtester = Backtester(config)
                    result = backtester.run(
                        symbols=[symbol],
                        signals=signals,
                        start_date=start_date,
                        end_date=end_date,
                        period=period if not start_date else None,
                    )
                    
                    metric_value = result.metrics.get(metric, 0)
                    
                    results.append({
                        'parameter': param_name,
                        'value': value,
                        'metric_value': metric_value,
                        'return_pct': result.metrics.get('total_return_pct', 0),
                        'max_drawdown_pct': result.metrics.get('max_drawdown_pct', 0),
                        'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                        'num_trades': len(result.trades),
                    })
                    
                except Exception as e:
                    results.append({
                        'parameter': param_name,
                        'value': value,
                        'metric_value': -999,
                        'return_pct': -999,
                        'max_drawdown_pct': 0,
                        'sharpe_ratio': 0,
                        'num_trades': 0,
                    })
        
        self.results_df = pd.DataFrame(results)
        
        self._calculate_impact_scores()
        
        if verbose:
            print(f"  Analyzed {len(self.results_df)} parameter combinations")
        
        return self.results_df
    
    def _calculate_impact_scores(self):
        """Calculate impact scores for each parameter."""
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        scores = {}
        
        for param in self.results_df['parameter'].unique():
            param_data = self.results_df[self.results_df['parameter'] == param]
            
            metric_range = param_data['metric_value'].max() - param_data['metric_value'].min()
            metric_std = param_data['metric_value'].std()
            
            dd_range = param_data['max_drawdown_pct'].max() - param_data['max_drawdown_pct'].min()
            dd_std = param_data['max_drawdown_pct'].std()
            
            scores[param] = {
                'metric_range': metric_range,
                'metric_std': metric_std,
                'dd_range': dd_range,
                'dd_std': dd_std,
                'impact_score': (abs(metric_range) + abs(dd_range)) / 2,
            }
        
        self.impact_scores = scores
    
    def get_impact_summary(self) -> pd.DataFrame:
        """Get summary of parameter impact scores."""
        if self.impact_scores is None:
            return pd.DataFrame()
        
        impact_df = pd.DataFrame([
            {'parameter': k, **v} for k, v in self.impact_scores.items()
        ])
        return impact_df.sort_values('impact_score', ascending=False)
    
    def get_parameter_sensitivity(self, param_name: str) -> pd.DataFrame:
        """Get sensitivity data for a specific parameter."""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df[self.results_df['parameter'] == param_name].copy()
    
    def summary(self) -> str:
        """Generate sensitivity analysis summary."""
        if self.results_df is None:
            return "No results yet. Run analysis first."
        
        output = []
        output.append("=" * 70)
        output.append("SENSITIVITY ANALYSIS SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nParameters analyzed: {len(self.results_df['parameter'].unique())}")
        output.append(f"Total combinations tested: {len(self.results_df)}")
        
        output.append("\n" + "-" * 70)
        output.append("PARAMETER IMPACT RANKING (Most to Least Sensitive)")
        output.append("-" * 70)
        
        impact_summary = self.get_impact_summary()
        for i, row in impact_summary.iterrows():
            output.append(f"\n{row['parameter']}:")
            output.append(f"  Impact Score: {row['impact_score']:.4f}")
            output.append(f"  Metric Range: {row['metric_range']:.4f}")
            output.append(f"  Metric Std:   {row['metric_std']:.4f}")
            output.append(f"  DD Range:     {row['dd_range']:.4f}")
        
        output.append("\n" + "-" * 70)
        output.append("OPTIMAL VALUES BY PARAMETER")
        output.append("-" * 70)
        
        for param in impact_summary['parameter']:
            param_data = self.get_parameter_sensitivity(param)
            if len(param_data) > 0:
                best_idx = param_data['metric_value'].idxmax()
                best_row = param_data.loc[best_idx]
                output.append(f"\n{param}: {best_row['value']} -> {best_row['metric_value']:.4f}")
        
        return "\n".join(output)


class StatisticalSignificanceTester:
    """
    Statistical Significance Testing for strategy results.
    
    Uses t-tests and confidence intervals to determine if strategy performance
    is statistically significant vs random chance or benchmark.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical significance tester.
        
        Args:
            confidence_level: Confidence level for tests (0.0-1.0)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results: Optional[Dict[str, Any]] = None
        
    def run_t_test(
        self,
        sample1_returns: List[float],
        sample2_returns: Optional[List[float]] = None,
        baseline: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run t-test on return samples.
        
        Args:
            sample1_returns: First sample of returns
            sample2_returns: Second sample for paired/independent t-test (optional)
            baseline: Baseline return to test against (default: 0)
            
        Returns:
            Dict with t-statistic, p-value, and significance
        """
        if sample2_returns is None:
            sample2_returns = [baseline] * len(sample1_returns)
        
        n1 = len(sample1_returns)
        n2 = len(sample2_returns)
        
        mean1 = np.mean(sample1_returns)
        mean2 = np.mean(sample2_returns)
        
        std1 = np.std(sample1_returns, ddof=1) if n1 > 1 else 0
        std2 = np.std(sample2_returns, ddof=1) if n2 > 1 else 0
        
        se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        se = max(se, 1e-10)
        
        t_stat = (mean1 - mean2) / se
        
        df = min(n1, n2) - 1
        df = max(1, df)
        
        p_value = self._t_distribution_cdf(abs(t_stat), df)
        p_value = 2 * (1 - p_value)
        
        is_significant = p_value < self.alpha
        
        ci_lower, ci_upper = self._confidence_interval(sample1_returns, self.confidence_level)
        
        self.results = {
            'mean1': mean1,
            'mean2': mean2,
            'difference': mean1 - mean2,
            'std1': std1,
            'std2': std2,
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'is_significant': is_significant,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n1': n1,
            'n2': n2,
        }
        
        return self.results
    
    def _t_distribution_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF using normal approximation."""
        x = df / (df + t**2)
        return 1 - 0.5 * (x ** (df / 2))
    
    def _confidence_interval(self, returns: List[float], level: float) -> Tuple[float, float]:
        """Calculate confidence interval for returns."""
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        se = std / np.sqrt(n)
        
        z_value = 1.96 if level == 0.95 else 1.645
        ci_lower = mean - z_value * se
        ci_upper = mean + z_value * se
        
        return ci_lower, ci_upper
    
    def test_vs_benchmark(
        self,
        strategy_returns: List[float],
        benchmark_returns: List[float],
    ) -> Dict[str, Any]:
        """
        Test if strategy significantly outperforms benchmark.
        
        Args:
            strategy_returns: List of strategy returns
            benchmark_returns: List of benchmark returns
            
        Returns:
            Dict with alpha, beta, t-statistic, p-value
        """
        n = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = list(strategy_returns[:n])
        benchmark_returns = list(benchmark_returns[:n])
        
        excess_returns = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        se_excess = std_excess / np.sqrt(n)
        
        t_stat = mean_excess / se_excess if se_excess > 0 else 0
        
        p_value = self._t_distribution_cdf(abs(t_stat), n - 1)
        p_value = 2 * (1 - p_value)
        
        alpha = mean_excess
        beta = np.corrcoef(strategy_returns, benchmark_returns)[0, 1] if n > 1 else 0
        
        is_significant = p_value < self.alpha
        
        self.results = {
            'alpha': alpha,
            'beta': beta,
            'mean_excess_return': mean_excess,
            'std_excess_return': std_excess,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'n_observations': n,
            'confidence_level': self.confidence_level,
        }
        
        return self.results
    
    def bootstrap_confidence_interval(
        self,
        returns: List[float],
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            returns: List of returns
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dict with CI bounds and percentiles
        """
        np.random.seed(42)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 + self.confidence_level) / 2 * 100)
        
        return {
            'mean': np.mean(returns),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(bootstrap_means),
            'n_bootstrap': n_bootstrap,
            'confidence_level': self.confidence_level,
        }
    
    def summary(self) -> str:
        """Generate statistical significance summary."""
        if self.results is None:
            return "No results yet. Run test first."
        
        output = []
        output.append("=" * 70)
        output.append("STATISTICAL SIGNIFICANCE TEST SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nConfidence Level: {self.confidence_level * 100:.0f}%")
        output.append(f"Alpha (significance level): {self.alpha * 100:.0f}%")
        
        if 'mean1' in self.results:
            output.append(f"\nSample Statistics:")
            output.append(f"  Mean 1: {self.results['mean1']:.4f}")
            output.append(f"  Mean 2: {self.results['mean2']:.4f}")
            output.append(f"  Difference: {self.results['difference']:.4f}")
            output.append(f"  Std 1: {self.results['std1']:.4f}")
            output.append(f"  Std 2: {self.results['std2']:.4f}")
            
            output.append(f"\nT-Test Results:")
            output.append(f"  t-statistic: {self.results['t_statistic']:.4f}")
            output.append(f"  p-value: {self.results['p_value']:.4f}")
            output.append(f"  Degrees of freedom: {self.results['degrees_of_freedom']}")
            
            output.append(f"\nConfidence Interval:")
            output.append(f"  Lower: {self.results['ci_lower']:.4f}")
            output.append(f"  Upper: {self.results['ci_upper']:.4f}")
        
        elif 'alpha' in self.results:
            output.append(f"\nAlpha (excess return): {self.results['alpha']:.4f}")
            output.append(f"Beta (correlation): {self.results['beta']:.4f}")
            output.append(f"Mean Excess Return: {self.results['mean_excess_return']:.4f}")
            output.append(f"t-statistic: {self.results['t_statistic']:.4f}")
            output.append(f"p-value: {self.results['p_value']:.4f}")
            output.append(f"Observations: {self.results['n_observations']}")
        
        sig_status = "SIGNIFICANT" if self.results['is_significant'] else "NOT SIGNIFICANT"
        output.append(f"\nConclusion: {sig_status}")
        
        return "\n".join(output)


class OutOfSampleValidator:
    """
    Out-of-Sample Validation for detecting overfitting.
    
    Tests optimized parameters on completely held-out data to ensure
    the strategy generalizes beyond the training period.
    """
    
    def __init__(self, test_ratio: float = 0.3, seed: Optional[int] = None):
        """
        Initialize validator.
        
        Args:
            test_ratio: Fraction of data to hold out for testing (0.0-1.0)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.test_ratio = test_ratio
        self.results: Optional[Dict[str, Any]] = None
        
    def validate(
        self,
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        start_date: datetime,
        end_date: datetime,
        param_grid: Dict[str, List[Any]],
        metric: str = 'total_return_pct',
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run out-of-sample validation.
        
        Args:
            symbol: Stock symbol
            signals: Trading signals
            start_date: Data start date
            end_date: Data end date
            param_grid: Parameters to optimize
            metric: Metric to evaluate
            verbose: Print progress
            
        Returns:
            Dict with in-sample, out-of-sample, and overfitting metrics
        """
        total_days = (end_date - start_date).days
        train_days = int(total_days * (1 - self.test_ratio))
        
        train_end = start_date + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        
        if verbose:
            print(f"OOS Validation:")
            print(f"  Train: {start_date.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {end_date.date()}")
        
        if verbose:
            print(f"\n  Optimizing on training data...")
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [[]]
        for value_list in values:
            combinations = [combo + [v] for combo in combinations for v in value_list]
        
        best_score = float('-inf')
        best_params = {k: v[0] for k, v in param_grid.items()}
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            config_params = {
                'initial_capital': 1000000,
                'warmup_days': 10,
                'max_positions': int(params.get('max_positions', 3)),
                'max_position_pct': float(params.get('max_position_pct', 15.0)),
                'default_target_pct': float(params.get('target_pct', 5.0)),
                'default_stop_loss_atr': float(params.get('stop_loss_atr', 1.5)),
                'min_conviction': int(params.get('min_conviction', 6)),
            }
            
            try:
                config = BacktestConfig(**config_params)
                backtester = Backtester(config)
                result = backtester.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=start_date,
                    end_date=train_end,
                )
                score = result.metrics.get(metric, -999)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception:
                continue
        
        if verbose:
            print(f"  Best train params: {best_params}")
            print(f"  Best train {metric}: {best_score:.4f}")
            print(f"\n  Validating on out-of-sample data...")
        
        config_params = {
            'initial_capital': 1000000,
            'warmup_days': 10,
            'max_positions': int(best_params.get('max_positions', 3)),
            'max_position_pct': float(best_params.get('max_position_pct', 15.0)),
            'default_target_pct': float(best_params.get('target_pct', 5.0)),
            'default_stop_loss_atr': float(best_params.get('stop_loss_atr', 1.5)),
            'min_conviction': int(best_params.get('min_conviction', 6)),
        }
        
        train_config = BacktestConfig(**config_params)
        train_bt = Backtester(train_config)
        train_result = train_bt.run(
            symbols=[symbol],
            signals=signals,
            start_date=start_date,
            end_date=train_end,
        )
        
        test_config = BacktestConfig(**config_params)
        test_bt = Backtester(test_config)
        test_result = test_bt.run(
            symbols=[symbol],
            signals=signals,
            start_date=test_start,
            end_date=end_date,
        )
        
        train_metric = train_result.metrics.get(metric, 0)
        test_metric = test_result.metrics.get(metric, 0)
        oos_gap = test_metric - train_metric
        oos_gap_pct = (oos_gap / abs(train_metric)) * 100 if train_metric != 0 else 0
        
        overfitting_ratio = abs(test_metric) / abs(train_metric) if train_metric != 0 else 0
        
        is_overfit = oos_gap_pct < -20 or overfitting_ratio < 0.7
        
        self.results = {
            'train_start': start_date,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': end_date,
            'best_params': best_params,
            'train_metric': train_metric,
            'test_metric': test_metric,
            'oos_gap': oos_gap,
            'oos_gap_pct': oos_gap_pct,
            'overfitting_ratio': overfitting_ratio,
            'is_overfit': is_overfit,
            'train_metrics': train_result.metrics,
            'test_metrics': test_result.metrics,
            'train_trades': len(train_result.trades),
            'test_trades': len(test_result.trades),
        }
        
        if verbose:
            print(f"  Train {metric}: {train_metric:.4f}")
            print(f"  Test {metric}:  {test_metric:.4f}")
            print(f"  OOS Gap:       {oos_gap:.4f} ({oos_gap_pct:.1f}%)")
            print(f"  Overfit Ratio: {overfitting_ratio:.3f}")
            print(f"  Overfit Flag:  {'YES' if is_overfit else 'No'}")
        
        return self.results
    
    def summary(self) -> str:
        """Generate validation summary."""
        if self.results is None:
            return "No results yet. Run validate() first."
        
        r = self.results
        output = []
        output.append("=" * 70)
        output.append("OUT-OF-SAMPLE VALIDATION SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nPeriods:")
        output.append(f"  Train: {r['train_start'].date()} to {r['train_end'].date()}")
        output.append(f"  Test:  {r['test_start'].date()} to {r['test_end'].date()}")
        
        output.append(f"\nBest Parameters: {r['best_params']}")
        
        output.append(f"\nPerformance:")
        output.append(f"  Train metric:  {r['train_metric']:.4f}")
        output.append(f"  Test metric:   {r['test_metric']:.4f}")
        output.append(f"  OOS Gap:       {r['oos_gap']:.4f} ({r['oos_gap_pct']:.1f}%)")
        
        output.append(f"\nOverfitting Analysis:")
        output.append(f"  Overfit Ratio: {r['overfitting_ratio']:.3f}")
        output.append(f"  Overfit Flag: {'YES - Strategy may be overfit' if r['is_overfit'] else 'NO - Strategy generalizes well'}")
        
        output.append(f"\nTrade Counts:")
        output.append(f"  Train trades: {r['train_trades']}")
        output.append(f"  Test trades:  {r['test_trades']}")
        
        return "\n".join(output)


class OverfittingDetector:
    """
    Overfitting Detection for strategy validation.
    
    Uses multiple methods to detect and quantify overfitting:
    - Cross-validation gap analysis
    - Parameter stability testing
    - Information criterion
    - Train/test performance comparison
    """
    
    def __init__(
        self,
        overfit_threshold: float = 0.2,
        min_stability_score: float = 0.5,
    ):
        """
        Initialize overfitting detector.
        
        Args:
            overfit_threshold: Maximum allowed OOS gap ratio (default 20%)
            min_stability_score: Minimum stability score (0.0-1.0)
        """
        self.overfit_threshold = overfit_threshold
        self.min_stability_score = min_stability_score
        self.results: Optional[Dict[str, Any]] = None
        
    def detect(
        self,
        in_sample_metrics: Dict[str, float],
        out_of_sample_metrics: Dict[str, float],
        param_variations: Optional[List[Dict[str, Any]]] = None,
        param_sensitivity_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect overfitting using multiple indicators.
        
        Args:
            in_sample_metrics: Metrics from in-sample/backtest period
            out_of_sample_metrics: Metrics from out-of-sample validation
            param_variations: List of parameter variations tested
            param_sensitivity_scores: Dict of sensitivity scores by parameter
            
        Returns:
            Dict with overfitting indicators and overall verdict
        """
        results = {}
        
        is_return = 'total_return_pct' in in_sample_metrics
        is_sharpe = 'sharpe_ratio' in in_sample_metrics
        
        metric_key = 'total_return_pct' if is_return else ('sharpe_ratio' if is_sharpe else list(in_sample_metrics.keys())[0])
        
        is_metric = in_sample_metrics.get(metric_key, 0)
        os_metric = out_of_sample_metrics.get(metric_key, 0)
        
        if abs(is_metric) > 0.001:
            oos_gap_ratio = (os_metric - is_metric) / abs(is_metric)
        else:
            oos_gap_ratio = -1.0 if os_metric < 0 else 1.0
        
        results['oos_gap_ratio'] = oos_gap_ratio
        results['oos_gap_pct'] = (os_metric - is_metric) * 100
        
        results['is_overfit_gap'] = abs(oos_gap_ratio) > self.overfit_threshold
        
        train_mdd = abs(in_sample_metrics.get('max_drawdown_pct', 0))
        test_mdd = abs(out_of_sample_metrics.get('max_drawdown_pct', 0))
        mdd_ratio = train_mdd / test_mdd if test_mdd > 0 else 1.0
        results['mdd_ratio'] = mdd_ratio
        results['is_overfit_mdd'] = mdd_ratio > 1.5 or mdd_ratio < 0.5
        
        stability_score = 1.0
        if param_sensitivity_scores:
            max_sensitivity = max(param_sensitivity_scores.values()) if param_sensitivity_scores else 0
            stability_score = 1.0 / (1.0 + max_sensitivity)
        results['stability_score'] = stability_score
        results['is_overfit_stability'] = stability_score < self.min_stability_score
        
        win_rate_is = in_sample_metrics.get('win_rate', 0)
        win_rate_os = out_of_sample_metrics.get('win_rate', 0)
        wr_diff = abs(win_rate_is - win_rate_os)
        results['win_rate_diff'] = wr_diff
        results['is_overfit_win_rate'] = wr_diff > 0.15
        
        overfit_indicators = [
            results['is_overfit_gap'],
            results.get('is_overfit_mdd', False),
            results['is_overfit_stability'],
            results['is_overfit_win_rate'],
        ]
        overfit_score = sum(overfit_indicators) / len(overfit_indicators)
        results['overfit_score'] = overfit_score
        
        results['is_overfit'] = (
            results['is_overfit_gap'] or 
            results['is_overfit_stability'] or 
            overfit_score > 0.5
        )
        
        results['in_sample_metrics'] = in_sample_metrics
        results['out_of_sample_metrics'] = out_of_sample_metrics
        
        self.results = results
        return results
    
    def check_parameter_stability(
        self,
        optimization_results: pd.DataFrame,
        metric_col: str = '_metric_value',
    ) -> Dict[str, Any]:
        """
        Check parameter stability across optimization runs.
        
        Args:
            optimization_results: DataFrame with optimization results
            metric_col: Column name for metric values
            
        Returns:
            Dict with stability metrics
        """
        if optimization_results is None or len(optimization_results) == 0:
            return {}
        
        param_cols = [c for c in optimization_results.columns if c.startswith('_') or c in ['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio']]
        param_cols = [c for c in optimization_results.columns if c not in [metric_col, '_run_id', '_iteration']]
        
        stability_metrics = {}
        
        for col in param_cols:
            if col in optimization_results.columns:
                top_10_pct = optimization_results.head(max(1, len(optimization_results) // 10))
                if len(top_10_pct) > 1:
                    stability_metrics[col] = {
                        'top10_std': top_10_pct[col].std(),
                        'top10_mean': top_10_pct[col].mean(),
                        'top10_range': top_10_pct[col].max() - top_10_pct[col].min(),
                        'is_stable': top_10_pct[col].std() / (abs(top_10_pct[col].mean()) + 0.001) < 0.2,
                    }
        
        overall_stable = all(
            m.get('is_stable', False) for m in stability_metrics.values()
        )
        
        return {
            'parameter_stability': stability_metrics,
            'overall_stable': overall_stable,
        }
    
    def calculate_information_ratio(
        self,
        in_sample_metric: float,
        out_of_sample_metric: float,
        n_params: int,
    ) -> Dict[str, float]:
        """
        Calculate adjusted performance metrics that penalize complexity.
        
        Args:
            in_sample_metric: In-sample metric value
            out_of_sample_metric: Out-of-sample metric value
            n_params: Number of parameters optimized
            
        Returns:
            Dict with adjusted metrics
        """
        ic_ratio = out_of_sample_metric / abs(in_sample_metric) if in_sample_metric != 0 else 0
        
        aic_penalty = 2 * n_params
        bic_penalty = n_params * np.log(10)
        
        adjusted_oos = out_of_sample_metric - aic_penalty * abs(out_of_sample_metric) * 0.1
        
        return {
            'information_coefficient_ratio': ic_ratio,
            'oos_adjusted': adjusted_oos,
            'aic_penalty': aic_penalty,
            'bic_penalty': bic_penalty,
        }
    
    def summary(self) -> str:
        """Generate overfitting detection summary."""
        if self.results is None:
            return "No results yet. Run detect() first."
        
        r = self.results
        output = []
        output.append("=" * 70)
        output.append("OVERFITTING DETECTION SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nPerformance Comparison:")
        output.append(f"  In-Sample Return:  {r['in_sample_metrics'].get('total_return_pct', 0):.4f}")
        output.append(f"  Out-of-Sample Ret: {r['out_of_sample_metrics'].get('total_return_pct', 0):.4f}")
        output.append(f"  OOS Gap:          {r['oos_gap_pct']:.2f}%")
        output.append(f"  OOS Gap Ratio:    {r['oos_gap_ratio']:.4f}")
        
        output.append(f"\nOverfitting Indicators:")
        output.append(f"  Gap-based:        {'YES' if r['is_overfit_gap'] else 'No'}")
        output.append(f"  Stability-based:  {'YES' if r['is_overfit_stability'] else 'No'}")
        output.append(f"  Win Rate diff:    {r['win_rate_diff']:.2%} ({'YES' if r['is_overfit_win_rate'] else 'No'})")
        output.append(f"  Overfit Score:    {r['overfit_score']:.2f}")
        
        output.append(f"\nStability Metrics:")
        output.append(f"  Stability Score: {r['stability_score']:.4f}")
        output.append(f"  MDD Ratio:       {r['mdd_ratio']:.4f}")
        
        verdict = "OVERFIT" if r['is_overfit'] else "ROBUST"
        output.append(f"\nVERDICT: {verdict}")
        
        if r['is_overfit']:
            output.append("Recommendations:")
            output.append("  - Reduce number of optimized parameters")
            output.append("  - Use simpler strategy rules")
            output.append("  - Increase training data")
            output.append("  - Use walk-forward optimization")
        
        return "\n".join(output)


class ParameterConfigManager:
    """
    Parameter Configuration Manager for saving and loading best parameters.
    
    Saves optimized parameters to JSON config files and loads them
    for live trading or further backtesting.
    """
    
    def __init__(self, config_dir: str = "config/optimization"):
        """
        Initialize parameter config manager.
        
        Args:
            config_dir: Directory to save/load config files
        """
        self.config_dir = config_dir
        self.current_params: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        
    def save_params(
        self,
        params: Dict[str, Any],
        strategy_name: str = "default",
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save parameters to JSON config file.
        
        Args:
            params: Parameter dictionary to save
            strategy_name: Name of the strategy
            metrics: Performance metrics dictionary
            metadata: Additional metadata (dates, notes, etc.)
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        import json
        import os
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_name}_params_{timestamp}.json"
        
        os.makedirs(self.config_dir, exist_ok=True)
        filepath = os.path.join(self.config_dir, filename)
        
        config = {
            "strategy_name": strategy_name,
            "parameters": params,
            "metrics": metrics or {},
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                **(metadata or {})
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.current_params = params
        self.metadata = metadata
        
        return filepath
    
    def load_params(self, filepath: str) -> Dict[str, Any]:
        """
        Load parameters from JSON config file.
        
        Args:
            filepath: Path to config file
            
        Returns:
            Parameter dictionary
        """
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.current_params = config.get("parameters", {})
        self.metadata = config.get("metadata", {})
        
        return self.current_params
    
    def get_params_for_backtest(self) -> Dict[str, Any]:
        """
        Get parameters formatted for BacktestConfig.
        
        Returns:
            Dictionary with BacktestConfig-compatible parameters
        """
        if self.current_params is None:
            return {}
        
        return {
            "max_positions": self.current_params.get("max_positions", 3),
            "max_position_pct": self.current_params.get("max_position_pct", 15.0),
            "default_target_pct": self.current_params.get("target_pct", 5.0),
            "default_stop_loss_atr": self.current_params.get("stop_loss_atr", 1.5),
            "min_conviction": self.current_params.get("min_conviction", 6),
            "warmup_days": self.current_params.get("warmup_days", 15),
        }
    
    def export_to_env(self, filepath: str = ".env.optimized") -> str:
        """
        Export parameters to .env format for easy loading.
        
        Args:
            filepath: Path to export file
            
        Returns:
            Path to exported file
        """
        if self.current_params is None:
            return ""
        
        lines = ["# Optimized Parameters"]
        for key, value in self.current_params.items():
            lines.append(f"OPT_{key.upper()}={value}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """
        List all saved parameter configs.
        
        Returns:
            List of config summaries
        """
        import json
        import os
        
        configs = []
        
        if not os.path.exists(self.config_dir):
            return configs
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.config_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        config = json.load(f)
                    configs.append({
                        "filename": filename,
                        "filepath": filepath,
                        "strategy": config.get("strategy_name", "unknown"),
                        "saved_at": config.get("metadata", {}).get("saved_at", "unknown"),
                        "metrics": config.get("metrics", {}),
                        "n_params": len(config.get("parameters", {})),
                    })
                except Exception:
                    continue
        
        return sorted(configs, key=lambda x: x["saved_at"], reverse=True)
    
    def summary(self) -> str:
        """Generate configuration summary."""
        output = []
        output.append("=" * 70)
        output.append("PARAMETER CONFIG MANAGER SUMMARY")
        output.append("=" * 70)
        
        if self.current_params:
            output.append(f"\nCurrent Parameters:")
            for key, value in self.current_params.items():
                output.append(f"  {key}: {value}")
        else:
            output.append("\nNo parameters loaded.")
        
        output.append(f"\nSaved Configs: {len(self.list_configs())}")
        
        return "\n".join(output)


class OptimizationReportGenerator:
    """
    Comprehensive Optimization Report Generator.
    
    Generates detailed reports summarizing all optimization work including:
    - Best parameters found
    - Performance metrics
    - Overfitting analysis
    - Recommendations
    """
    
    def __init__(self):
        self.report_data: Dict[str, Any] = {}
        
    def add_section(self, section_name: str, data: Any):
        """Add a section to the report."""
        self.report_data[section_name] = data
        
    def generate_text_report(self) -> str:
        """Generate comprehensive text report."""
        output = []
        output.append("=" * 80)
        output.append("STRATEGY OPTIMIZATION REPORT")
        output.append("=" * 80)
        output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'best_params' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("1. BEST PARAMETERS FOUND")
            output.append("-" * 80)
            params = self.report_data['best_params']
            for key, value in params.items():
                output.append(f"  {key}: {value}")
        
        if 'grid_search' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("2. GRID SEARCH RESULTS")
            output.append("-" * 80)
            gs = self.report_data['grid_search']
            output.append(f"  Combinations tested: {gs.get('n_combinations', 'N/A')}")
            output.append(f"  Best metric: {gs.get('best_metric', 'N/A')}")
            output.append(f"  Best params: {gs.get('best_params', 'N/A')}")
        
        if 'random_search' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("3. RANDOM SEARCH RESULTS")
            output.append("-" * 80)
            rs = self.report_data['random_search']
            output.append(f"  Iterations: {rs.get('n_iterations', 'N/A')}")
            output.append(f"  Best metric: {rs.get('best_metric', 'N/A')}")
        
        if 'walk_forward' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("4. WALK-FORWARD ANALYSIS")
            output.append("-" * 80)
            wf = self.report_data['walk_forward']
            output.append(f"  Windows: {wf.get('n_windows', 'N/A')}")
            output.append(f"  Avg OOS return: {wf.get('avg_oos_return', 'N/A')}")
            output.append(f"  OOS gap: {wf.get('oos_gap', 'N/A')}")
        
        if 'monte_carlo' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("5. MONTE CARLO SIMULATION")
            output.append("-" * 80)
            mc = self.report_data['monte_carlo']
            output.append(f"  Simulations: {mc.get('n_sims', 'N/A')}")
            output.append(f"  Mean return: {mc.get('mean_return', 'N/A')}")
            output.append(f"  Std deviation: {mc.get('std_return', 'N/A')}")
            output.append(f"  Prob(positive): {mc.get('prob_positive', 'N/A')}")
        
        if 'sensitivity' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("6. SENSITIVITY ANALYSIS")
            output.append("-" * 80)
            sens = self.report_data['sensitivity']
            output.append(f"  Most sensitive param: {sens.get('most_sensitive', 'N/A')}")
            output.append(f"  Impact score: {sens.get('impact_score', 'N/A')}")
        
        if 'overfitting' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("7. OVERFITTING ANALYSIS")
            output.append("-" * 80)
            of = self.report_data['overfitting']
            output.append(f"  Verdict: {of.get('verdict', 'N/A')}")
            output.append(f"  Overfit score: {of.get('score', 'N/A')}")
            output.append(f"  IC ratio: {of.get('ic_ratio', 'N/A')}")
        
        if 'statistical' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("8. STATISTICAL SIGNIFICANCE")
            output.append("-" * 80)
            stat = self.report_data['statistical']
            output.append(f"  t-statistic: {stat.get('t_stat', 'N/A')}")
            output.append(f"  p-value: {stat.get('p_value', 'N/A')}")
            output.append(f"  Significant: {stat.get('is_significant', 'N/A')}")
        
        if 'recommendations' in self.report_data:
            output.append("\n" + "-" * 80)
            output.append("9. RECOMMENDATIONS")
            output.append("-" * 80)
            for i, rec in enumerate(self.report_data['recommendations'], 1):
                output.append(f"  {i}. {rec}")
        
        output.append("\n" + "=" * 80)
        output.append("END OF REPORT")
        output.append("=" * 80)
        
        return "\n".join(output)
    
    def generate_summary_dict(self) -> Dict[str, Any]:
        """Generate summary dictionary for programmatic access."""
        return {
            'best_params': self.report_data.get('best_params', {}),
            'best_metric': self.report_data.get('grid_search', {}).get('best_metric'),
            'oos_gap': self.report_data.get('walk_forward', {}).get('oos_gap'),
            'mean_return': self.report_data.get('monte_carlo', {}).get('mean_return'),
            'is_overfit': self.report_data.get('overfitting', {}).get('verdict') == 'OVERFIT',
            'is_significant': self.report_data.get('statistical', {}).get('is_significant'),
            'most_sensitive_param': self.report_data.get('sensitivity', {}).get('most_sensitive'),
            'recommendations': self.report_data.get('recommendations', []),
        }
    
    def save_report(self, filepath: str = "reports/optimization_report.txt"):
        """Save report to file."""
        import os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.generate_text_report())
        return filepath


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.
    
    Runs multiple simulations with randomized parameters and market conditions
    to estimate probability distribution of returns and drawdowns.
    """
    
    def __init__(self, n_simulations: int = 100, seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            n_simulations: Number of simulations to run
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.n_simulations = n_simulations
        self.results_df: Optional[pd.DataFrame] = None
        self.statistics: Optional[Dict[str, float]] = None
        
    def run_simulation(
        self,
        symbol: str,
        signals: Dict[str, List[TradingSignal]],
        start_date: datetime,
        end_date: datetime,
        base_params: Optional[Dict[str, Any]] = None,
        param_variation: float = 0.2,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.
        
        Args:
            symbol: Stock symbol
            signals: Trading signals
            start_date: Backtest start date
            end_date: Backtest end date
            base_params: Base parameters to vary
            param_variation: Fractional variation in parameters (±param_variation)
            verbose: Print progress
            
        Returns:
            DataFrame with simulation results
        """
        if base_params is None:
            base_params = {
                'max_positions': 3,
                'max_position_pct': 15.0,
                'target_pct': 5.0,
                'stop_loss_atr': 1.0,
                'min_conviction': 6,
            }
        
        results = []
        
        for i in range(self.n_simulations):
            if verbose and i % 20 == 0:
                print(f"  Simulation {i+1}/{self.n_simulations}...")
            
            varied_params = {}
            for key, value in base_params.items():
                if isinstance(value, int):
                    variation = int(value * param_variation)
                    varied_params[key] = max(1, value + random.randint(-variation, max(1, variation)))
                elif isinstance(value, float):
                    varied_params[key] = value * (1 + random.uniform(-param_variation, param_variation))
                else:
                    varied_params[key] = value
            
            config_params = {
                'initial_capital': 1000000,
                'warmup_days': 15,
                'max_positions': varied_params.get('max_positions', 3),
                'max_position_pct': varied_params.get('max_position_pct', 15.0),
                'default_target_pct': varied_params.get('target_pct', 5.0),
                'default_stop_loss_atr': varied_params.get('stop_loss_atr', 1.0),
                'min_conviction': varied_params.get('min_conviction', 6),
            }
            
            try:
                config = BacktestConfig(**config_params)
                backtester = Backtester(config)
                result = backtester.run(
                    symbols=[symbol],
                    signals=signals,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                results.append({
                    'simulation': i + 1,
                    'return_pct': result.metrics.get('total_return_pct', 0),
                    'max_drawdown_pct': result.metrics.get('max_drawdown_pct', 0),
                    'win_rate': result.metrics.get('win_rate', 0),
                    'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                    'num_trades': len(result.trades),
                    **{f'param_{k}': v for k, v in varied_params.items()},
                })
                
            except Exception as e:
                results.append({
                    'simulation': i + 1,
                    'return_pct': -999,
                    'max_drawdown_pct': 0,
                    'win_rate': 0,
                    'sharpe_ratio': 0,
                    'num_trades': 0,
                    **{f'param_{k}': v for k, v in varied_params.items()},
                })
        
        self.results_df = pd.DataFrame(results)
        
        self.statistics = {
            'mean_return': self.results_df['return_pct'].mean(),
            'median_return': self.results_df['return_pct'].median(),
            'std_return': self.results_df['return_pct'].std(),
            'min_return': self.results_df['return_pct'].min(),
            'max_return': self.results_df['return_pct'].max(),
            'percentile_5': self.results_df['return_pct'].quantile(0.05),
            'percentile_95': self.results_df['return_pct'].quantile(0.95),
            'prob_positive': (self.results_df['return_pct'] > 0).mean() * 100,
            'mean_max_dd': self.results_df['max_drawdown_pct'].mean(),
            'prob_large_dd': (self.results_df['max_drawdown_pct'] > 5).mean() * 100,
        }
        
        return self.results_df
    
    def get_statistics(self) -> Dict[str, float]:
        """Get simulation statistics."""
        return self.statistics if self.statistics else {}
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile of returns."""
        if self.results_df is None:
            return 0.0
        return self.results_df['return_pct'].quantile(percentile)
    
    def summary(self) -> str:
        """Generate Monte Carlo summary report."""
        if self.results_df is None or self.statistics is None:
            return "No results yet. Run simulation first."
        
        output = []
        output.append("=" * 70)
        output.append("MONTE CARLO SIMULATION SUMMARY")
        output.append("=" * 70)
        
        output.append(f"\nSimulations: {len(self.results_df)}")
        output.append(f"Mean Return:   {self.statistics['mean_return']:.2f}%")
        output.append(f"Median Return: {self.statistics['median_return']:.2f}%")
        output.append(f"Std Dev:       {self.statistics['std_return']:.2f}%")
        output.append(f"Min Return:    {self.statistics['min_return']:.2f}%")
        output.append(f"Max Return:    {self.statistics['max_return']:.2f}%")
        output.append(f"5th Percentile: {self.statistics['percentile_5']:.2f}%")
        output.append(f"95th Percentile: {self.statistics['percentile_95']:.2f}%")
        output.append(f"Prob(Return > 0): {self.statistics['prob_positive']:.1f}%")
        output.append(f"Mean Max DD:   {self.statistics['mean_max_dd']:.2f}%")
        output.append(f"Prob(Max DD > 5%): {self.statistics['prob_large_dd']:.1f}%")
        
        output.append("\n" + "-" * 70)
        output.append("RISK METRICS DISTRIBUTION")
        output.append("-" * 70)
        
        output.append(f"\nReturn Distribution:")
        output.append(f"  5th percentile:   {self.statistics['percentile_5']:.2f}%")
        output.append(f"  25th percentile:  {self.results_df['return_pct'].quantile(0.25):.2f}%")
        output.append(f"  50th percentile: {self.statistics['median_return']:.2f}%")
        output.append(f"  75th percentile:  {self.results_df['return_pct'].quantile(0.75):.2f}%")
        output.append(f"  95th percentile:  {self.statistics['percentile_95']:.2f}%")
        
        return "\n".join(output)
