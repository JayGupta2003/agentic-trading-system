"""
Risk management rules for backtesting.

Implements position sizing, drawdown limits, VaR constraints, etc.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_size_pct: float = 20.0
    max_portfolio_var_pct: float = 5.0
    max_drawdown_pct: float = 15.0
    max_daily_loss_pct: float = 5.0
    max_sector_exposure_pct: float = 30.0
    max_correlation: float = 0.7
    max_open_positions: int = 5
    trailing_stop_atr_multiplier: float = 2.0
    stop_loss_atr_multiplier: float = 2.0


class RiskManager:
    """
    Manages risk rules and position sizing.
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.peak_portfolio_value = 0.0
        self.circuit_breaker_triggered = False
        self._position_correlations: Dict[str, float] = {}
        self._sector_exposure: Dict[str, float] = {}
    
    def calculate_position_size(
        self,
        conviction: int,
        portfolio_value: float,
        price: float,
        atr: float = 0.0,
        kelly_fraction: float = 0.25,
        win_rate: float = 0.55,
        avg_win_pct: float = 8.0,
        avg_loss_pct: float = 4.0,
    ) -> int:
        """
        Calculate position size based on Kelly Criterion and risk rules.
        
        Returns:
            Number of shares to buy
        """
        max_position_value = portfolio_value * (self.limits.max_position_size_pct / 100)
        
        kelly_size = self._kelly_position_size(
            portfolio_value, win_rate, avg_win_pct, avg_loss_pct, kelly_fraction
        )
        
        raw_size = min(kelly_size, max_position_value)
        
        if atr > 0:
            risk_amount = portfolio_value * (self.limits.max_daily_loss_pct / 100)
            atr_risk_size = risk_amount / (atr * self.limits.stop_loss_atr_multiplier)
            raw_size = min(raw_size, atr_risk_size * price)
        
        shares = int(raw_size / price)
        return max(1, shares)
    
    def _kelly_position_size(
        self,
        portfolio_value: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        kelly_fraction: float,
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Kelly % = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        """
        if win_rate <= 0 or avg_loss_pct <= 0:
            return portfolio_value * 0.1
        
        win_loss_ratio = avg_win_pct / avg_loss_pct
        
        if win_loss_ratio <= 0:
            return portfolio_value * 0.1
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        kelly = max(0.01, min(0.25, kelly * kelly_fraction))
        
        return portfolio_value * kelly
    
    def should_trigger_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        atr: float,
        position_high: float,
    ) -> tuple[bool, str]:
        """
        Check if stop loss should trigger.
        
        Returns:
            (should_exit, reason)
        """
        stop_loss_price = entry_price - (atr * self.limits.stop_loss_atr_multiplier)
        
        if current_price <= stop_loss_price:
            return True, "stop_loss"
        
        trailing_stop_price = position_high - (atr * self.limits.trailing_stop_atr_multiplier)
        if current_price <= trailing_stop_price and position_high > entry_price * 1.02:
            return True, "trailing_stop"
        
        return False, ""
    
    def should_trigger_target(
        self,
        current_price: float,
        target_price: float,
        partial_exit: bool = True,
    ) -> tuple[bool, str]:
        """
        Check if target price is hit.
        
        Returns:
            (should_exit, reason)
        """
        if target_price and current_price >= target_price:
            if partial_exit:
                return True, "target_partial"
            else:
                return True, "target_full"
        
        return False, ""
    
    def check_risk_limits(
        self,
        portfolio_value: float,
        daily_loss: float,
        open_positions: int,
    ) -> tuple[bool, str]:
        """
        Check if risk limits are breached.
        
        Returns:
            (limits_breached, reason)
        """
        if open_positions >= self.limits.max_open_positions:
            return True, "max_positions"
        
        if daily_loss > 0:
            daily_loss_pct = (daily_loss / portfolio_value) * 100
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                return True, "daily_loss_limit"
        
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
            if current_drawdown >= self.limits.max_drawdown_pct:
                self.circuit_breaker_triggered = True
                return True, "circuit_breaker_drawdown"
        
        return False, ""
    
    def update_peak(self, portfolio_value: float) -> None:
        """Update peak portfolio value for drawdown tracking."""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
    
    def get_current_drawdown(self, portfolio_value: float) -> float:
        """Get current drawdown percentage."""
        if self.peak_portfolio_value == 0:
            return 0.0
        return (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value * 100
    
    def is_trading_paused(self) -> bool:
        """Check if trading is paused due to circuit breaker."""
        return self.circuit_breaker_triggered
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (for new backtest)."""
        self.circuit_breaker_triggered = False
        self.peak_portfolio_value = 0.0
    
    def update_correlation(self, symbol: str, correlation: float) -> None:
        """
        Update correlation for a position relative to existing portfolio.
        
        Args:
            symbol: Stock symbol
            correlation: Correlation coefficient (-1 to 1)
        """
        self._position_correlations[symbol] = abs(correlation)
    
    def can_add_position(self, new_correlation: float) -> tuple[bool, str]:
        """
        Check if new position can be added based on correlation limits.
        
        Args:
            new_correlation: Correlation of new position with existing portfolio
        
        Returns:
            (can_add, reason)
        """
        if not self._position_correlations:
            return True, "First position"
        
        max_corr = max(self._position_correlations.values()) if self._position_correlations else 0
        combined_corr = max(max_corr, abs(new_correlation))
        
        if combined_corr > self.limits.max_correlation:
            return False, f"correlation_exceeded: {combined_corr:.2f} > {self.limits.max_correlation}"
        
        return True, "OK"
    
    def update_sector_exposure(self, sector: str, exposure_pct: float) -> None:
        """
        Update sector exposure tracking.
        
        Args:
            sector: Sector name (e.g., "IT", "BANKING")
            exposure_pct: Exposure as percentage of portfolio
        """
        self._sector_exposure[sector] = exposure_pct
    
    def check_sector_limit(self, sector: str, additional_exposure: float = 0) -> tuple[bool, str]:
        """
        Check if sector exposure limit would be breached.
        
        Args:
            sector: Sector name
            additional_exposure: Additional exposure to add
        
        Returns:
            (within_limit, reason)
        """
        current = self._sector_exposure.get(sector, 0)
        total = current + additional_exposure
        
        if total > self.limits.max_sector_exposure_pct:
            return False, f"sector_exceeded: {total:.1f}% > {self.limits.max_sector_exposure_pct}%"
        
        return True, "OK"
    
    def get_portfolio_stats(self) -> dict:
        """Get current portfolio risk statistics."""
        return {
            "max_correlation": max(self._position_correlations.values()) if self._position_correlations else 0,
            "num_positions": len(self._position_correlations),
            "sectors": dict(self._sector_exposure),
            "peak_value": self.peak_portfolio_value,
            "circuit_breaker": self.circuit_breaker_triggered,
        }
