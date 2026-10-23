"""
Strategy configurations for backtesting.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str = "default"
    description: str = ""
    
    # Conviction thresholds
    min_conviction: int = 6
    strong_buy_threshold: int = 8
    
    # Position sizing
    max_position_pct: float = 20.0
    max_positions: int = 5
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.5
    
    # Entry
    entry_type: str = "eod"  # "eod" or "next_open"
    require_technical_confirmation: bool = False
    
    # Exit
    stop_loss_atr: float = 2.0
    trailing_stop_atr: float = 2.0
    target_pct: float = 10.0
    max_hold_days: int = 30
    
    # Risk
    max_drawdown_pct: float = 15.0
    max_daily_loss_pct: float = 5.0
    allow_short: bool = False
    
    # Sector limits
    max_sector_exposure_pct: float = 30.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'min_conviction': self.min_conviction,
            'strong_buy_threshold': self.strong_buy_threshold,
            'max_position_pct': self.max_position_pct,
            'max_positions': self.max_positions,
            'use_kelly_sizing': self.use_kelly_sizing,
            'kelly_fraction': self.kelly_fraction,
            'entry_type': self.entry_type,
            'require_technical_confirmation': self.require_technical_confirmation,
            'stop_loss_atr': self.stop_loss_atr,
            'trailing_stop_atr': self.trailing_stop_atr,
            'target_pct': self.target_pct,
            'max_hold_days': self.max_hold_days,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'allow_short': self.allow_short,
            'max_sector_exposure_pct': self.max_sector_exposure_pct,
        }


STRATEGY_PRESETS: Dict[str, StrategyConfig] = {
    "conservative": StrategyConfig(
        name="conservative",
        description="High conviction only, small positions",
        min_conviction=7,
        strong_buy_threshold=9,
        max_position_pct=15.0,
        max_positions=3,
        stop_loss_atr=1.5,
        trailing_stop_atr=1.5,
        target_pct=8.0,
        max_hold_days=20,
    ),
    "aggressive": StrategyConfig(
        name="aggressive",
        description="Lower threshold, larger positions",
        min_conviction=5,
        strong_buy_threshold=7,
        max_position_pct=25.0,
        max_positions=7,
        stop_loss_atr=2.5,
        trailing_stop_atr=2.5,
        target_pct=15.0,
        max_hold_days=45,
    ),
    "diversified": StrategyConfig(
        name="diversified",
        description="Max positions always filled",
        min_conviction=6,
        strong_buy_threshold=8,
        max_position_pct=20.0,
        max_positions=10,
        use_kelly_sizing=False,
        stop_loss_atr=2.0,
        trailing_stop_atr=2.0,
        target_pct=12.0,
        max_hold_days=30,
    ),
    "momentum": StrategyConfig(
        name="momentum",
        description="Technical confirmation required, trend following",
        min_conviction=6,
        strong_buy_threshold=8,
        max_position_pct=20.0,
        max_positions=5,
        require_technical_confirmation=True,
        stop_loss_atr=2.0,
        trailing_stop_atr=2.5,
        target_pct=15.0,
        max_hold_days=25,
    ),
    "value": StrategyConfig(
        name="value",
        description="Long-term value investing, fundamental focus",
        min_conviction=7,
        strong_buy_threshold=9,
        max_position_pct=15.0,
        max_positions=5,
        stop_loss_atr=2.5,
        trailing_stop_atr=3.0,
        target_pct=25.0,
        max_hold_days=60,
    ),
}


def get_strategy(name: str) -> StrategyConfig:
    """Get a strategy by name or return default."""
    return STRATEGY_PRESETS.get(name, StrategyConfig())
