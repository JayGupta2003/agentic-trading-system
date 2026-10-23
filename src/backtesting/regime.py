"""
Regime testing module for backtesting across different market conditions.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

@dataclass
class MarketRegime:
    """Defines a market regime/period for testing."""
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    regime_type: str  # "bull", "bear", "sideways", "high_vol", "low_vol"

REGIME_DEFINITIONS: Dict[str, MarketRegime] = {
    "bull_2020_2021": MarketRegime(
        name="bull_2020_2021",
        description="Post-COVID bull market (2020-2021)",
        start_date=datetime(2020, 4, 1),
        end_date=datetime(2021, 12, 31),
        regime_type="bull",
    ),
    "bear_2020": MarketRegime(
        name="bear_2020",
        description="COVID crash (March 2020)",
        start_date=datetime(2020, 2, 1),
        end_date=datetime(2020, 3, 31),
        regime_type="bear",
    ),
    "bear_2022": MarketRegime(
        name="bear_2022",
        description="Rate hike bear market (2022)",
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 12, 31),
        regime_type="bear",
    ),
    "sideways_2022_2023": MarketRegime(
        name="sideways_2022_2023",
        description="Sideways market (2022-2023)",
        start_date=datetime(2022, 6, 1),
        end_date=datetime(2023, 6, 30),
        regime_type="sideways",
    ),
    "high_vol_2020": MarketRegime(
        name="high_vol_2020",
        description="High volatility period (Feb-March 2020)",
        start_date=datetime(2020, 2, 1),
        end_date=datetime(2020, 4, 30),
        regime_type="high_vol",
    ),
    "low_vol_2021": MarketRegime(
        name="low_vol_2021",
        description="Low volatility bull run (2021)",
        start_date=datetime(2021, 4, 1),
        end_date=datetime(2021, 12, 31),
        regime_type="low_vol",
    ),
}


class RegimeTester:
    """Tests strategies across different market regimes."""
    
    def __init__(self, backtester, metrics_calculator):
        self.backtester = backtester
        self.metrics_calculator = metrics_calculator
    
    def test_regime(
        self,
        symbols: List[str],
        signals: Dict[str, List],
        regime: MarketRegime,
        **backtest_kwargs
    ) -> Dict:
        """Run backtest for a specific regime."""
        result = self.backtester.run(
            symbols=symbols,
            signals=signals,
            start_date=regime.start_date,
            end_date=regime.end_date,
            **backtest_kwargs
        )
        
        return {
            "regime": regime.name,
            "regime_type": regime.regime_type,
            "description": regime.description,
            "start_date": regime.start_date,
            "end_date": regime.end_date,
            "metrics": result.metrics,
            "total_trades": len(result.trades),
            "equity_curve": result.equity_curve,
        }
    
    def test_all_regimes(
        self,
        symbols: List[str],
        signals: Dict[str, List],
        regimes: List[str] = None,
        **backtest_kwargs
    ) -> Dict[str, Dict]:
        """Run backtest across all defined regimes."""
        if regimes is None:
            regimes = list(REGIME_DEFINITIONS.keys())
        
        results = {}
        for regime_name in regimes:
            if regime_name in REGIME_DEFINITIONS:
                regime = REGIME_DEFINITIONS[regime_name]
                try:
                    results[regime_name] = self.test_regime(
                        symbols, signals, regime, **backtest_kwargs
                    )
                except Exception as e:
                    results[regime_name] = {
                        "regime": regime_name,
                        "error": str(e),
                    }
        
        return results


def get_regime(name: str) -> MarketRegime:
    """Get a regime by name."""
    return REGIME_DEFINITIONS.get(name)


def list_regimes() -> List[str]:
    """List all available regime names."""
    return list(REGIME_DEFINITIONS.keys())


def get_regimes_by_type(regime_type: str) -> List[MarketRegime]:
    """Get all regimes of a specific type."""
    return [
        r for r in REGIME_DEFINITIONS.values()
        if r.regime_type == regime_type
    ]
