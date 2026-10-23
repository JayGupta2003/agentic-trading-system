"""
Multi-Factor Screening Engine

Implements various screening factors:
- Piotroski F-Score
- Momentum factors
- Value factors
- Quality factors
- Growth factors
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime

from src.data.market_data import MarketDataFetcher, calculate_technical_indicators
from src.data.fundamental import get_fundamental_data

import loguru
logger = loguru.logger


class ScreeningFactors:
    """Calculate various screening factors for stocks."""

    def __init__(self):
        self.market = MarketDataFetcher()

    def calculate_all_factors(self, symbol: str) -> Optional[Dict]:
        """
        Calculate all screening factors for a symbol.

        Returns:
            dict with all factor values, or None if data unavailable
        """
        try:
            fundamental = get_fundamental_data(symbol)
            price_data = self.market.get_price_data(symbol, period="1y")

            if fundamental is None or price_data.empty:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            factors = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

            factors.update(self._calculate_quality_factors(fundamental))
            factors.update(self._calculate_value_factors(fundamental, price_data))
            factors.update(self._calculate_growth_factors(fundamental, price_data))
            factors.update(self._calculate_momentum_factors(price_data))
            factors.update(self._calculate_technical_factors(price_data))

            factors["composite_score"] = self._calculate_composite_score(factors)

            return factors

        except Exception as e:
            logger.error(f"Error calculating factors for {symbol}: {e}")
            return None

    def _calculate_quality_factors(self, fundamental: Dict) -> Dict:
        """Calculate quality factors."""
        factors = {}

        factors["piotroski_fscore"] = fundamental.get("piotroski_fscore", 0)

        net_profit = fundamental.get("fy_net_profit") or fundamental.get("net_profit") or 0
        total_assets = fundamental.get("bs_total_assets") or 1
        factors["roa"] = (net_profit / total_assets) * 100 if total_assets else 0

        factors["roe"] = fundamental.get("roe", 0) or 0
        factors["roce"] = fundamental.get("roce", 0) or 0

        debt_equity = fundamental.get("debt_equity_ratio") or 0
        factors["debt_equity"] = debt_equity
        factors["low_debt"] = 1 if debt_equity < 0.5 else 0

        factors["current_ratio"] = fundamental.get("current_ratio") or 1
        factors["good_current_ratio"] = 1 if factors["current_ratio"] > 1 else 0

        factors["opm"] = fundamental.get("fy_opm") or fundamental.get("opm") or 0
        factors["npm"] = fundamental.get("npm") or 0

        promoters = fundamental.get("promoters") or 0
        factors["high_promoter_holding"] = 1 if 50 <= promoters <= 75 else 0

        return factors

    def _calculate_value_factors(self, fundamental: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate value factors."""
        factors = {}

        pe = fundamental.get("stock_p_e") or fundamental.get("pe_ratio")
        factors["pe_ratio"] = pe or 50
        factors["low_pe"] = 1 if (pe and 0 < pe < 25) else 0
        factors["reasonable_pe"] = 1 if (pe and 0 < pe < 35) else 0

        book_value = fundamental.get("book_value") or 1
        current_price = price_data.iloc[-1]["close"] if not price_data.empty else 0
        factors["pb_ratio"] = current_price / book_value if book_value > 0 else 100
        factors["low_pb"] = 1 if factors["pb_ratio"] < 3 else 0

        market_cap = fundamental.get("market_cap_cr") or 0
        if market_cap > 50000:
            factors["large_cap"] = 1
        elif market_cap > 5000:
            factors["mid_cap"] = 1
        else:
            factors["small_cap"] = 1

        return factors

    def _calculate_growth_factors(self, fundamental: Dict, price_data: pd.DataFrame) -> Dict:
        """Calculate growth factors."""
        factors = {}

        factors["sales_yoy"] = self._estimate_sales_growth(fundamental)
        factors["profit_yoy"] = self._estimate_profit_growth(fundamental)
        factors["positive_growth"] = 1 if factors["profit_yoy"] > 0 else 0

        if not price_data.empty and len(price_data) >= 252:
            price_1y_ago = price_data.iloc[-252]["close"]
            price_now = price_data.iloc[-1]["close"]
            factors["price_return_1y"] = ((price_now - price_1y_ago) / price_1y_ago) * 100
        else:
            factors["price_return_1y"] = 0

        if not price_data.empty and len(price_data) >= 63:
            price_3m_ago = price_data.iloc[-63]["close"]
            price_now = price_data.iloc[-1]["close"]
            factors["price_return_3m"] = ((price_now - price_3m_ago) / price_3m_ago) * 100
        else:
            factors["price_return_3m"] = 0

        return factors

    def _estimate_sales_growth(self, fundamental: Dict) -> float:
        """Estimate sales YoY growth from available data."""
        return 0.0

    def _estimate_profit_growth(self, fundamental: Dict) -> float:
        """Estimate profit YoY growth from available data."""
        return 0.0

    def _calculate_momentum_factors(self, price_data: pd.DataFrame) -> Dict:
        """Calculate momentum factors."""
        factors = {}

        if price_data.empty or len(price_data) < 200:
            return factors

        current_price = price_data.iloc[-1]["close"]
        sma_50 = price_data["close"].rolling(50).mean().iloc[-1]
        sma_200 = price_data["close"].rolling(200).mean().iloc[-1]

        factors["price_vs_sma50"] = ((current_price - sma_50) / sma_50) * 100
        factors["price_vs_sma200"] = ((current_price - sma_200) / sma_200) * 100

        factors["above_sma50"] = 1 if current_price > sma_50 else 0
        factors["above_sma200"] = 1 if current_price > sma_200 else 0
        factors["golden_cross"] = 1 if sma_50 > sma_200 else 0

        returns_20d = price_data["close"].pct_change(20).iloc[-1] * 100
        factors["momentum_20d"] = returns_20d
        factors["positive_momentum"] = 1 if returns_20d > 0 else 0

        return factors

    def _calculate_technical_factors(self, price_data: pd.DataFrame) -> Dict:
        """Calculate technical factors."""
        factors = {}

        if price_data.empty or len(price_data) < 14:
            return factors

        df = calculate_technical_indicators(price_data)
        latest = df.iloc[-1]

        factors["rsi_14"] = latest.get("rsi_14", 50)
        factors["rsi_oversold"] = 1 if factors["rsi_14"] < 30 else 0
        factors["rsi_neutral"] = 1 if 40 <= factors["rsi_14"] <= 60 else 0

        macd = latest.get("macd", 0)
        macd_signal = latest.get("macd_signal", 0)
        factors["macd_bullish"] = 1 if macd > macd_signal else 0

        atr = latest.get("atr_14", 0)
        current_price = latest.get("close", 0)
        factors["atr_pct"] = (atr / current_price) * 100 if current_price > 0 else 0

        volume_ratio = latest.get("volume_ratio", 1)
        factors["high_volume"] = 1 if volume_ratio > 1.5 else 0

        return factors

    def _calculate_composite_score(self, factors: Dict) -> float:
        """Calculate weighted composite score."""
        score = 0.0

        score += factors.get("piotroski_fscore", 0) * 10
        score += factors.get("reasonable_pe", 0) * 5
        score += factors.get("positive_growth", 0) * 10
        score += factors.get("above_sma200", 0) * 10
        score += factors.get("golden_cross", 0) * 5
        score += factors.get("low_debt", 0) * 5
        score += factors.get("good_current_ratio", 0) * 5
        score += factors.get("macd_bullish", 0) * 5
        score += factors.get("rsi_neutral", 0) * 5

        score += factors.get("price_return_1y", 0) * 0.5
        score += factors.get("price_return_3m", 0) * 1.0

        return round(score, 2)


class StockScreener:
    """Screen stocks based on multiple criteria."""

    def __init__(self):
        self.factors = ScreeningFactors()

    def screen_stocks(
        self,
        symbols: List[str],
        min_fscore: int = 4,
        max_pe: float = 35,
        min_momentum: float = -10,
        above_sma200: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Screen stocks based on criteria.

        Args:
            symbols: List of stock symbols to screen
            min_fscore: Minimum Piotroski F-Score
            max_pe: Maximum P/E ratio
            min_momentum: Minimum 3-month price return (%)
            above_sma200: Require price above 200-day SMA
            top_n: Number of top stocks to return

        Returns:
            DataFrame with screened stocks, sorted by composite score
        """
        results = []

        for symbol in symbols:
            try:
                factors = self.factors.calculate_all_factors(symbol)

                if factors is None:
                    continue

                if factors.get("piotroski_fscore", 0) < min_fscore:
                    continue

                if factors.get("pe_ratio", 100) > max_pe:
                    continue

                if factors.get("price_return_3m", -100) < min_momentum:
                    continue

                if above_sma200 and factors.get("above_sma200", 0) == 0:
                    continue

                clean_factors = {
                    k: v for k, v in factors.items()
                    if k not in ['fscore_breakdown', 'analysis']
                    and v is not None
                    and not isinstance(v, (list, dict))
                }
                results.append(clean_factors)

            except Exception as e:
                logger.warning(f"Error screening {symbol}: {e}")
                continue

        if not results:
            logger.warning("No stocks passed the screening criteria")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("composite_score", ascending=False)

        return df.head(top_n)

    def run_daily_screen(self, universe: str = "NIFTY50", **kwargs) -> List[str]:
        """
        Run daily screening on a universe.

        Args:
            universe: Stock universe ("NIFTY50", "NIFTY100", "NIFTY500")
            **kwargs: Additional screening criteria

        Returns:
            List of top stock symbols
        """
        from src.screening.nifty500 import nifty_universe

        symbols = nifty_universe.get_universe(universe)

        df = self.screen_stocks(symbols, **kwargs)

        if df.empty:
            return []

        return df["symbol"].tolist()


screening_factors = ScreeningFactors()
stock_screener = StockScreener()


if __name__ == "__main__":
    screener = StockScreener()

    test_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
    results = screener.screen_stocks(test_stocks, top_n=5)

    print("=== Screening Results ===")
    print(results[["symbol", "piotroski_fscore", "pe_ratio", "composite_score"]].to_string())
