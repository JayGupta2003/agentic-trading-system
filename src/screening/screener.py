"""
Stock Screener - Main Module

Orchestrates the screening process and outputs top stock recommendations.
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from src.screening.nifty500 import nifty_universe
from src.screening.factors import screening_factors, stock_screener
from config.settings import settings

import loguru
logger = loguru.logger


class StockShortlister:
    """
    Main stock screening and shortlisting module.

    Screens stocks from NIFTY universe and outputs top 5 recommendations.
    """

    def __init__(self):
        self.output_dir = Path(settings.data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_screening(
        self,
        universe: str = "NIFTY50",
        min_fscore: int = 4,
        max_pe: float = 40,
        min_momentum: float = -15,
        top_n: int = 5,
        save_output: bool = True,
    ) -> Dict:
        """
        Run stock screening and return top recommendations.

        Args:
            universe: Stock universe to screen
            min_fscore: Minimum Piotroski F-Score
            max_pe: Maximum P/E ratio
            min_momentum: Minimum 3-month momentum (%)
            top_n: Number of stocks to recommend
            save_output: Save results to JSON file

        Returns:
            Dict with screening results
        """
        logger.info(f"Starting screening on {universe}")

        symbols = nifty_universe.get_universe(universe)
        logger.info(f"Screening {len(symbols)} stocks")

        results = stock_screener.screen_stocks(
            symbols=symbols,
            min_fscore=min_fscore,
            max_pe=max_pe,
            min_momentum=min_momentum,
            top_n=top_n,
        )

        if results.empty:
            logger.warning("No stocks passed screening criteria")
            return {
                "status": "no_results",
                "timestamp": datetime.now().isoformat(),
                "universe": universe,
                "criteria": {
                    "min_fscore": min_fscore,
                    "max_pe": max_pe,
                    "min_momentum": min_momentum,
                },
                "stocks": [],
            }

        shortlist = self._format_results(results)

        output = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "universe": universe,
            "criteria": {
                "min_fscore": min_fscore,
                "max_pe": max_pe,
                "min_momentum": min_momentum,
            },
            "stocks_count": len(shortlist),
            "stocks": shortlist,
        }

        if save_output:
            self._save_to_json(output)
            self._save_to_csv(results)

        logger.info(f"Screening complete. Found {len(shortlist)} stocks")
        return output

    def _format_results(self, df: pd.DataFrame) -> List[Dict]:
        """Format screening results for output."""
        shortlist = []

        for _, row in df.iterrows():
            stock = {
                "symbol": row["symbol"],
                "composite_score": row["composite_score"],
                "quality": {
                    "piotroski_fscore": int(row.get("piotroski_fscore", 0)),
                    "roe": round(row.get("roe", 0), 2),
                    "roce": round(row.get("roce", 0), 2),
                    "debt_equity": round(row.get("debt_equity", 0), 2),
                },
                "valuation": {
                    "pe_ratio": round(row.get("pe_ratio", 0), 2),
                    "pb_ratio": round(row.get("pb_ratio", 0), 2),
                },
                "momentum": {
                    "return_3m": round(row.get("price_return_3m", 0), 2),
                    "return_1y": round(row.get("price_return_1y", 0), 2),
                    "above_sma200": bool(row.get("above_sma200", 0)),
                },
                "technical": {
                    "rsi_14": round(row.get("rsi_14", 50), 1),
                    "macd_bullish": bool(row.get("macd_bullish", 0)),
                },
            }
            shortlist.append(stock)

        return shortlist

    def _save_to_json(self, output: Dict) -> Path:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_shortlist_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved shortlist to {filepath}")
        return filepath

    def _save_to_csv(self, df: pd.DataFrame) -> Path:
        """Save full screening results to CSV."""
        if df.empty:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"screening_results_{timestamp}.csv"
        filepath = self.output_dir / filename

        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
            
            df.reset_index(drop=True, inplace=True)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved full results to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save CSV: {e}")
        
        return filepath

    def get_previous_screening(self) -> Optional[Dict]:
        """Get the most recent screening results."""
        try:
            files = list(self.output_dir.glob("stock_shortlist_*.json"))
            if not files:
                return None

            latest = max(files, key=lambda p: p.stat().st_mtime)

            with open(latest) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load previous screening: {e}")
            return None


stock_shortlister = StockShortlister()


def run_daily_screening() -> Dict:
    """
    Convenience function to run daily screening.

    Returns:
        Dict with top 5 stock recommendations
    """
    return stock_shortlister.run_screening(
        universe="NIFTY50",
        min_fscore=4,
        max_pe=40,
        min_momentum=-15,
        top_n=5,
    )


if __name__ == "__main__":
    print("=== Stock Screening Module ===")
    print("Testing screening for NIFTY 50 stocks...\n")

    result = stock_shortlister.run_screening(
        universe="NIFTY50",
        min_fscore=3,
        max_pe=50,
        min_momentum=-20,
        top_n=5,
    )

    print(f"\nStatus: {result['status']}")
    print(f"Found: {result['stocks_count']} stocks\n")

    for i, stock in enumerate(result["stocks"], 1):
        print(f"{i}. {stock['symbol']}")
        print(f"   Score: {stock['composite_score']}")
        print(f"   F-Score: {stock['quality']['piotroski_fscore']}")
        print(f"   P/E: {stock['valuation']['pe_ratio']}")
        print(f"   3M Return: {stock['momentum']['return_3m']}%")
        print()
