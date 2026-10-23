"""
NIFTY 500 Universe Loader

Loads the list of NIFTY 500 stocks for screening.
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from config.settings import settings

import loguru
logger = loguru.logger


class NiftyUniverseLoader:
    """Loads and manages stock universe (NIFTY 50, NIFTY 100, NIFTY 500)."""

    NIFTY_500_URL = "https://www.nseindia.com/content/indices/ind_nifty500list.csv"

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or settings.data_dir
        self.cache_file = self.cache_dir / "nifty500_universe.csv"
        self.cache_days = 7

    def load_nifty500(self, force_refresh: bool = False) -> List[str]:
        """
        Load NIFTY 500 stock list.

        Args:
            force_refresh: Force download even if cache exists

        Returns:
            List of NIFTY 500 stock symbols
        """
        if not force_refresh and self._is_cache_valid():
            return self._load_from_cache()

        return self._download_nifty500()

    def _is_cache_valid(self) -> bool:
        """Check if cache file exists and is fresh."""
        if not self.cache_file.exists():
            return False

        try:
            mtime = datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            return age_days < self.cache_days
        except Exception:
            return False

    def _download_nifty500(self) -> List[str]:
        """Download NIFTY 500 list from NSE."""
        symbols = []

        try:
            logger.info("Downloading NIFTY 500 list...")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/csv,application/json",
            }

            response = requests.get(self.NIFTY_500_URL, headers=headers, timeout=30)

            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))

                symbols = df["Symbol"].tolist()

                self._save_to_cache(df)

                logger.info(f"Downloaded {len(symbols)} NIFTY 500 symbols")
            else:
                logger.warning(f"NSE returned status {response.status_code}, using backup")
                symbols = self._load_backup_universe()

        except Exception as e:
            logger.error(f"Failed to download NIFTY 500: {e}")
            symbols = self._load_backup_universe()

        return symbols

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        """Save universe to cache file."""
        try:
            df.to_csv(self.cache_file, index=False)
            logger.info(f"Saved universe to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_from_cache(self) -> List[str]:
        """Load universe from cache file."""
        try:
            df = pd.read_csv(self.cache_file)
            symbols = df["Symbol"].tolist()
            logger.info(f"Loaded {len(symbols)} symbols from cache")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return self._download_nifty500()

    def _load_backup_universe(self) -> List[str]:
        """Load backup universe of major NIFTY stocks."""
        backup_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFOSYS", "HINDUJAL", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HDFC", "AXISBANK",
            "ASIANPAINT", "MARUTI", "NESTLEIND", "ULTRACEMCO", "TITAN", "BAJFINANCE",
            "SUNPHARMA", "TATASTEEL", "WIPRO", "POWERGRID", "NTPC", "ONGC",
            "COALINDIA", "JSWSTEEL", "ADANIPORTS", "CIPLA", "TECHM", "M&M",
            "BAJAJFINSV", "HCLTECH", "INDUSINDBK", "GRASIM", "ADANIENT",
            "SBILIFE", "BRITANNIA", "DRREDDY", "HEROMOTOCO", "EICHERMOT",
            "DIVISLAB", "APOLLOHOSP", "UPL", "BPCL", "IOC", "SHREECEM",
            "HINDUNILVR", "NAUKRI", "TATAMOTORS", "TATA consumer", "BANKBARODA",
            "TRENT", "VEDL", "ZOMATO", "PAYTM", "DMART", "JUBLFOOD",
        ]
        return [s.replace(" ", "").upper() for s in backup_stocks]

    def get_nifty50(self) -> List[str]:
        """Get NIFTY 50 stocks."""
        nifty50 = [
            "RELIANCE", "TCS", "HDFCBANK", "INFOSYS", "HINDUJAL", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HDFC", "AXISBANK",
            "ASIANPAINT", "MARUTI", "NESTLEIND", "ULTRACEMCO", "TITAN", "BAJFINANCE",
            "SUNPHARMA", "TATASTEEL", "WIPRO", "POWERGRID", "NTPC", "ONGC",
            "COALINDIA", "JSWSTEEL", "ADANIPORTS", "CIPLA", "TECHM", "M&M",
            "BAJAJFINSV", "HCLTECH", "INDUSINDBK", "GRASIM", "ADANIENT",
            "SBILIFE", "BRITANNIA", "DRREDDY", "HEROMOTOCO", "EICHERMOT",
            "DIVISLAB", "APOLLOHOSP", "UPL", "BPCL", "IOC", "SHREECEM",
            "HINDUNILVR"
        ]
        return [s.replace(" ", "").upper() for s in nifty50]

    def get_universe(self, universe: str = "NIFTY500") -> List[str]:
        """
        Get stock universe by name.

        Args:
            universe: "NIFTY50", "NIFTY100", or "NIFTY500"

        Returns:
            List of stock symbols
        """
        if universe == "NIFTY50":
            return self.get_nifty50()
        elif universe == "NIFTY100":
            return self.load_nifty500()[:100]
        else:
            return self.load_nifty500()


nifty_universe = NiftyUniverseLoader()


if __name__ == "__main__":
    loader = NiftyUniverseLoader()
    nifty500 = loader.get_universe("NIFTY500")
    print(f"Loaded {len(nifty500)} stocks")
    print(f"First 10: {nifty500[:10]}")
