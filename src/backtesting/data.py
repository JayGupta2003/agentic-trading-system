"""
Data loading and preprocessing for backtesting.

Fetches and prepares historical data from yfinance.
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class DataLoader:
    """
    Loads and preprocesses historical data for backtesting.
    """
    
    DEFAULT_PERIOD = "3y"
    CACHE_DIR = "data/cache"
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or self.CACHE_DIR
    
    def load_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "3y",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            period: yfinance period (if dates not provided)
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        data = {}
        
        for symbol in symbols:
            df = self._load_symbol_data(symbol, start_date, end_date, period)
            if df is not None and not df.empty:
                data[symbol] = df
        
        return data
    
    def _load_symbol_data(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        period: str,
    ) -> Optional[pd.DataFrame]:
        """Load data for a single symbol."""
        try:
            ticker_symbol = self._format_symbol(symbol)
            ticker = yf.Ticker(ticker_symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                df = ticker.history(period=period, auto_adjust=True)
            
            if df.empty:
                return None
            
            df = self._preprocess_data(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for yfinance."""
        symbol = symbol.upper().strip()
        
        if symbol.endswith(".NS") or symbol.startswith("^"):
            return symbol
        
        known_indices = ["^NSEI", "^BSESN", "^INDIAVIX"]
        if symbol in known_indices:
            return symbol
        
        return f"{symbol}.NS"
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean data."""
        df = df.copy()
        
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        df = df[df['Volume'] > 0]
        
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data."""
        df = df.copy()
        
        df['Returns'] = df['Close'].pct_change()
        
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        df['Daily_Change'] = df['High'] - df['Low']
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def get_prices_for_date(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
    ) -> Dict[str, float]:
        """Get closing prices for all symbols on a specific date."""
        prices = {}
        
        for symbol, df in data.items():
            try:
                date_ts = pd.Timestamp(date).normalize()
                
                if date_ts in df.index:
                    prices[symbol] = float(df.loc[date_ts, 'Close'])
                else:
                    closest_idx = df.index.get_indexer([date_ts], method='nearest')[0]
                    if 0 <= closest_idx < len(df):
                        prices[symbol] = float(df.iloc[closest_idx]['Close'])
            except (KeyError, IndexError):
                continue
        
        return prices
    
    def split_train_test(
        self,
        data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.7,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data into train and test sets."""
        train_data = {}
        test_data = {}
        
        for symbol, df in data.items():
            split_idx = int(len(df) * train_ratio)
            train_data[symbol] = df.iloc[:split_idx]
            test_data[symbol] = df.iloc[split_idx:]
        
        return train_data, test_data
    
    def validate_data(
        self,
        data: Dict[str, pd.DataFrame],
        min_days: int = 200,
    ) -> Tuple[bool, List[str]]:
        """
        Validate loaded data.
        
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        for symbol, df in data.items():
            if len(df) < min_days:
                issues.append(f"{symbol}: Only {len(df)} days (min: {min_days})")
            
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 5:
                issues.append(f"{symbol}: {missing_pct:.1f}% missing data")
            
            recent_volume = df['Volume'].tail(20).mean()
            if recent_volume == 0:
                issues.append(f"{symbol}: Zero volume in recent data")
        
        return len(issues) == 0, issues
