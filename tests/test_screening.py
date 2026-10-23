"""
Test Screening Functionality

Run with: python -m pytest tests -v
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


def test_screener_imports():
    """Test that screener module can be imported."""
    from src.screening.screener import StockShortlister
    from src.screening.factors import ScreeningFactors, StockScreener
    assert StockShortlister is not None
    assert ScreeningFactors is not None
    assert StockScreener is not None


def test_stock_shortlister_init():
    """Test StockShortlister initialization."""
    from src.screening.screener import StockShortlister
    
    screener = StockShortlister()
    assert screener is not None
    assert screener.output_dir is not None


def test_screening_factors_init():
    """Test ScreeningFactors initialization."""
    from src.screening.factors import ScreeningFactors
    
    factors = ScreeningFactors()
    assert factors is not None
    assert factors.market is not None


def test_stock_screener_init():
    """Test StockScreener initialization."""
    from src.screening.factors import StockScreener
    
    screener = StockScreener()
    assert screener is not None
    assert screener.factors is not None


@patch('src.screening.factors.MarketDataFetcher')
@patch('src.screening.factors.get_fundamental_data')
def test_calculate_all_factors(mock_fundamental, mock_market):
    """Test factor calculation."""
    from src.screening.factors import ScreeningFactors
    
    mock_fundamental.return_value = {
        'piotroski_fscore': 7,
        'roe': 20,
        'roce': 15,
        'debt_equity_ratio': 0.3,
        'stock_p_e': 20,
    }
    
    mock_df = pd.DataFrame({
        'close': [100, 105, 110, 115, 120],
        'high': [105, 110, 115, 120, 125],
        'low': [95, 100, 105, 110, 115],
        'open': [100, 105, 110, 115, 120],
        'volume': [1000] * 5,
    })
    mock_market.return_value.get_price_data.return_value = mock_df
    
    factors = ScreeningFactors()
    result = factors.calculate_all_factors('TEST')
    
    assert result is not None
    assert result['symbol'] == 'TEST'
    assert result['piotroski_fscore'] == 7


def test_nifty_universe():
    """Test that NIFTY universe returns stocks."""
    from src.screening.nifty500 import nifty_universe
    
    nifty50 = nifty_universe.get_universe('NIFTY50')
    assert isinstance(nifty50, list)
    assert len(nifty50) > 0
    assert 'RELIANCE' in nifty50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
