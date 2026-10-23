"""Regression tests for data loading in the event-driven backtester."""

import pandas as pd

from src.backtesting.event_driven import EventDrivenBacktester
from src.screening.nifty500 import nifty_universe


def test_naive_index_strips_timezone():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    out = EventDrivenBacktester._naive_index(df)
    assert out.index.tz is None
    assert (out["Close"].values == df["Close"].values).all()


def test_format_symbol_handles_special_tickers():
    bt = EventDrivenBacktester()
    assert bt._format_symbol("M&M") == "M&M.NS"
    assert bt._format_symbol("RELIANCE") == "RELIANCE.NS"
    assert bt._format_symbol("^NSEI") == "^NSEI"


def test_nifty50_universe_has_valid_tickers():
    symbols = nifty_universe.get_universe("NIFTY50")
    assert "INFY" in symbols and "INFOSYS" not in symbols
    assert "HINDUJAL" not in symbols and "HDFC" not in symbols
    assert len(symbols) == len(set(symbols))
