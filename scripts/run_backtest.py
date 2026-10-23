"""Run the event-driven backtester and report honest metrics.

What this measures
------------------
The backtester replays the trigger engine, risk rules and execution engine day
by day. It does NOT call the LLM agents: signals come from the rule-based
stand-in in BacktestSignalGenerator (RSI, moving averages, ATR). The numbers
therefore describe that framework with rule-based signals, not the AI agents.

Method
------
* One continuous run from a warm-up start. The engine sits in cash for its
  first 200 trading days (indicator warm-up), so metrics are computed only on
  the evaluation window, sliced from the single equity curve.
* Benchmarks over the same window: the NIFTY 50 index (price only, no
  dividends) and an equal-weight, daily-rebalanced basket of the same stocks.
* Costs: brokerage + slippage are charged on every trade. A sensitivity run
  repeats the backtest with zero and doubled costs.

Known biases
------------
The universe is a fixed list of large caps picked with hindsight (roughly the
NIFTY 50), so there is survivorship bias.
It affects the strategy and the equal-weight benchmark alike.

Usage: python scripts/run_backtest.py [--universe NIFTY50] [--out results]
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting.event_driven import BacktestConfig, EventDrivenBacktester  # noqa: E402
from src.screening.nifty500 import nifty_universe  # noqa: E402

WARMUP_START = datetime(2020, 3, 2)
END = datetime(2026, 10, 23)
EVAL_START = pd.Timestamp("2021-01-01")


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx.normalize()


def window_metrics(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Return, CAGR, Sharpe (rf=0), volatility and max drawdown on [start, end]."""
    eq = equity[(equity.index >= start) & (equity.index <= end)].dropna()
    if len(eq) < 20:
        return {}
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    vol = rets.std() * np.sqrt(252)
    peak = eq.cummax()
    return {
        "days": int(len(eq)),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2) if years >= 0.9 else None,
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(rets.mean() / rets.std() * np.sqrt(252), 2) if rets.std() > 0 else 0.0,
        "max_drawdown_pct": round(((eq / peak) - 1).min() * 100, 2),
    }


def trade_stats(trades, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    exits = [t for t in trades if t.action == "EXIT" and start <= pd.Timestamp(t.date).tz_localize(None) <= end]
    if not exits:
        return {"trades": 0}
    wins = [t for t in exits if t.pnl > 0]
    losses = [t for t in exits if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    return {
        "trades": len(exits),
        "win_rate_pct": round(len(wins) / len(exits) * 100, 1),
        "avg_trade_pct": round(float(np.mean([t.pnl_pct for t in exits])), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    }


def run_once(symbols, cap_types, brokerage, slippage, data=None):
    cfg = BacktestConfig(
        initial_capital=1_000_000.0,
        start_date=WARMUP_START,
        end_date=END,
        symbols=list(symbols),
        cap_types=cap_types,
        brokerage_rate=brokerage,
        slippage_pct=slippage,
    )
    bt = EventDrivenBacktester(cfg)
    if data is None:
        bt.load_data()
    else:
        bt._data, bt._vix_data, bt._nifty_data = data
    result = bt.run()
    return bt, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="NIFTY50")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    symbols = nifty_universe.get_universe(args.universe)
    cap_types = {s: "large" for s in symbols}
    base_b, base_s = BacktestConfig().brokerage_rate, BacktestConfig().slippage_pct

    print(f"Loading data for {len(symbols)} symbols ...")
    bt, res = run_once(symbols, cap_types, base_b, base_s)
    data = (bt._data, bt._vix_data, bt._nifty_data)
    print(f"Loaded {len(bt._data)} symbols. Trades: {res.metrics.total_trades}")

    eq = res.equity_curve["portfolio_value"].copy()
    eq.index = _naive(eq.index)

    nifty = bt._nifty_data["Close"].copy()
    nifty.index = _naive(nifty.index)
    closes = pd.DataFrame({s: d["Close"] for s, d in bt._data.items()})
    closes.index = _naive(closes.index)
    ew = (1 + closes.pct_change().mean(axis=1).fillna(0)).cumprod()

    last = eq.index[-1]
    windows = {str(y): (pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")) for y in range(2021, 2026)}
    windows["2026 YTD"] = (pd.Timestamp("2026-01-01"), last)
    windows["Full 2021 to 2026 YTD"] = (EVAL_START, last)

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "universe": args.universe,
        "symbols_loaded": len(bt._data),
        "signal_source": "rule-based stand-in (RSI / SMA / ATR); LLM agents are NOT called",
        "costs": {"brokerage_rate": base_b, "slippage_pct": base_s},
        "windows": {},
    }
    for name, (s, e) in windows.items():
        out["windows"][name] = {
            "strategy": window_metrics(eq, s, e),
            "trades": trade_stats(res.trades, s, e),
            "nifty50_index": window_metrics(nifty, s, e),
            "equal_weight_basket": window_metrics(ew, s, e),
        }

    # Average exposure over the full evaluation window
    snaps = pd.DataFrame(
        {"d": _naive(pd.Index([x.date for x in bt._daily_snapshots])),
         "pv": [x.portfolio_value for x in bt._daily_snapshots],
         "pos": [x.positions_value for x in bt._daily_snapshots]}
    ).set_index("d")
    snaps = snaps[snaps.index >= EVAL_START]
    out["avg_exposure_pct"] = round(float((snaps["pos"] / snaps["pv"]).mean() * 100), 1)

    # Cost sensitivity on the full window
    sens = {}
    for label, mult in (("zero costs", 0.0), ("base costs", 1.0), ("double costs", 2.0)):
        if mult == 1.0:
            e2, r2 = eq, res
        else:
            b2, r2 = run_once(symbols, cap_types, base_b * mult, base_s * mult, data=copy.copy(data))
            e2 = r2.equity_curve["portfolio_value"].copy()
            e2.index = _naive(e2.index)
        sens[label] = window_metrics(e2, EVAL_START, last)
    out["cost_sensitivity_full_window"] = sens

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    (outdir / "backtest_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def row(name, d):
        m, t, n, w = d["strategy"], d["trades"], d["nifty50_index"], d["equal_weight_basket"]
        f = lambda x, k: ("n/a" if not x or x.get(k) is None else f"{x[k]:.2f}")  # noqa: E731
        return (f"| {name} | {f(m,'total_return_pct')} | {f(m,'sharpe')} | {f(m,'max_drawdown_pct')} | "
                f"{t.get('trades', 0)} | {f(n,'total_return_pct')} | {f(w,'total_return_pct')} |")

    lines = [
        "| Window | Strategy return % | Sharpe | Max DD % | Trades | NIFTY 50 return % | Equal-weight basket return % |",
        "|---|---|---|---|---|---|---|",
    ] + [row(n, d) for n, d in out["windows"].items() if d["strategy"]]
    (outdir / "backtest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(out, indent=2))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
