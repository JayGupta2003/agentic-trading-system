"""
Test Phase 6.13.10 - Store Backtest Results to JSON
"""
import sys
sys.path.insert(0, "J:/KAAM/Python/Quant/Agentic Trading System Part 2")

from datetime import datetime, timedelta
from src.backtesting.interface import BacktesterInterface
from src.backtesting.signals import Signal, TradingSignal

def test_save_load_backtest_result():
    """Test saving and loading backtest results."""
    print("=" * 60)
    print("Phase 6.13.10: Store Backtest Results to JSON")
    print("=" * 60)
    
    interface = BacktesterInterface()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("\n1. Running backtest...")
    signals = {
        "RELIANCE.NS": [
            TradingSignal(
                symbol="RELIANCE.NS",
                signal=Signal.BUY,
                conviction=8,
                timestamp=start_date,
                reasoning="Test BUY signal"
            )
        ],
        "TCS.NS": [
            TradingSignal(
                symbol="TCS.NS",
                signal=Signal.STRONG_BUY,
                conviction=9,
                timestamp=start_date,
                reasoning="Test STRONG_BUY signal"
            )
        ]
    }
    
    result = interface.run_backtest(
        signals=signals,
        symbols=["RELIANCE.NS", "TCS.NS"],
        start_date=start_date,
        end_date=end_date,
    )
    
    print(f"   Backtest completed: {result.metrics.get('total_trades', 0)} trades")
    print(f"   Total Return: {result.metrics.get('total_return_pct', 0):.2f}%")
    print(f"   Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    
    print("\n2. Testing save_backtest_result()...")
    output_path = interface.save_backtest_result(
        backtest_result=result,
        output_dir="backtest_results",
        name="test_run_2026",
        tags=["test", "demo"],
    )
    print(f"   Saved to: {output_path}")
    
    print("\n3. Testing load_backtest_result()...")
    loaded = interface.load_backtest_result(output_path)
    
    print(f"   Metadata name: {loaded['metadata'].get('name')}")
    print(f"   Tags: {loaded['metadata'].get('tags')}")
    print(f"   Config initial_capital: {loaded['config'].initial_capital}")
    print(f"   Metrics keys: {list(loaded['metrics'].keys())[:5]}...")
    print(f"   Trades count: {len(loaded['trades'])}")
    print(f"   Equity curve length: {len(loaded['equity_curve'])}")
    
    print("\n4. Testing list_backtest_results()...")
    results_list = interface.list_backtest_results("backtest_results")
    print(f"   Found {len(results_list)} backtest result(s):")
    for r in results_list:
        print(f"   - {r['name']}: {r['total_trades']} trades, {r['total_return_pct']:.2f}% return")
    
    print("\n5. Verifying round-trip data integrity...")
    assert loaded['metadata']['name'] == 'test_run_2026'
    assert loaded['metadata']['tags'] == ['test', 'demo']
    assert loaded['summary']['total_trades'] == result.metrics.get('total_trades', 0)
    print("   Round-trip integrity: PASSED")
    
    print("\n" + "=" * 60)
    print("Phase 6.13.10: COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        test_save_load_backtest_result()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
