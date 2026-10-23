"""
Test Phase 6.14.2 - Handle Missing Prices / Data Gaps
"""
import sys
sys.path.insert(0, "J:/KAAM/Python/Quant/Agentic Trading System Part 2")

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.backtesting.interface import BacktesterInterface
from src.backtesting.signals import Signal, TradingSignal

def create_test_data_with_gaps():
    """Create test data with known gaps."""
    np.random.seed(42)
    
    dates1 = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    df1 = pd.DataFrame({
        'Open': np.random.randn(len(dates1)) * 10 + 2500,
        'High': np.random.randn(len(dates1)) * 10 + 2550,
        'Low': np.random.randn(len(dates1)) * 10 + 2450,
        'Close': np.random.randn(len(dates1)) * 10 + 2500,
        'Volume': np.random.randint(1000000, 5000000, len(dates1)),
    }, index=dates1)
    
    dates_before_gap = pd.date_range(start='2024-01-01', periods=10, freq='D')
    dates_after_gap = pd.date_range(start='2024-01-25', end='2024-01-31', freq='D')
    dates_with_gap = dates_before_gap.append(dates_after_gap)
    
    df2 = pd.DataFrame({
        'Open': np.random.randn(len(dates_with_gap)) * 10 + 3500,
        'High': np.random.randn(len(dates_with_gap)) * 10 + 3550,
        'Low': np.random.randn(len(dates_with_gap)) * 10 + 3450,
        'Close': np.random.randn(len(dates_with_gap)) * 10 + 3500,
        'Volume': np.random.randint(500000, 2000000, len(dates_with_gap)),
    }, index=dates_with_gap)
    
    dates3 = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    df3 = pd.DataFrame({
        'Open': np.random.randn(len(dates3)) * 10 + 4500,
        'High': np.random.randn(len(dates3)) * 10 + 4550,
        'Low': np.random.randn(len(dates3)) * 10 + 4450,
        'Close': np.random.randn(len(dates3)) * 10 + 4500,
        'Volume': np.random.randint(800000, 3000000, len(dates3)),
    }, index=dates3)
    
    return {
        'SYMBOL_A': df1,
        'SYMBOL_B': df2,
        'SYMBOL_C': df3,
    }

def test_data_gap_handling():
    """Test data gap handling methods."""
    print("=" * 60)
    print("Phase 6.14.2: Handle Missing Prices / Data Gaps")
    print("=" * 60)
    
    interface = BacktesterInterface()
    
    print("\n1. Testing detect_data_gaps()...")
    
    test_data = create_test_data_with_gaps()
    
    for sym, df in test_data.items():
        print(f"   {sym}: {len(df)} rows, index span: {df.index[0]} to {df.index[-1]}")
        if sym == 'SYMBOL_B':
            df_idx = pd.to_datetime(df.index)
            diffs = df_idx.diff()
            print(f"   {sym} diffs: first={diffs[1]}, max={diffs.max()}")
    
    gap_report = interface.detect_data_gaps(test_data, max_gap_days=5)
    
    print(f"   Symbols Analyzed: {gap_report['symbols_analyzed']}")
    print(f"   Total Gaps: {gap_report['total_gaps']}")
    print(f"   Critical Gaps: {gap_report['critical_gaps']}")
    print(f"   Symbols with Gaps: {gap_report['symbols_with_gaps']}")
    print(f"   Symbols without Gaps: {gap_report['symbols_without_gaps']}")
    
    for symbol, detail in gap_report['gaps_detail'].items():
        if detail['total_gaps'] > 0:
            print(f"   - {symbol}: {detail['total_gaps']} gaps, {detail['critical_gaps']} critical")
            for g in detail['gaps']:
                print(f"     Gap: {g['gap_start'][:10]} to {g['gap_end'][:10]} ({g['gap_days']} days)")
    
    print("\n2. Testing fill_data_gaps() with 'interpolate'...")
    
    filled_data = interface.fill_data_gaps(test_data, method='interpolate', max_fill_days=5)
    
    for symbol in ['SYMBOL_A', 'SYMBOL_B', 'SYMBOL_C']:
        orig_len = len(test_data[symbol])
        filled_len = len(filled_data[symbol])
        print(f"   {symbol}: {orig_len} -> {filled_len} rows (gaps filled)")
    
    print("\n3. Testing handle_data_gaps() with 'error' strategy...")
    
    error_result = interface.handle_data_gaps(
        test_data, strategy='error', max_gap_days=5, max_fill_days=3
    )
    
    print(f"   Can Proceed: {error_result['can_proceed']}")
    print(f"   Action Taken: {error_result['action_taken']}")
    print(f"   Warnings: {len(error_result['warnings'])}")
    print(f"   Errors: {len(error_result['errors'])}")
    
    print("\n4. Testing handle_data_gaps() with 'skip' strategy...")
    
    skip_result = interface.handle_data_gaps(
        test_data, strategy='skip', max_gap_days=5
    )
    
    print(f"   Can Proceed: {skip_result['can_proceed']}")
    print(f"   Action Taken: {skip_result['action_taken']}")
    print(f"   Symbols Skipped: {skip_result.get('symbols_skipped', [])}")
    print(f"   Modified Symbols: {list(skip_result['modified_data'].keys())}")
    
    print("\n5. Testing handle_data_gaps() with 'fill' strategy...")
    
    fill_result = interface.handle_data_gaps(
        test_data, strategy='fill', max_gap_days=5, max_fill_days=3
    )
    
    print(f"   Can Proceed: {fill_result['can_proceed']}")
    print(f"   Action Taken: {fill_result['action_taken']}")
    print(f"   Data Modified: {fill_result['data_modified']}")
    
    print("\n6. Testing validate_data_continuity()...")
    
    signals = {
        'SYMBOL_A': [
            TradingSignal(
                symbol='SYMBOL_A',
                signal=Signal.BUY,
                conviction=7,
                timestamp=datetime(2024, 1, 10),
                reasoning='Test signal'
            )
        ],
        'SYMBOL_B': [
            TradingSignal(
                symbol='SYMBOL_B',
                signal=Signal.BUY,
                conviction=8,
                timestamp=datetime(2024, 1, 16),
                reasoning='Signal in gap period'
            )
        ]
    }
    
    continuity = interface.validate_data_continuity(test_data, signals, max_gap_days=5)
    
    print(f"   Is Valid: {continuity['is_valid']}")
    print(f"   Issues: {continuity['issues']}")
    print(f"   Warnings: {continuity['warnings']}")
    print(f"   Gap Summary: {continuity['gap_summary']}")
    
    print("\n7. Testing with real data (RELIANCE)...")
    
    real_data = interface.data_loader.load_data(
        ['RELIANCE.NS'], 
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    
    if real_data:
        real_gap_report = interface.detect_data_gaps(real_data, max_gap_days=5)
        print(f"   RELIANCE Data Points: {len(real_data.get('RELIANCE.NS', []))}")
        print(f"   Total Gaps: {real_gap_report['total_gaps']}")
        print(f"   Symbols with Gaps: {real_gap_report['symbols_with_gaps']}")
    
    print("\n" + "=" * 60)
    print("Phase 6.14.2: COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    assert gap_report['symbols_analyzed'] == 3, "Should analyze 3 symbols"
    assert len(gap_report['symbols_with_gaps']) > 0, "Should detect gaps in test data"
    assert skip_result['can_proceed'] == True, "Skip strategy should proceed"
    assert fill_result['data_modified'] == True, "Fill strategy should modify data"
    
    print("\nAll assertions passed!")
    return True

if __name__ == "__main__":
    try:
        test_data_gap_handling()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
