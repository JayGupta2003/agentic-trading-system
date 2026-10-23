"""
Test Phase 6.14.3 - Handle Zero Volume Days
"""
import sys
sys.path.insert(0, "J:/KAAM/Python/Quant/Agentic Trading System Part 2")

from datetime import datetime
import pandas as pd
import numpy as np
from src.backtesting.interface import BacktesterInterface

def create_test_data_with_zero_volume():
    """Create test data with zero volume days."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    volumes = np.random.randint(1000000, 5000000, len(dates))
    volumes[5] = 0
    volumes[12] = 0
    volumes[20] = 0
    volumes[7] = 100
    volumes[15] = 500
    
    df1 = pd.DataFrame({
        'Open': np.random.randn(len(dates)) * 10 + 2500,
        'High': np.random.randn(len(dates)) * 10 + 2550,
        'Low': np.random.randn(len(dates)) * 10 + 2450,
        'Close': np.random.randn(len(dates)) * 10 + 2500,
        'Volume': volumes,
    }, index=dates)
    
    volumes2 = np.random.randint(1000000, 5000000, len(dates))
    df2 = pd.DataFrame({
        'Open': np.random.randn(len(dates)) * 10 + 3500,
        'High': np.random.randn(len(dates)) * 10 + 3550,
        'Low': np.random.randn(len(dates)) * 10 + 3450,
        'Close': np.random.randn(len(dates)) * 10 + 3500,
        'Volume': volumes2,
    }, index=dates)
    
    return {
        'SYMBOL_WITH_ZERO': df1,
        'SYMBOL_NORMAL': df2,
    }

def test_zero_volume_handling():
    """Test zero volume day handling methods."""
    print("=" * 60)
    print("Phase 6.14.3: Handle Zero Volume Days")
    print("=" * 60)
    
    interface = BacktesterInterface()
    
    print("\n1. Testing detect_zero_volume_days()...")
    
    test_data = create_test_data_with_zero_volume()
    
    detection = interface.detect_zero_volume_days(test_data, min_volume_threshold=1000)
    
    print(f"   Symbols Analyzed: {detection['symbols_analyzed']}")
    print(f"   Total Zero Volume Days: {detection['total_zero_volume_days']}")
    print(f"   Total Low Volume Days: {detection['total_low_volume_days']}")
    print(f"   Symbols with Zero Volume: {detection['symbols_with_zero_volume']}")
    print(f"   Symbols without Issues: {detection['symbols_without_issues']}")
    
    for symbol, detail in detection['detail'].items():
        print(f"   - {symbol}: zero={detail['zero_volume_days']}, low={detail['low_volume_days']}, avg_vol={detail['avg_volume']:.0f}")
        if detail['zero_volume_dates']:
            print(f"     Zero volume dates: {detail['zero_volume_dates']}")
    
    print("\n2. Testing handle_zero_volume_days() with 'filter' strategy...")
    
    filter_result = interface.handle_zero_volume_days(
        test_data, strategy='filter', min_volume_threshold=1000
    )
    
    print(f"   Can Proceed: {filter_result['can_proceed']}")
    print(f"   Action Taken: {filter_result['action_taken']}")
    print(f"   Data Modified: {filter_result['data_modified']}")
    
    for sym, df in filter_result['modified_data'].items():
        orig_len = len(test_data[sym])
        new_len = len(df)
        print(f"   {sym}: {orig_len} -> {new_len} rows")
    
    print("\n3. Testing handle_zero_volume_days() with 'skip' strategy...")
    
    skip_result = interface.handle_zero_volume_days(
        test_data, strategy='skip', min_volume_threshold=1000
    )
    
    print(f"   Can Proceed: {skip_result['can_proceed']}")
    print(f"   Action Taken: {skip_result['action_taken']}")
    print(f"   Symbols Skipped: {skip_result.get('symbols_skipped', [])}")
    print(f"   Modified Symbols: {list(skip_result['modified_data'].keys())}")
    
    print("\n4. Testing handle_zero_volume_days() with 'interpolate' strategy...")
    
    interp_result = interface.handle_zero_volume_days(
        test_data, strategy='interpolate', min_volume_threshold=1000
    )
    
    print(f"   Can Proceed: {interp_result['can_proceed']}")
    print(f"   Action Taken: {interp_result['action_taken']}")
    print(f"   Data Modified: {interp_result['data_modified']}")
    
    sym_with_zero = interp_result['modified_data']['SYMBOL_WITH_ZERO']
    zero_count_after = (sym_with_zero['Volume'] == 0).sum()
    print(f"   SYMBOL_WITH_ZERO zero volume after interp: {zero_count_after}")
    
    print("\n5. Testing handle_zero_volume_days() with 'warn' strategy...")
    
    warn_result = interface.handle_zero_volume_days(
        test_data, strategy='warn', min_volume_threshold=1000
    )
    
    print(f"   Can Proceed: {warn_result['can_proceed']}")
    print(f"   Action Taken: {warn_result['action_taken']}")
    print(f"   Data Modified: {warn_result['data_modified']}")
    
    print("\n6. Testing validate_volume_quality()...")
    
    quality = interface.validate_volume_quality(
        test_data, min_volume_threshold=1000, min_avg_volume=10000
    )
    
    print(f"   Is Valid: {quality['is_valid']}")
    print(f"   Issues: {quality['issues']}")
    print(f"   Warnings: {quality['warnings']}")
    print(f"   Volume Summary: {quality['volume_summary']}")
    
    print("\n7. Testing with real data (RELIANCE)...")
    
    real_data = interface.data_loader.load_data(
        ['RELIANCE.NS'], 
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    
    if real_data:
        real_detection = interface.detect_zero_volume_days(real_data, min_volume_threshold=1000)
        print(f"   RELIANCE Data Points: {len(real_data.get('RELIANCE.NS', []))}")
        print(f"   Total Zero Volume Days: {real_detection['total_zero_volume_days']}")
        print(f"   Total Low Volume Days: {real_detection['total_low_volume_days']}")
    
    print("\n" + "=" * 60)
    print("Phase 6.14.3: COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    assert detection['total_zero_volume_days'] == 3, "Should detect 3 zero volume days"
    assert detection['symbols_with_zero_volume'] == ['SYMBOL_WITH_ZERO'], "Should detect SYMBOL_WITH_ZERO"
    assert skip_result['can_proceed'] == True, "Skip should proceed"
    assert filter_result['data_modified'] == True, "Filter should modify data"
    assert zero_count_after == 0, "Interpolation should fill zero volumes"
    
    print("\nAll assertions passed!")
    return True

if __name__ == "__main__":
    from datetime import datetime
    try:
        test_zero_volume_handling()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
