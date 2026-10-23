"""
Test Phase 6.14.4 - Handle Stock Delisting / Suspension
"""
import sys
sys.path.insert(0, "J:/KAAM/Python/Quant/Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime
from src.backtesting.interface import BacktesterInterface

def create_test_data_with_delist():
    """Create test data including potentially delisted/suspended symbols."""
    np.random.seed(42)
    
    dates_normal = pd.date_range(start='2024-01-01', periods=120, freq='D')
    df_normal = pd.DataFrame({
        'Open': np.random.randn(len(dates_normal)) * 10 + 2500,
        'High': np.random.randn(len(dates_normal)) * 10 + 2550,
        'Low': np.random.randn(len(dates_normal)) * 10 + 2450,
        'Close': np.random.randn(len(dates_normal)) * 10 + 2500,
        'Volume': np.random.randint(1000000, 5000000, len(dates_normal)),
    }, index=dates_normal)
    
    dates_suspended = pd.date_range(start='2024-01-01', periods=60, freq='D')
    volumes_suspended = np.random.randint(1000000, 5000000, len(dates_suspended))
    volumes_suspended[-20:] = 0
    df_suspended = pd.DataFrame({
        'Open': np.random.randn(len(dates_suspended)) * 10 + 3500,
        'High': np.random.randn(len(dates_suspended)) * 10 + 3550,
        'Low': np.random.randn(len(dates_suspended)) * 10 + 3450,
        'Close': np.random.randn(len(dates_suspended)) * 10 + 3500,
        'Volume': volumes_suspended,
    }, index=dates_suspended)
    
    dates_suspicious = pd.date_range(start='2024-01-01', periods=30, freq='D')
    df_suspicious = pd.DataFrame({
        'Open': np.random.randn(len(dates_suspicious)) * 10 + 4500,
        'High': np.random.randn(len(dates_suspicious)) * 10 + 4550,
        'Low': np.random.randn(len(dates_suspicious)) * 10 + 4450,
        'Close': np.random.randn(len(dates_suspicious)) * 10 + 4500,
        'Volume': np.random.randint(500000, 2000000, len(dates_suspicious)),
    }, index=dates_suspicious)
    
    return {
        'SYMBOL_NORMAL': df_normal,
        'SYMBOL_SUSPENDED': df_suspended,
        'SYMBOL_SHORT': df_suspicious,
    }

def test_delist_handling():
    """Test delisting/suspension handling methods."""
    print("=" * 60)
    print("Phase 6.14.4: Handle Stock Delisting / Suspension")
    print("=" * 60)
    
    interface = BacktesterInterface()
    
    print("\n1. Testing detect_delist_or_suspension()...")
    
    test_data = create_test_data_with_delist()
    symbols = ['SYMBOL_NORMAL', 'SYMBOL_SUSPENDED', 'SYMBOL_SHORT', 'SYMBOL_DELISTED']
    test_data['SYMBOL_DELISTED'] = None
    
    detection = interface.detect_delist_or_suspension(symbols, test_data, min_data_points=100)
    
    print(f"   Symbols Checked: {detection['symbols_checked']}")
    print(f"   Potentially Delisted: {detection['potentially_delisted']}")
    print(f"   Potentially Suspended: {detection['potentially_suspended']}")
    print(f"   Active Symbols: {detection['active_symbols']}")
    
    for symbol, detail in detection['detail'].items():
        print(f"   - {symbol}: status={detail['status']}, points={detail['data_points']}, issues={detail['issues']}")
    
    print("\n2. Testing handle_delist_or_suspend() with 'skip' strategy...")
    
    skip_result = interface.handle_delist_or_suspend(
        symbols, test_data, strategy='skip', min_data_points=100
    )
    
    print(f"   Can Proceed: {skip_result['can_proceed']}")
    print(f"   Action Taken: {skip_result['action_taken']}")
    print(f"   Symbols Skipped: {skip_result.get('symbols_skipped', [])}")
    print(f"   Modified Symbols: {list(skip_result['modified_data'].keys())}")
    
    print("\n3. Testing handle_delist_or_suspend() with 'error' strategy...")
    
    error_result = interface.handle_delist_or_suspend(
        symbols, test_data, strategy='error', min_data_points=100
    )
    
    print(f"   Can Proceed: {error_result['can_proceed']}")
    print(f"   Action Taken: {error_result['action_taken']}")
    print(f"   Errors: {error_result['errors']}")
    
    print("\n4. Testing handle_delist_or_suspend() with 'warn' strategy...")
    
    warn_result = interface.handle_delist_or_suspend(
        symbols, test_data, strategy='warn', min_data_points=100
    )
    
    print(f"   Can Proceed: {warn_result['can_proceed']}")
    print(f"   Action Taken: {warn_result['action_taken']}")
    print(f"   Warnings: {warn_result['warnings']}")
    
    print("\n5. Testing validate_symbols_for_backtest()...")
    
    validation = interface.validate_symbols_for_backtest(symbols, test_data, min_data_points=100)
    
    print(f"   Is Valid: {validation['is_valid']}")
    print(f"   Can Proceed: {validation['can_proceed']}")
    print(f"   Issues: {validation['issues']}")
    print(f"   Warnings: {validation['warnings']}")
    print(f"   Summary: {validation['summary']}")
    
    print("\n6. Testing with real data (NIFTY stocks)...")
    
    real_symbols = ['RELIANCE.NS', 'TCS.NS', 'INVALID_SYMBOL_XYZ']
    real_data = interface.data_loader.load_data(
        real_symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    
    real_validation = interface.validate_symbols_for_backtest(real_symbols, real_data, min_data_points=100)
    
    print(f"   Real Symbols: {real_symbols}")
    print(f"   Loaded Data: {list(real_data.keys())}")
    print(f"   Is Valid: {real_validation['is_valid']}")
    print(f"   Summary: {real_validation['summary']}")
    
    print("\n" + "=" * 60)
    print("Phase 6.14.4: COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    assert 'SYMBOL_DELISTED' in detection['potentially_delisted'], "Should detect delisted symbol"
    assert 'SYMBOL_SUSPENDED' in detection['potentially_suspended'], "Should detect suspended symbol"
    assert skip_result['can_proceed'] == True, "Skip should proceed with valid symbols"
    assert error_result['can_proceed'] == False, "Error should block"
    assert 'SYMBOL_NORMAL' in detection['active_symbols'], "Normal symbol should be active"
    
    print("\nAll assertions passed!")
    return True

if __name__ == "__main__":
    try:
        test_delist_handling()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
