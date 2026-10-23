"""
Test Phase 6.14.5 - Handle Circuit Limit Hits (Upper/Lower)
"""
import sys
sys.path.insert(0, "J:/KAAM/Python/Quant/Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime
from src.backtesting.interface import BacktesterInterface

def create_test_data_with_circuit_hits():
    """Create test data with potential circuit limit hits."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    dfs = {}
    
    for symbol, base, circuit_type in [
        ('SYMBOL_NORMAL', 2500, None),
        ('SYMBOL_CIRCUIT_UP', 3500, 'upper'),
        ('SYMBOL_CIRCUIT_DOWN', 4500, 'lower')
    ]:
        prices = [base]
        for j in range(len(dates) - 1):
            change = np.random.randn() * 50
            if circuit_type == 'upper' and j == 15:
                change = base * 0.22
            elif circuit_type == 'lower' and j == 20:
                change = -base * 0.22
            new_price = prices[-1] + change
            new_price = max(new_price, base * 0.5)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'Open': [p + np.random.randn() * 10 for p in prices],
            'High': [p + abs(np.random.randn() * 20) for p in prices],
            'Low': [p - abs(np.random.randn() * 20) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates)),
        }, index=dates)
        dfs[symbol] = df
    
    return dfs

def test_circuit_limit_handling():
    """Test circuit limit hit handling methods."""
    print("=" * 60)
    print("Phase 6.14.5: Handle Circuit Limit Hits (Upper/Lower)")
    print("=" * 60)
    
    interface = BacktesterInterface()
    
    print("\n1. Testing detect_circuit_limit_hits()...")
    
    test_data = create_test_data_with_circuit_hits()
    
    detection = interface.detect_circuit_limit_hits(
        test_data, upper_limit_pct=20.0, lower_limit_pct=20.0, lookback_days=30
    )
    
    print(f"   Symbols Analyzed: {detection['symbols_analyzed']}")
    print(f"   Total Upper Hits: {detection['total_upper_hits']}")
    print(f"   Total Lower Hits: {detection['total_lower_hits']}")
    print(f"   Symbols with Circuit Hits: {detection['symbols_with_circuit_hits']}")
    print(f"   Symbols without Issues: {detection['symbols_without_issues']}")
    
    for symbol, detail in detection['detail'].items():
        print(f"   - {symbol}: upper={detail['upper_circuit_hits']}, lower={detail['lower_circuit_hits']}, has_hits={detail['has_circuit_hits']}")
        if detail['upper_hit_dates']:
            for hit in detail['upper_hit_dates']:
                print(f"     Upper: {hit['date']}, change={hit['change_pct']:.1f}%")
        if detail['lower_hit_dates']:
            for hit in detail['lower_hit_dates']:
                print(f"     Lower: {hit['date']}, change={hit['change_pct']:.1f}%")
    
    print("\n2. Testing handle_circuit_limit_hits() with 'skip' strategy...")
    
    skip_result = interface.handle_circuit_limit_hits(
        test_data, strategy='skip', upper_limit_pct=20.0, lower_limit_pct=20.0
    )
    
    print(f"   Can Proceed: {skip_result['can_proceed']}")
    print(f"   Action Taken: {skip_result['action_taken']}")
    print(f"   Symbols Skipped: {skip_result.get('symbols_skipped', [])}")
    print(f"   Modified Symbols: {list(skip_result['modified_data'].keys())}")
    
    print("\n3. Testing handle_circuit_limit_hits() with 'warn' strategy...")
    
    warn_result = interface.handle_circuit_limit_hits(
        test_data, strategy='warn', upper_limit_pct=20.0, lower_limit_pct=20.0
    )
    
    print(f"   Can Proceed: {warn_result['can_proceed']}")
    print(f"   Action Taken: {warn_result['action_taken']}")
    print(f"   Warnings: {warn_result['warnings']}")
    
    print("\n4. Testing handle_circuit_limit_hits() with 'adjust' strategy...")
    
    adjust_result = interface.handle_circuit_limit_hits(
        test_data, strategy='adjust', upper_limit_pct=20.0, lower_limit_pct=20.0
    )
    
    print(f"   Can Proceed: {adjust_result['can_proceed']}")
    print(f"   Action Taken: {adjust_result['action_taken']}")
    print(f"   Data Modified: {adjust_result['data_modified']}")
    
    print("\n5. Testing validate_market_conditions()...")
    
    conditions = interface.validate_market_conditions(
        test_data, max_circuit_hits_per_symbol=2, max_total_circuit_hits=5
    )
    
    print(f"   Is Valid: {conditions['is_valid']}")
    print(f"   Market Status: {conditions['market_status']}")
    print(f"   Issues: {conditions['issues']}")
    print(f"   Warnings: {conditions['warnings']}")
    print(f"   Circuit Summary: {conditions['circuit_summary']}")
    
    print("\n6. Testing with real data (RELIANCE)...")
    
    real_data = interface.data_loader.load_data(
        ['RELIANCE.NS'],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )
    
    if real_data:
        real_detection = interface.detect_circuit_limit_hits(real_data, lookback_days=120)
        print(f"   RELIANCE Data Points: {len(real_data.get('RELIANCE.NS', []))}")
        print(f"   Total Upper Hits: {real_detection['total_upper_hits']}")
        print(f"   Total Lower Hits: {real_detection['total_lower_hits']}")
        print(f"   Market Status: NORMAL" if real_detection['total_upper_hits'] == 0 and real_detection['total_lower_hits'] == 0 else "   Market Status: Circuit hits detected")
    
    print("\n" + "=" * 60)
    print("Phase 6.14.5: COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    assert detection['symbols_analyzed'] == 3, "Should analyze 3 symbols"
    assert skip_result['can_proceed'] == True, "Skip should proceed"
    assert warn_result['can_proceed'] == True, "Warn should proceed"
    assert adjust_result['data_modified'] == True, "Adjust should modify data"
    
    print("\nAll assertions passed!")
    return True

if __name__ == "__main__":
    try:
        test_circuit_limit_handling()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
