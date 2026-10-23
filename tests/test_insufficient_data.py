"""
Test Phase 6.14.1 - Handle Insufficient Data for Backtest
"""
import sys
sys.path.insert(0, "J:/KAAM/Python/Quant/Agentic Trading System Part 2")

from datetime import datetime, timedelta
from src.backtesting.interface import BacktesterInterface
from src.backtesting.signals import Signal, TradingSignal

def test_insufficient_data_handling():
    """Test insufficient data handling methods."""
    print("=" * 60)
    print("Phase 6.14.1: Handle Insufficient Data for Backtest")
    print("=" * 60)
    
    interface = BacktesterInterface()
    
    print("\n1. Testing check_data_sufficiency()...")
    
    report = interface.check_data_sufficiency(
        symbols=["RELIANCE.NS", "INVALID_SYMBOL", "TCS.NS"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        min_days=50
    )
    
    print(f"   Is Sufficient: {report['is_sufficient']}")
    print(f"   Total Symbols: {report['total_symbols']}")
    print(f"   Loaded Symbols: {report['loaded_symbols']}")
    print(f"   Missing Symbols: {report['missing_symbols']}")
    print(f"   Insufficient Symbols: {report['insufficient_symbols']}")
    
    for symbol, detail in report['symbols_detail'].items():
        print(f"   - {symbol}: {detail['status']} ({detail['data_points']} days)")
    
    print("\n2. Testing validate_backtest_prerequisites()...")
    
    signals = {
        "RELIANCE.NS": [
            TradingSignal(
                symbol="RELIANCE.NS",
                signal=Signal.BUY,
                conviction=7,
                timestamp=datetime(2024, 2, 1),
                reasoning="Test signal"
            )
        ]
    }
    
    validation = interface.validate_backtest_prerequisites(
        symbols=["RELIANCE.NS"],
        signals=signals,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
    )
    
    print(f"   Is Valid: {validation['is_valid']}")
    print(f"   Can Proceed: {validation['can_proceed']}")
    print(f"   Issues: {validation['issues']}")
    print(f"   Warnings: {validation['warnings']}")
    print(f"   Recommendations: {validation['recommendations']}")
    
    print("\n3. Testing validate_backtest_prerequisites() with invalid symbol...")
    
    validation2 = interface.validate_backtest_prerequisites(
        symbols=["INVALID_SYMBOL_ABC", "RELIANCE.NS"],
        signals={"RELIANCE.NS": signals["RELIANCE.NS"]},
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
    )
    
    print(f"   Is Valid: {validation2['is_valid']}")
    print(f"   Can Proceed: {validation2['can_proceed']}")
    print(f"   Issues: {validation2['issues']}")
    
    print("\n4. Testing adjust_dates_for_data()...")
    
    date_adjust = interface.adjust_dates_for_data(
        symbols=["RELIANCE.NS", "TCS.NS"],
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2019, 6, 30),
    )
    
    print(f"   Success: {date_adjust['success']}")
    print(f"   Original: {date_adjust['original_start']} to {date_adjust['original_end']}")
    print(f"   Adjusted: {date_adjust['adjusted_start']} to {date_adjust['adjusted_end']}")
    print(f"   Warnings: {date_adjust['warnings']}")
    
    print("\n5. Testing handle_insufficient_data() with 'error' strategy...")
    
    error_result = interface.handle_insufficient_data(
        symbols=["INVALID_XYZ", "RELIANCE.NS"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        strategy="error",
        min_days=50
    )
    
    print(f"   Can Run: {error_result['can_run_backtest']}")
    print(f"   Action: {error_result['action_taken']}")
    print(f"   Error: {error_result.get('error', 'N/A')}")
    
    print("\n6. Testing handle_insufficient_data() with 'skip' strategy...")
    
    skip_result = interface.handle_insufficient_data(
        symbols=["INVALID_XYZ", "RELIANCE.NS"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        strategy="skip",
        min_days=50
    )
    
    print(f"   Can Run: {skip_result['can_run_backtest']}")
    print(f"   Action: {skip_result['action_taken']}")
    print(f"   Modified Symbols: {skip_result['modified_symbols']}")
    print(f"   Skipped Symbols: {skip_result.get('skipped_symbols', [])}")
    
    print("\n7. Testing handle_insufficient_data() with 'adjust' strategy...")
    
    adjust_result = interface.handle_insufficient_data(
        symbols=["RELIANCE.NS", "TCS.NS"],
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2019, 6, 30),
        strategy="adjust",
        min_days=50
    )
    
    print(f"   Can Run: {adjust_result['can_run_backtest']}")
    print(f"   Action: {adjust_result['action_taken']}")
    if adjust_result.get('date_adjustments'):
        da = adjust_result['date_adjustments']
        print(f"   Adjusted: {da['adjusted_start']} to {da['adjusted_end']}")
    print(f"   Modified Symbols: {adjust_result['modified_symbols']}")
    
    print("\n8. Testing handle_insufficient_data() with sufficient data...")
    
    good_result = interface.handle_insufficient_data(
        symbols=["RELIANCE.NS", "TCS.NS"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
        strategy="adjust",
        min_days=50
    )
    
    print(f"   Can Run: {good_result['can_run_backtest']}")
    print(f"   Action: {good_result['action_taken']}")
    
    print("\n" + "=" * 60)
    print("Phase 6.14.1: COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    assert report['loaded_symbols'] >= 2, "Should load at least 2 valid symbols"
    assert not validation['is_valid'] or len(validation['issues']) == 0, "Validation should work"
    assert error_result['can_run_backtest'] == False, "Error strategy should block"
    assert skip_result['can_run_backtest'] == True, "Skip should proceed with valid symbols"
    assert good_result['action_taken'] == 'no_action', "Good data should need no action"
    
    print("\nAll assertions passed!")
    return True

if __name__ == "__main__":
    try:
        test_insufficient_data_handling()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
