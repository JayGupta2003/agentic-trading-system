"""
Test file for Phase 6.14.6 - Handle weekends/market holidays
"""
import sys
sys.path.insert(0, r"J:\KAAM\Python\Quant\Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_detect_weekends_and_holidays():
    """Test detecting weekends and holidays in data."""
    print("\n=== Test 6.14.6.1: detect_weekends_and_holidays ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Create test data spanning a month including weekends
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=45, freq='D')
    
    # Create DataFrame with dates including weekends
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
    })
    test_df.set_index('Date', inplace=True)
    
    # Add a known holiday (Republic Day - Jan 26, 2024)
    republic_day = pd.Timestamp('2024-01-26')
    test_df.loc[republic_day, 'Close'] = test_df.loc[republic_day, 'Open']  # No change (holiday)
    
    data = {"TEST.NS": test_df}
    
    result = interface.detect_weekends_and_holidays(data)
    
    print(f"Symbols analyzed: {result['symbols_analyzed']}")
    print(f"Total weekdays: {result['total_weekdays']}")
    print(f"Total weekends: {result['total_weekend_days']}")
    print(f"Total holidays: {result['total_holidays']}")
    print(f"Trading days: {result['trading_days']}")
    
    # Print weekend dates
    if result['detail']['TEST.NS']['weekend_dates']:
        print(f"Weekend dates (sample): {result['detail']['TEST.NS']['weekend_dates'][:3]}")
    
    # Print holiday dates
    if result['detail']['TEST.NS']['holiday_dates']:
        print(f"Holiday dates: {result['detail']['TEST.NS']['holiday_dates']}")
    
    # Check assertions
    assert result['symbols_analyzed'] == 1, "Should analyze 1 symbol"
    assert result['total_weekend_days'] > 0, "Should have weekend days"
    assert 'TEST.NS' in result['detail'], "Should have detail for TEST.NS"
    
    print("[PASS] detect_weekends_and_holidays: PASSED")
    return result


def test_handle_weekends_and_holidays():
    """Test handling weekends and holidays with different strategies."""
    print("\n=== Test 6.14.6.2: handle_weekends_and_holidays ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Create test data with weekends
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=30, freq='D')
    
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
    })
    test_df.set_index('Date', inplace=True)
    
    data = {"TEST.NS": test_df}
    
    # Test 'warn' strategy (default)
    result_warn = interface.handle_weekends_and_holidays(data, strategy="warn")
    print(f"Warn strategy - warnings: {len(result_warn['warnings'])}")
    assert result_warn['action_taken'] == 'warnings_only', "Should add warnings only"
    assert result_warn['can_proceed'] == True, "Should be able to proceed"
    assert len(result_warn['warnings']) > 0, "Should have warnings"
    
    # Test 'skip' strategy
    result_skip = interface.handle_weekends_and_holidays(data, strategy="skip")
    print(f"Skip strategy - days removed: {result_skip.get('days_removed', 0)}")
    assert result_skip['data_modified'] == True, "Should modify data"
    assert result_skip['modified_data'] is not None, "Should return modified data"
    
    # Check that weekends are removed
    modified_df = result_skip['modified_data']['TEST.NS']
    weekend_mask = modified_df.index.dayofweek.isin([5, 6])
    assert weekend_mask.sum() == 0, "Should have no weekends in filtered data"
    print(f"  Verified: Filtered data has no weekends ({len(modified_df)} days)")
    
    # Test 'filter' strategy (alias for skip)
    result_filter = interface.handle_weekends_and_holidays(data, strategy="filter")
    assert result_filter['data_modified'] == True, "Filter should modify data"
    
    print("[PASS] handle_weekends_and_holidays: PASSED")
    return result_skip


def test_get_nse_holidays():
    """Test getting NSE holidays for a year."""
    print("\n=== Test 6.14.6.4: get_nse_holidays ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Test 2024 holidays
    holidays_2024 = interface.get_nse_holidays(2024)
    print(f"NSE Holidays 2024: {len(holidays_2024)} holidays")
    for h in holidays_2024[:5]:
        print(f"  {h['name']}: {h['date']}")
    
    assert len(holidays_2024) > 0, "Should have holidays for 2024"
    
    # Verify some known holidays
    holiday_names = [h['name'] for h in holidays_2024]
    assert 'Republic Day' in holiday_names, "Should include Republic Day"
    assert 'Independence Day' in holiday_names, "Should include Independence Day"
    
    # Test 2025 holidays
    holidays_2025 = interface.get_nse_holidays(2025)
    print(f"NSE Holidays 2025: {len(holidays_2025)} holidays")
    assert len(holidays_2025) > 0, "Should have holidays for 2025"
    
    print("[PASS] get_nse_holidays: PASSED")
    return holidays_2024


def test_validate_trading_calendar():
    """Test validating trading calendar."""
    print("\n=== Test 6.14.6.3: validate_trading_calendar ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Create test data with various gaps
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=60, freq='D')
    
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
    })
    test_df.set_index('Date', inplace=True)
    
    data = {"TEST.NS": test_df}
    
    result = interface.validate_trading_calendar(data, max_consecutive_holidays=3)
    
    print(f"Is valid: {result['is_valid']}")
    print(f"Market status: {result['market_status']}")
    print(f"Issues: {result['issues']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Calendar summary - trading days: {result['calendar_summary']['trading_days']}")
    print(f"Calendar summary - non-trading: {result['calendar_summary']['non_trading_days']}")
    
    # Check structure
    assert 'is_valid' in result, "Should have is_valid field"
    assert 'market_status' in result, "Should have market_status field"
    assert 'calendar_summary' in result, "Should have calendar_summary"
    assert 'trading_days' in result['calendar_summary'], "Should have trading_days"
    assert 'non_trading_days' in result['calendar_summary'], "Should have non_trading_days"
    
    print("[PASS] validate_trading_calendar: PASSED")
    return result


def test_integration():
    """Test full integration with backtester."""
    print("\n=== Test 6.14.6.5: Integration Test ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.engine import BacktestConfig
    
    interface = BacktesterInterface()
    
    # Create test data with weekends
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=30, freq='D')
    
    test_df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(100, 110, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
    })
    test_df.set_index('Date', inplace=True)
    
    # Add technical indicators
    test_df['SMA_20'] = test_df['Close'].rolling(5).mean()
    test_df['SMA_50'] = test_df['Close'].rolling(10).mean()
    test_df['RSI'] = 50.0
    test_df['MACD'] = 0.0
    test_df['MACD_Signal'] = 0.0
    test_df['ATR'] = 2.0
    
    data = {"TEST.NS": test_df}
    
    # Test handle_weekends_and_holidays with skip strategy
    result = interface.handle_weekends_and_holidays(data, strategy="skip")
    
    print(f"Holidays/Weekends result - can_proceed: {result.get('can_proceed', 'N/A')}")
    print(f"Holidays/Weekends result - warnings: {len(result.get('warnings', []))}")
    
    # Check weekend removal worked
    if result.get('modified_data'):
        modified_df = result['modified_data'].get('TEST.NS')
        if modified_df is not None:
            weekend_mask = modified_df.index.dayofweek.isin([5, 6])
            assert weekend_mask.sum() == 0, "Prepared data should have no weekends"
            print(f"  Verified: Modified data has no weekends ({len(modified_df)} days)")
    
    # Test validate_trading_calendar
    calendar_result = interface.validate_trading_calendar(data)
    print(f"Calendar validation - is_valid: {calendar_result.get('is_valid', 'N/A')}")
    print(f"Calendar validation - market_status: {calendar_result.get('market_status', 'N/A')}")
    
    print("[PASS] Integration test: PASSED")
    return result


def run_all_tests():
    """Run all tests for phase 6.14.6."""
    print("=" * 60)
    print("Phase 6.14.6 - Weekends/Market Holidays")
    print("=" * 60)
    
    try:
        test_detect_weekends_and_holidays()
        test_handle_weekends_and_holidays()
        test_validate_trading_calendar()
        test_get_nse_holidays()
        test_integration()
        
        print("\n" + "=" * 60)
        print("ALL PHASE 6.14.6 TESTS PASSED!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
