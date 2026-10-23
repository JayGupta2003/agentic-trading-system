"""
Test file for Phase 6.14.8 - Validate Input Parameters
"""
import sys
sys.path.insert(0, r"J:\KAAM\Python\Quant\Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_validate_backtest_config():
    """Test validating backtest configuration parameters."""
    print("\n=== Test 6.14.8.1: validate_backtest_config ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.engine import BacktestConfig
    
    interface = BacktesterInterface()
    
    # Test valid config
    valid_config = BacktestConfig(
        initial_capital=1000000,
        max_positions=5,
        max_position_pct=20.0,
        min_conviction=6,
    )
    
    result_valid = interface.validate_backtest_config(valid_config)
    print(f"Valid config - is_valid: {result_valid['is_valid']}")
    print(f"Valid config - issues: {result_valid['issues']}")
    assert result_valid['is_valid'] == True, "Valid config should pass"
    assert len(result_valid['issues']) == 0, "Should have no issues"
    
    # Test invalid config - negative capital
    invalid_config = BacktestConfig(
        initial_capital=-1000,
        max_positions=0,
        max_position_pct=150.0,
    )
    
    result_invalid = interface.validate_backtest_config(invalid_config)
    print(f"Invalid config - is_valid: {result_invalid['is_valid']}")
    print(f"Invalid config - issues: {result_invalid['issues']}")
    assert result_invalid['is_valid'] == False, "Invalid config should fail"
    assert len(result_invalid['issues']) > 0, "Should have issues"
    
    print("[PASS] validate_backtest_config: PASSED")
    return result_invalid


def test_validate_date_range():
    """Test validating date ranges."""
    print("\n=== Test 6.14.8.2: validate_date_range ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Test valid date range
    result_valid = interface.validate_date_range(
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    print(f"Valid range - is_valid: {result_valid['is_valid']}")
    assert result_valid['is_valid'] == True, "Valid range should pass"
    
    # Test invalid - start after end
    result_invalid = interface.validate_date_range(
        start_date="2024-01-01",
        end_date="2023-01-01"
    )
    print(f"Invalid range - is_valid: {result_invalid['is_valid']}")
    print(f"Invalid range - issues: {result_invalid['issues']}")
    assert result_invalid['is_valid'] == False, "Invalid range should fail"
    
    # Test future dates
    result_future = interface.validate_date_range(
        start_date="2027-01-01",
        end_date="2028-01-01"
    )
    print(f"Future dates - is_valid: {result_future['is_valid']}")
    print(f"Future dates - warnings: {result_future['warnings']}")
    assert result_future['is_valid'] == True, "Future dates should be valid but with warnings"
    assert len(result_future['warnings']) > 0, "Should have warnings for future dates"
    
    print("[PASS] validate_date_range: PASSED")
    return result_valid


def test_validate_symbols():
    """Test validating symbol formats."""
    print("\n=== Test 6.14.8.3: validate_symbols ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Test valid symbols
    valid_symbols = ["RELIANCE.NS", "TCS.NS", "^NSEI"]
    result_valid = interface.validate_symbols(valid_symbols)
    print(f"Valid symbols - is_valid: {result_valid['is_valid']}")
    print(f"Valid symbols - issues: {result_valid['issues']}")
    assert result_valid['is_valid'] == True, "Valid symbols should pass"
    
    # Test invalid symbols
    invalid_symbols = ["", "INVALID", "A", "TOOLONG_SYMBOL_NAME.NS"]
    result_invalid = interface.validate_symbols(invalid_symbols)
    print(f"Invalid symbols - is_valid: {result_invalid['is_valid']}")
    print(f"Invalid symbols - issues: {result_invalid['issues']}")
    assert result_invalid['is_valid'] == False, "Invalid symbols should fail"
    assert len(result_invalid['issues']) > 0, "Should have issues"
    
    print("[PASS] validate_symbols: PASSED")
    return result_valid


def test_validate_risk_parameters():
    """Test validating risk parameters."""
    print("\n=== Test 6.14.8.4: validate_risk_parameters ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Test valid risk params
    valid_params = {
        "max_position_pct": 20.0,
        "max_drawdown_pct": 15.0,
        "stop_loss_atr": 2.0,
        "max_positions": 5,
    }
    
    result_valid = interface.validate_risk_parameters(valid_params)
    print(f"Valid params - is_valid: {result_valid['is_valid']}")
    assert result_valid['is_valid'] == True, "Valid params should pass"
    
    # Test invalid risk params
    invalid_params = {
        "max_position_pct": 100.0,
        "max_drawdown_pct": -5.0,
        "stop_loss_atr": 0,
        "max_positions": 50,
    }
    
    result_invalid = interface.validate_risk_parameters(invalid_params)
    print(f"Invalid params - is_valid: {result_invalid['is_valid']}")
    print(f"Invalid params - issues: {result_invalid['issues']}")
    assert result_invalid['is_valid'] == False, "Invalid params should fail"
    assert len(result_invalid['issues']) > 0, "Should have issues"
    
    print("[PASS] validate_risk_parameters: PASSED")
    return result_valid


def test_validate_signal_parameters():
    """Test validating signal parameters."""
    print("\n=== Test 6.14.8.5: validate_signal_parameters ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.signals import TradingSignal, Signal
    
    interface = BacktesterInterface()
    
    # Test valid signals
    valid_signals = [
        TradingSignal(symbol="RELIANCE.NS", signal=Signal.BUY, conviction=7, timestamp=datetime.now(), reasoning="Good setup"),
        TradingSignal(symbol="TCS.NS", signal=Signal.SELL, conviction=5, timestamp=datetime.now(), reasoning="Weakness"),
    ]
    
    result_valid = interface.validate_signal_parameters(valid_signals)
    print(f"Valid signals - is_valid: {result_valid['is_valid']}")
    assert result_valid['is_valid'] == True, "Valid signals should pass"
    
    # Test invalid signals
    invalid_signals = [
        TradingSignal(symbol="RELIANCE.NS", signal=Signal.BUY, conviction=0, timestamp=datetime.now(), reasoning=""),
        TradingSignal(symbol="", signal=Signal.SELL, conviction=15, timestamp=datetime.now(), reasoning="Test"),
    ]
    
    result_invalid = interface.validate_signal_parameters(invalid_signals)
    print(f"Invalid signals - is_valid: {result_invalid['is_valid']}")
    print(f"Invalid signals - issues: {result_invalid['issues']}")
    assert result_invalid['is_valid'] == False, "Invalid signals should fail"
    assert len(result_invalid['issues']) > 0, "Should have issues"
    
    print("[PASS] validate_signal_parameters: PASSED")
    return result_valid


def test_validate_all_parameters():
    """Test comprehensive parameter validation."""
    print("\n=== Test 6.14.8.6: validate_all_parameters ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.engine import BacktestConfig
    
    interface = BacktesterInterface()
    
    # Create valid parameters
    config = BacktestConfig(initial_capital=1000000, max_positions=5)
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    symbols = ["RELIANCE.NS", "TCS.NS"]
    risk_params = {"max_position_pct": 20.0, "max_drawdown_pct": 15.0}
    
    result = interface.validate_all_parameters(
        config=config,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        risk_parameters=risk_params,
    )
    
    print(f"Valid params - is_valid: {result['is_valid']}")
    print(f"Config valid: {result['validation_results']['config_valid']}")
    print(f"Dates valid: {result['validation_results']['dates_valid']}")
    print(f"Symbols valid: {result['validation_results']['symbols_valid']}")
    print(f"Risk valid: {result['validation_results']['risk_valid']}")
    
    assert result['is_valid'] == True, "Valid params should pass all checks"
    
    print("[PASS] validate_all_parameters: PASSED")
    return result


def run_all_tests():
    """Run all tests for phase 6.14.8."""
    print("=" * 60)
    print("Phase 6.14.8 - Validate Input Parameters")
    print("=" * 60)
    
    try:
        test_validate_backtest_config()
        test_validate_date_range()
        test_validate_symbols()
        test_validate_risk_parameters()
        test_validate_signal_parameters()
        test_validate_all_parameters()
        
        print("\n" + "=" * 60)
        print("ALL PHASE 6.14.8 TESTS PASSED!")
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
