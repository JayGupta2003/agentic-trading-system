"""
Test file for Phase 6.14.7 - Concurrent Order Conflicts
"""
import sys
sys.path.insert(0, r"J:\KAAM\Python\Quant\Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_detect_concurrent_order_conflicts():
    """Test detecting concurrent order conflicts."""
    print("\n=== Test 6.14.7.1: detect_concurrent_order_conflicts ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.signals import TradingSignal, Signal
    
    interface = BacktesterInterface()
    
    base_date = pd.Timestamp('2024-01-15')
    
    signals = [
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.BUY,
            conviction=7,
            timestamp=base_date,
            reasoning="Bullish setup"
        ),
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.SELL,
            conviction=8,
            timestamp=base_date,
            reasoning="Bearish reversal"
        ),
        TradingSignal(
            symbol="TCS.NS",
            signal=Signal.BUY,
            conviction=6,
            timestamp=base_date,
            reasoning="Technical breakout"
        ),
    ]
    
    result = interface.detect_concurrent_order_conflicts(signals)
    
    print(f"Total signals: {result['total_signals']}")
    print(f"Conflicts found: {result['conflicts_found']}")
    print(f"Conflicting symbols: {result['conflicting_symbols']}")
    print(f"Resolution actions: {result['resolution_actions']}")
    
    assert result['total_signals'] == 3, "Should have 3 signals"
    assert result['conflicts_found'] == 1, "Should have 1 conflict"
    assert "RELIANCE.NS" in result['conflicting_symbols'], "RELIANCE should conflict"
    
    detail = result['detail']['RELIANCE.NS']
    assert detail['conflict_type'] == 'opposite_signals', "Should be opposite signals"
    assert len(detail['signals']) == 2, "Should have 2 conflicting signals"
    
    print("[PASS] detect_concurrent_order_conflicts: PASSED")
    return result


def test_handle_concurrent_order_conflicts():
    """Test handling concurrent order conflicts with different strategies."""
    print("\n=== Test 6.14.7.2: handle_concurrent_order_conflicts ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.signals import TradingSignal, Signal
    
    interface = BacktesterInterface()
    
    base_date = pd.Timestamp('2024-01-15')
    
    signals = [
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.BUY,
            conviction=7,
            timestamp=base_date,
            reasoning="Bullish setup"
        ),
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.SELL,
            conviction=8,
            timestamp=base_date,
            reasoning="Bearish reversal"
        ),
        TradingSignal(
            symbol="TCS.NS",
            signal=Signal.BUY,
            conviction=6,
            timestamp=base_date,
            reasoning="Technical breakout"
        ),
    ]
    
    # Test 'prioritize_conviction' strategy (default)
    result_priority = interface.handle_concurrent_order_conflicts(
        signals, strategy="prioritize_conviction"
    )
    print(f"Prioritize conviction - kept: {len(result_priority['resolved_signals'])}")
    assert len(result_priority['resolved_signals']) == 2, "Should keep 2 signals"
    assert result_priority['action_taken'] == 'kept_highest_conviction', "Should keep highest conviction"
    
    # Find RELIANCE signal in result
    reliance_signals = [s for s in result_priority['resolved_signals'] if s.symbol == "RELIANCE.NS"]
    assert len(reliance_signals) == 1, "Should have 1 RELIANCE signal"
    assert reliance_signals[0].signal == Signal.SELL, "Should keep SELL (conviction 8)"
    
    # Test 'skip' strategy
    result_skip = interface.handle_concurrent_order_conflicts(
        signals, strategy="skip"
    )
    print(f"Skip strategy - kept: {len(result_skip['resolved_signals'])}")
    assert len(result_skip['resolved_signals']) == 1, "Should keep only TCS signal"
    assert result_skip['action_taken'] == 'skipped_conflicting_signals', "Should skip conflicts"
    
    # Test 'error' strategy
    result_error = interface.handle_concurrent_order_conflicts(
        signals, strategy="error"
    )
    print(f"Error strategy - errors: {len(result_error['errors'])}")
    assert result_error['can_proceed'] == False, "Should not proceed"
    assert len(result_error['errors']) > 0, "Should have errors"
    
    # Test 'keep_first' strategy
    result_first = interface.handle_concurrent_order_conflicts(
        signals, strategy="keep_first"
    )
    print(f"Keep first strategy - kept: {len(result_first['resolved_signals'])}")
    assert len(result_first['resolved_signals']) == 2, "Should keep 2 signals"
    reliance_signals = [s for s in result_first['resolved_signals'] if s.symbol == "RELIANCE.NS"]
    assert reliance_signals[0].signal == Signal.BUY, "Should keep first (BUY)"
    
    print("[PASS] handle_concurrent_order_conflicts: PASSED")
    return result_priority


def test_validate_order_consistency():
    """Test validating order consistency."""
    print("\n=== Test 6.14.7.3: validate_order_consistency ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.signals import TradingSignal, Signal
    
    interface = BacktesterInterface()
    
    base_date = pd.Timestamp('2024-01-15')
    
    # Test case 1: No issues
    clean_signals = [
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.BUY,
            conviction=7,
            timestamp=base_date,
            reasoning="Bullish"
        ),
        TradingSignal(
            symbol="TCS.NS",
            signal=Signal.SELL,
            conviction=6,
            timestamp=base_date,
            reasoning="Bearish"
        ),
    ]
    
    result_clean = interface.validate_order_consistency(clean_signals)
    print(f"Clean signals - is_valid: {result_clean['is_valid']}")
    assert result_clean['is_valid'] == True, "Clean signals should be valid"
    assert len(result_clean['issues']) == 0, "Should have no issues"
    
    # Test case 2: With conflicts
    conflicting_signals = [
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.BUY,
            conviction=7,
            timestamp=base_date,
            reasoning="Bullish"
        ),
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.SELL,
            conviction=8,
            timestamp=base_date,
            reasoning="Bearish"
        ),
    ]
    
    result_conflict = interface.validate_order_consistency(conflicting_signals)
    print(f"Conflicting signals - is_valid: {result_conflict['is_valid']}")
    print(f"Issues: {result_conflict['issues']}")
    assert result_conflict['is_valid'] == False, "Conflicting signals should not be valid"
    assert len(result_conflict['issues']) > 0, "Should have issues"
    
    print("[PASS] validate_order_consistency: PASSED")
    return result_conflict


def test_resolve_order_priority():
    """Test resolving order priority."""
    print("\n=== Test 6.14.7.4: resolve_order_priority ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.signals import TradingSignal, Signal
    
    interface = BacktesterInterface()
    
    base_date = pd.Timestamp('2024-01-15')
    
    conflicting_signals = [
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.BUY,
            conviction=7,
            timestamp=base_date,
            reasoning="Bullish"
        ),
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.STRONG_SELL,
            conviction=9,
            timestamp=base_date,
            reasoning="Very bearish"
        ),
        TradingSignal(
            symbol="RELIANCE.NS",
            signal=Signal.HOLD,
            conviction=5,
            timestamp=base_date,
            reasoning="Neutral"
        ),
    ]
    
    result = interface.resolve_order_priority(conflicting_signals)
    
    print(f"Resolved to: {result['signal'].name} (conviction {result['conviction']})")
    print(f"Resolution basis: {result['basis']}")
    
    assert result['signal'] == Signal.STRONG_SELL, "Should resolve to STRONG_SELL (highest priority)"
    assert result['conviction'] == 9, "Should use conviction 9"
    
    print("[PASS] resolve_order_priority: PASSED")
    return result


def test_integration():
    """Test full integration with backtest flow."""
    print("\n=== Test 6.14.7.5: Integration Test ===")
    
    from src.backtesting.interface import BacktesterInterface
    from src.backtesting.signals import TradingSignal, Signal
    
    interface = BacktesterInterface()
    
    base_date = pd.Timestamp('2024-01-15')
    
    # Create a realistic set of signals with some conflicts
    signals = [
        TradingSignal(symbol="RELIANCE.NS", signal=Signal.BUY, conviction=7, timestamp=base_date, reasoning="Bullish"),
        TradingSignal(symbol="RELIANCE.NS", signal=Signal.SELL, conviction=6, timestamp=base_date, reasoning="Profit taking"),
        TradingSignal(symbol="TCS.NS", signal=Signal.BUY, conviction=8, timestamp=base_date, reasoning="Strong breakout"),
        TradingSignal(symbol="HDFCBANK.NS", signal=Signal.HOLD, conviction=5, timestamp=base_date, reasoning="Wait"),
        TradingSignal(symbol="INFY.NS", signal=Signal.STRONG_BUY, conviction=9, timestamp=base_date, reasoning="Momentum"),
        TradingSignal(symbol="INFY.NS", signal=Signal.SELL, conviction=4, timestamp=base_date, reasoning="Overbought"),
    ]
    
    # Step 1: Detect conflicts
    detection = interface.detect_concurrent_order_conflicts(signals)
    print(f"Conflicts detected: {detection['conflicts_found']}")
    assert detection['conflicts_found'] == 2, "Should have 2 conflicts"
    
    # Step 2: Handle conflicts
    resolved = interface.handle_concurrent_order_conflicts(signals, strategy="prioritize_conviction")
    print(f"Signals after resolution: {len(resolved['resolved_signals'])}")
    assert len(resolved['resolved_signals']) == 4, "Should have 4 signals (2 conflicts resolved to 1 each)"
    
    # Step 3: Validate consistency
    validation = interface.validate_order_consistency(resolved['resolved_signals'])
    print(f"Validation result: is_valid={validation['is_valid']}")
    assert validation['is_valid'] == True, "Resolved signals should be valid"
    
    print("[PASS] Integration test: PASSED")
    return resolved


def run_all_tests():
    """Run all tests for phase 6.14.7."""
    print("=" * 60)
    print("Phase 6.14.7 - Concurrent Order Conflicts")
    print("=" * 60)
    
    try:
        test_detect_concurrent_order_conflicts()
        test_handle_concurrent_order_conflicts()
        test_validate_order_consistency()
        test_resolve_order_priority()
        test_integration()
        
        print("\n" + "=" * 60)
        print("ALL PHASE 6.14.7 TESTS PASSED!")
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
