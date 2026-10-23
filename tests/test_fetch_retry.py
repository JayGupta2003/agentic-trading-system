"""
Test file for Phase 6.14.10 - Retry Logic for Data Fetch Failures
"""
import sys
sys.path.insert(0, r"J:\KAAM\Python\Quant\Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime


def test_fetch_with_retry():
    """Test fetch with retry logic."""
    print("\n=== Test 6.14.10.1: fetch_with_retry ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    call_count = 0
    
    def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return {"data": "success"}
    
    result = interface.fetch_with_retry(
        fetch_func=failing_func,
        max_retries=5,
        delay_seconds=0.1
    )
    
    print(f"Fetch with retry - success: {result['success']}")
    print(f"Fetch with retry - attempts: {result['attempts']}")
    print(f"Fetch with retry - data: {result['data']}")
    assert result['success'] == True, "Should succeed after retries"
    assert result['attempts'] == 3, "Should take 3 attempts"
    
    print("[PASS] fetch_with_retry: PASSED")
    return result


def test_fetch_with_retry_exhausted():
    """Test fetch with retry when all retries fail."""
    print("\n=== Test 6.14.10.2: fetch_with_retry_exhausted ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    def always_fails():
        raise Exception("Permanent failure")
    
    result = interface.fetch_with_retry(
        fetch_func=always_fails,
        max_retries=3,
        delay_seconds=0.1
    )
    
    print(f"Fetch exhausted - success: {result['success']}")
    print(f"Fetch exhausted - attempts: {result['attempts']}")
    print(f"Fetch exhausted - error: {result.get('error', 'N/A')}")
    assert result['success'] == False, "Should fail after retries exhausted"
    assert result['attempts'] == 3, "Should try 3 times"
    assert 'Permanent failure' in result.get('error', ''), "Should have error message"
    
    print("[PASS] fetch_with_retry_exhausted: PASSED")
    return result


def test_retry_with_exponential_backoff():
    """Test retry with exponential backoff."""
    print("\n=== Test 6.14.10.3: retry_with_exponential_backoff ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    call_times = []
    
    def timed_func():
        call_times.append(datetime.now())
        if len(call_times) < 2:
            raise Exception("Temporary")
        return "success"
    
    result = interface.retry_with_exponential_backoff(
        func=timed_func,
        max_retries=5,
        base_delay=0.1
    )
    
    print(f"Exponential backoff - success: {result['success']}")
    print(f"Exponential backoff - attempts: {result['attempts']}")
    
    if len(call_times) >= 2:
        delay1 = (call_times[1] - call_times[0]).total_seconds()
        print(f"Exponential backoff - delay: {delay1:.3f}s (expected ~0.1s)")
        assert delay1 >= 0.08, "Should have delay between attempts"
    
    assert result['success'] == True, "Should succeed"
    
    print("[PASS] retry_with_exponential_backoff: PASSED")
    return result


def test_get_retry_stats():
    """Test getting retry statistics."""
    print("\n=== Test 6.14.10.4: get_retry_stats ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Make some retry calls
    interface.fetch_with_retry(lambda: "success", max_retries=1)
    
    call_count = 0
    def fail_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Fail")
        return "success"
    
    interface.fetch_with_retry(fail_once, max_retries=3, delay_seconds=0.01)
    
    stats = interface.get_retry_stats()
    
    print(f"Retry stats - total_retries: {stats['total_retries']}")
    print(f"Retry stats - successful_retries: {stats['successful_retries']}")
    print(f"Retry stats - failed_retries: {stats['failed_retries']}")
    
    assert stats['total_retries'] >= 2, "Should have at least 2 retry attempts"
    
    print("[PASS] get_retry_stats: PASSED")
    return stats


def test_clear_retry_stats():
    """Test clearing retry statistics."""
    print("\n=== Test 6.14.10.5: clear_retry_stats ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Make some retry calls
    interface.fetch_with_retry(lambda: "success", max_retries=1)
    
    stats_before = interface.get_retry_stats()
    print(f"Stats before clear - total: {stats_before['total_retries']}")
    
    result = interface.clear_retry_stats()
    print(f"Clear stats - success: {result['success']}")
    
    stats_after = interface.get_retry_stats()
    print(f"Stats after clear - total: {stats_after['total_retries']}")
    
    assert stats_after['total_retries'] == 0, "Stats should be cleared"
    
    print("[PASS] clear_retry_stats: PASSED")
    return result


def run_all_tests():
    """Run all tests for phase 6.14.10."""
    print("=" * 60)
    print("Phase 6.14.10 - Retry Logic for Data Fetch Failures")
    print("=" * 60)
    
    try:
        test_fetch_with_retry()
        test_fetch_with_retry_exhausted()
        test_retry_with_exponential_backoff()
        test_get_retry_stats()
        test_clear_retry_stats()
        
        print("\n" + "=" * 60)
        print("ALL PHASE 6.14.10 TESTS PASSED!")
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
