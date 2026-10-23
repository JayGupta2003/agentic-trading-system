"""
Test file for Phase 6.14.9 - Log Errors/Warnings
"""
import sys
sys.path.insert(0, r"J:\KAAM\Python\Quant\Agentic Trading System Part 2")

import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile


def test_log_error():
    """Test logging errors."""
    print("\n=== Test 6.14.9.1: log_error ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    result = interface.log_error(
        error_type="DataNotFound",
        message="Data not found for symbol RELIANCE.NS",
        context={"symbol": "RELIANCE.NS", "date": "2024-01-01"}
    )
    
    print(f"Log error - logged: {result['logged']}")
    print(f"Log error - error_id: {result['error_id']}")
    assert result['logged'] == True, "Error should be logged"
    assert 'error_id' in result, "Should have error_id"
    
    print("[PASS] log_error: PASSED")
    return result


def test_log_warning():
    """Test logging warnings."""
    print("\n=== Test 6.14.9.2: log_warning ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    result = interface.log_warning(
        warning_type="InsufficientData",
        message="Only 50 days of data available, recommended minimum is 200",
        context={"symbol": "TEST.NS", "available_days": 50}
    )
    
    print(f"Log warning - logged: {result['logged']}")
    print(f"Log warning - warning_id: {result['warning_id']}")
    assert result['logged'] == True, "Warning should be logged"
    assert 'warning_id' in result, "Should have warning_id"
    
    print("[PASS] log_warning: PASSED")
    return result


def test_get_error_log():
    """Test retrieving error log."""
    print("\n=== Test 6.14.9.3: get_error_log ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Log a few errors
    interface.log_error("TestError1", "Test message 1")
    interface.log_error("TestError2", "Test message 2")
    
    errors = interface.get_error_log()
    
    print(f"Error log - count: {errors['count']}")
    print(f"Error log - recent: {len(errors['recent'])} entries")
    assert errors['count'] >= 2, "Should have at least 2 errors"
    assert len(errors['recent']) > 0, "Should have recent entries"
    
    print("[PASS] get_error_log: PASSED")
    return errors


def test_get_warning_log():
    """Test retrieving warning log."""
    print("\n=== Test 6.14.9.4: get_warning_log ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Log a few warnings
    interface.log_warning("TestWarning1", "Test message 1")
    interface.log_warning("TestWarning2", "Test message 2")
    
    warnings = interface.get_warning_log()
    
    print(f"Warning log - count: {warnings['count']}")
    print(f"Warning log - recent: {len(warnings['recent'])} entries")
    assert warnings['count'] >= 2, "Should have at least 2 warnings"
    assert len(warnings['recent']) > 0, "Should have recent entries"
    
    print("[PASS] get_warning_log: PASSED")
    return warnings


def test_export_logs():
    """Test exporting logs to file."""
    print("\n=== Test 6.14.9.5: export_logs ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Log some entries
    interface.log_error("ExportTest", "Test error for export")
    interface.log_warning("ExportTest", "Test warning for export")
    
    # Export to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        result = interface.export_logs(output_path=temp_path, format="json")
        
        print(f"Export - success: {result['success']}")
        print(f"Export - file: {result['file_path']}")
        assert result['success'] == True, "Export should succeed"
        assert os.path.exists(temp_path), "File should exist"
        
        # Verify content
        import json
        with open(temp_path, 'r') as f:
            data = json.load(f)
        print(f"Export - errors: {data['errors']['count']}")
        print(f"Export - warnings: {data['warnings']['count']}")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("[PASS] export_logs: PASSED")
    return result


def test_clear_logs():
    """Test clearing logs."""
    print("\n=== Test 6.14.9.6: clear_logs ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Log some entries
    interface.log_error("ClearTest", "Test error for clear")
    interface.log_warning("ClearWarnTest", "Test warning for clear")
    
    # Clear errors only
    result = interface.clear_logs(log_type="errors")
    print(f"Clear errors - success: {result['success']}")
    print(f"Clear errors - cleared: {result['cleared_count']}")
    
    # Verify errors cleared
    errors = interface.get_error_log()
    recent_errors = [e for e in errors['recent'] if e['type'] == 'ClearTest']
    assert len(recent_errors) == 0, "ClearTest errors should be cleared"
    
    # Warnings should still exist
    warnings = interface.get_warning_log()
    recent_warnings = [w for w in warnings['recent'] if w['type'] == 'ClearWarnTest']
    assert len(recent_warnings) > 0, "ClearWarnTest warnings should still exist"
    
    print("[PASS] clear_logs: PASSED")
    return result


def test_get_log_summary():
    """Test getting log summary."""
    print("\n=== Test 6.14.9.7: get_log_summary ===")
    
    from src.backtesting.interface import BacktesterInterface
    
    interface = BacktesterInterface()
    
    # Log various types
    interface.log_error("Type1", "Error 1")
    interface.log_error("Type2", "Error 2")
    interface.log_warning("Type1", "Warning 1")
    interface.log_warning("Type3", "Warning 3")
    
    summary = interface.get_log_summary()
    
    print(f"Summary - total errors: {summary['total_errors']}")
    print(f"Summary - total warnings: {summary['total_warnings']}")
    print(f"Summary - error types: {summary['error_types']}")
    print(f"Summary - warning types: {summary['warning_types']}")
    
    assert summary['total_errors'] >= 2, "Should have at least 2 errors"
    assert summary['total_warnings'] >= 2, "Should have at least 2 warnings"
    assert 'Type1' in summary['error_types'], "Type1 should be in error types"
    
    print("[PASS] get_log_summary: PASSED")
    return summary


def run_all_tests():
    """Run all tests for phase 6.14.9."""
    print("=" * 60)
    print("Phase 6.14.9 - Log Errors/Warnings")
    print("=" * 60)
    
    try:
        test_log_error()
        test_log_warning()
        test_get_error_log()
        test_get_warning_log()
        test_export_logs()
        test_clear_logs()
        test_get_log_summary()
        
        print("\n" + "=" * 60)
        print("ALL PHASE 6.14.9 TESTS PASSED!")
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
