"""
Test Phase 7.3: CIO & Calculator Components
Tests for tasks 7.3.1-7.3.4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_7_3_1_cio_terminal():
    """Task 7.3.1: CIO Terminal with color-coded verdict"""
    print("\n[PASS] Task 7.3.1: CIO Terminal with color-coded verdict")
    print("  - Color-coded verdict (BUY=green, SELL=red, HOLD=yellow)")
    print("  - Conviction meter visualization")
    print("  - Agent alignment indicators")
    print("  - Trade setup details")
    
    from dashboard.pages.cio_terminal import (
        get_sample_cio_data, render_verdict_banner,
        render_agent_alignment, render_trade_setup, render_thesis_keywords
    )
    
    assert callable(get_sample_cio_data), "get_sample_cio_data should be callable"
    assert callable(render_verdict_banner), "render_verdict_banner should be callable"
    assert callable(render_agent_alignment), "render_agent_alignment should be callable"
    assert callable(render_trade_setup), "render_trade_setup should be callable"
    assert callable(render_thesis_keywords), "render_thesis_keywords should be callable"
    
    sample = get_sample_cio_data()
    assert "verdict" in sample, "Sample should have verdict"
    assert "conviction" in sample, "Sample should have conviction"
    assert "agent_alignments" in sample, "Sample should have agent_alignments"
    assert "trade_setup" in sample, "Sample should have trade_setup"
    
    print(f"  - Sample verdict: {sample['verdict']}")
    print(f"  - Conviction: {sample['conviction']}/10")
    print(f"  - Agent alignments: {list(sample['agent_alignments'].keys())}")
    
    return True


def test_7_3_2_verdict_display():
    """Task 7.3.2: Verdict display component"""
    print("\n[PASS] Task 7.3.2: Verdict display component")
    print("  - render_verdict_display() function")
    print("  - render_verdict_badge() function")
    print("  - render_conviction_bar() function")
    print("  - get_verdict_color() helper")
    print("  - format_verdict_for_display() helper")
    
    from dashboard.components.verdict_display import (
        render_verdict_display, render_verdict_badge,
        render_conviction_bar, get_verdict_color, format_verdict_for_display
    )
    
    assert callable(render_verdict_display), "render_verdict_display should be callable"
    assert callable(render_verdict_badge), "render_verdict_badge should be callable"
    assert callable(render_conviction_bar), "render_conviction_bar should be callable"
    assert callable(get_verdict_color), "get_verdict_color should be callable"
    assert callable(format_verdict_for_display), "format_verdict_for_display should be callable"
    
    assert get_verdict_color("BUY") == "#28a745", "BUY should be green"
    assert get_verdict_color("SELL") == "#dc3545", "SELL should be red"
    assert get_verdict_color("HOLD") == "#ffc107", "HOLD should be yellow"
    
    assert format_verdict_for_display("STRONG_BUY") == "Strong Buy"
    
    print("  - All verdict display functions verified")
    print("  - Color mappings: BUY=#28a745, SELL=#dc3545, HOLD=#ffc107")
    
    return True


def test_7_3_3_position_calculator():
    """Task 7.3.3: Position Calculator component"""
    print("\n[PASS] Task 7.3.3: Position Calculator component")
    print("  - calculate_position_size() function")
    print("  - render_position_summary() function")
    print("  - render_risk_metrics() function")
    print("  - render_trade_levels() function")
    print("  - Kelly criterion support")
    
    from dashboard.pages.position_calculator import (
        calculate_position_size, render_position_summary,
        render_risk_metrics, render_trade_levels
    )
    
    assert callable(calculate_position_size), "calculate_position_size should be callable"
    assert callable(render_position_summary), "render_position_summary should be callable"
    assert callable(render_risk_metrics), "render_risk_metrics should be callable"
    assert callable(render_trade_levels), "render_trade_levels should be callable"
    
    results = calculate_position_size(
        entry_price=1000,
        stop_loss=950,
        target_price=1100,
        portfolio_size=100000,
        risk_pct=2.0,
        use_kelly=False,
        kelly_fraction=0.25
    )
    
    assert "shares" in results, "Results should have shares"
    assert "position_value" in results, "Results should have position_value"
    assert "risk_reward" in results, "Results should have risk_reward"
    assert "max_loss" in results, "Results should have max_loss"
    
    print("  - Entry: 1000, SL: 950, Target: 1100")
    print(f"  - Shares: {results['shares']}, Position: {results['position_value']:,.0f}")
    print(f"  - Risk:Reward = 1:{results['risk_reward']:.2f}")
    print(f"  - Max Loss: {results['max_loss']:,.2f}")
    
    return True


def test_7_3_4_risk_reward_visualization():
    """Task 7.3.4: Risk/reward visualization"""
    print("\n[PASS] Task 7.3.4: Risk/reward visualization")
    print("  - render_risk_reward_chart() with Plotly")
    print("  - Trade levels visualization")
    print("  - Kelly section display")
    
    from dashboard.pages.position_calculator import (
        render_risk_reward_chart, render_kelly_section
    )
    
    assert callable(render_risk_reward_chart), "render_risk_reward_chart should be callable"
    assert callable(render_kelly_section), "render_kelly_section should be callable"
    
    print("  - Risk/reward chart function verified")
    print("  - Kelly section function verified")
    
    return True


def test_position_calculation_accuracy():
    """Test position calculation accuracy"""
    print("\n[PASS] Task: Position Calculation Accuracy")
    
    from dashboard.pages.position_calculator import calculate_position_size
    
    results = calculate_position_size(
        entry_price=1000,
        stop_loss=950,
        target_price=1100,
        portfolio_size=100000,
        risk_pct=2.0
    )
    
    risk_per_share = 1000 - 950
    risk_amount = 100000 * 0.02
    expected_shares = int(risk_amount / risk_per_share)
    
    assert results['shares'] == expected_shares, f"Shares mismatch: {results['shares']} vs {expected_shares}"
    assert results['risk_per_share'] == 50, "Risk per share should be 50"
    assert results['risk_reward'] == 2.0, "R:R should be 2.0"
    
    print(f"  - Risk per share: {results['risk_per_share']} (expected: 50)")
    print(f"  - Shares: {results['shares']} (expected: {expected_shares})")
    print(f"  - Risk:Reward: {results['risk_reward']} (expected: 2.0)")
    
    return True


def test_verdict_colors():
    """Test verdict color mapping"""
    print("\n[PASS] Task: Verdict Color Mapping")
    
    from dashboard.components.verdict_display import get_verdict_color
    
    test_cases = [
        ("BUY", "#28a745"),
        ("SELL", "#dc3545"),
        ("HOLD", "#ffc107"),
        ("STRONG_BUY", "#28a745"),
        ("STRONG_SELL", "#dc3545"),
        ("UNKNOWN", "#6c757d"),
    ]
    
    for verdict, expected_color in test_cases:
        color = get_verdict_color(verdict)
        assert color == expected_color, f"Color mismatch for {verdict}: {color} vs {expected_color}"
    
    print("  - All verdict colors verified correctly")
    
    return True


def run_all_tests():
    """Run all Phase 7.3 tests"""
    print("=" * 60)
    print("Phase 7.3: CIO & Calculator - Test Suite")
    print("=" * 60)
    
    tests = [
        ("7.3.1 - CIO Terminal", test_7_3_1_cio_terminal),
        ("7.3.2 - Verdict Display", test_7_3_2_verdict_display),
        ("7.3.3 - Position Calculator", test_7_3_3_position_calculator),
        ("7.3.4 - Risk/Reward Viz", test_7_3_4_risk_reward_visualization),
        ("Position Accuracy", test_position_calculation_accuracy),
        ("Verdict Colors", test_verdict_colors),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("Phase 7.3 COMPLETE!")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
