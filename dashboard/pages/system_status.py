"""
Phase 6: Trading System Status Dashboard

Displays:
- Current signal status for all holdings
- Trigger history with timestamps
- Review schedule (next scheduled review)
- Emergency review history
- Portfolio status and drawdown
"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

st.set_page_config(page_title="Trading System Status", page_icon="📊", layout="wide")

st.title("📊 Trading System Status")
st.markdown("Real-time monitoring of signal status, triggers, and review schedule")


def get_system_status() -> Dict[str, Any]:
    """Get current system status from event loop."""
    try:
        from src.trading.event_loop import TradingEventLoop, EventLoopConfig
        
        config = EventLoopConfig(
            storage_path="data/signals",
            trigger_log_path="data/trigger_log.json",
            execution_log_path="data/execution_log.json"
        )
        
        loop = TradingEventLoop(config)
        
        status = {
            "active_signals": loop.signal_store.get_signal_count(),
            "recent_triggers": loop.get_trigger_history(days=30),
            "trigger_summary": loop.get_trigger_summary(),
            "execution_summary": loop.get_execution_summary(),
            "review_schedule": loop.get_review_schedule(),
            "next_review": loop.get_next_review(),
            "loop": loop,
        }
        
        return status
    except Exception as e:
        return {
            "error": str(e),
            "active_signals": {},
            "recent_triggers": [],
            "trigger_summary": {},
            "execution_summary": {},
            "review_schedule": {},
            "next_review": ("N/A", datetime.now()),
            "loop": None,
        }


def run_event_loop_now():
    """Run the event loop and return updated status."""
    try:
        from src.trading.event_loop import TradingEventLoop, EventLoopConfig
        
        config = EventLoopConfig(
            storage_path="data/signals",
            trigger_log_path="data/trigger_log.json",
            execution_log_path="data/execution_log.json"
        )
        
        loop = TradingEventLoop(config)
        status_obj = loop.run_daily()
        
        return {
            "success": True,
            "message": f"Event loop completed! Triggers: {len(status_obj.recent_triggers)}, Executions: {len(status_obj.recent_executions)}, Portfolio: ₹{status_obj.portfolio_value:,.0f}"
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "traceback": traceback.format_exc()
        }


def render_signal_status(signals: List[Any]):
    """Render current signal status for all holdings."""
    st.subheader("📋 Current Signal Status")
    
    if not signals:
        st.info("No active signals. Run analysis to generate signals.")
        return
    
    data = []
    for signal in signals:
        status = "Active" if signal.position_open else "Signal Only"
        days_until_expiry = (signal.valid_until - datetime.now()).days
        
        data.append({
            "Symbol": signal.symbol,
            "Signal": signal.signal,
            "Conviction": f"{signal.conviction}/10",
            "Status": status,
            "Cap Type": signal.cap_type.title(),
            "Entry Price": f"₹{signal.entry_price:.2f}" if signal.entry_price > 0 else "N/A",
            "Stop Loss": f"₹{signal.stop_loss:.2f}" if signal.stop_loss > 0 else "N/A",
            "Target": f"₹{signal.target_price:.2f}" if signal.target_price > 0 else "N/A",
            "Valid Until": signal.valid_until.strftime("%Y-%m-%d"),
            "Days Left": days_until_expiry,
            "Review Type": signal.review_type.title(),
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_trigger_history(triggers: List[Any]):
    """Render trigger history table."""
    st.subheader("📈 Trigger History (Last 30 Days)")
    
    if not triggers:
        st.info("No triggers recorded in the last 30 days.")
        return
    
    data = []
    for trigger in reversed(triggers[-20:]):
        data.append({
            "Date": trigger.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Type": trigger.trigger_type.replace("_", " ").title(),
            "Symbols": ", ".join(trigger.symbols_affected) if trigger.symbols_affected else "N/A",
            "Agents Run": len(trigger.agents_run),
            "Decision": trigger.decision.replace("_", " ").title(),
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_trigger_summary(summary: Dict[str, int]):
    """Render trigger summary metrics."""
    st.subheader("📊 Trigger Summary")
    
    if not summary:
        st.info("No triggers recorded.")
        return
    
    cols = st.columns(len(summary))
    
    for i, (trigger_type, count) in enumerate(sorted(summary.items(), key=lambda x: -x[1])):
        with cols[i]:
            trigger_display = trigger_type.replace("_", " ").title()
            st.metric(label=trigger_display, value=count)


def render_review_schedule(schedule: Dict[str, datetime]):
    """Render upcoming review schedule."""
    st.subheader("📅 Review Schedule")
    
    if not schedule:
        st.info("No upcoming reviews scheduled.")
        return
    
    sorted_schedule = sorted(schedule.items(), key=lambda x: x[1])
    
    upcoming = [(s, d) for s, d in sorted_schedule if d >= datetime.now()]
    past = [(s, d) for s, d in sorted_schedule if d < datetime.now()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upcoming Reviews**")
        if upcoming:
            for symbol, date in upcoming[:5]:
                days = (date - datetime.now()).days
                st.markdown(f"- **{symbol}**: {date.strftime('%Y-%m-%d')} ({days} days)")
        else:
            st.info("No upcoming reviews")
    
    with col2:
        st.markdown("**Past Reviews**")
        if past:
            for symbol, date in reversed(past[-5:]):
                days = (datetime.now() - date).days
                st.markdown(f"- **{symbol}**: {date.strftime('%Y-%m-%d')} ({days} days ago)")
        else:
            st.info("No past reviews")


def render_next_review(next_review: tuple):
    """Render next review card."""
    symbol, date = next_review
    
    if symbol == "NONE":
        st.warning("No reviews scheduled")
        return
    
    days = (date - datetime.now()).days
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3 style="margin: 0; color: white;">Next Scheduled Review</h3>
        <h2 style="margin: 10px 0; color: white;">{symbol}</h2>
        <p style="margin: 0; font-size: 14px; opacity: 0.9;">
            {date.strftime('%B %d, %Y')} ({days} days)
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_execution_summary(summary: Dict[str, Any]):
    """Render execution summary."""
    st.subheader("📜 Execution Summary")
    
    if not summary or summary.get("total_executions", 0) == 0:
        st.info("No executions recorded.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Executions", summary.get("total_executions", 0))
    
    with col2:
        success = summary.get("successful", 0)
        st.metric("Successful", success)
    
    with col3:
        failed = summary.get("failed", 0)
        st.metric("Failed", failed)
    
    with col4:
        pnl = summary.get("total_pnl", 0)
        delta_color = "normal" if pnl >= 0 else "inverse"
        st.metric("Total P&L", f"₹{pnl:,.2f}", delta_color=delta_color)
    
    by_action = summary.get("by_action", {})
    col1, col2 = st.columns(2)
    
    with col1:
        entries = by_action.get("ENTRY", 0)
        st.metric("Total Entries", entries)
    
    with col2:
        exits = by_action.get("EXIT", 0)
        st.metric("Total Exits", exits)


def render_portfolio_status():
    """Render current portfolio status."""
    st.subheader("💼 Portfolio Status")
    
    try:
        from src.trading.brokers.historical import HistoricalBroker
        
        broker = HistoricalBroker()
        
        positions = broker.get_positions()
        cash = broker.get_account_balance()
        positions_value = sum(p.market_value for p in positions)
        total_value = cash + positions_value
        
        starting_capital = broker.starting_capital
        drawdown = (starting_capital - total_value) / starting_capital * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"₹{total_value:,.0f}")
        
        with col2:
            st.metric("Cash", f"₹{cash:,.0f}")
        
        with col3:
            st.metric("Positions", len(positions))
        
        with col4:
            delta_color = "normal" if drawdown <= 0 else "inverse"
            st.metric("Drawdown", f"{drawdown:.2f}%", delta_color=delta_color)
        
        if positions:
            st.markdown("**Open Positions:**")
            pos_data = []
            for p in positions:
                pnl = p.unrealized_pnl
                pnl_color = "green" if pnl >= 0 else "red"
                pos_data.append({
                    "Symbol": p.symbol,
                    "Quantity": p.quantity,
                    "Entry": f"₹{p.entry_price:.2f}",
                    "Current": f"₹{p.current_price:.2f}",
                    "P&L": f"₹{pnl:,.0f}",
                    "P&L %": f"{pnl / (p.entry_price * p.quantity) * 100:.1f}%",
                })
            
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not load portfolio status: {e}")


def render_emergency_review_guide():
    """Render information about emergency reviews."""
    with st.expander("ℹ️ Emergency Review Triggers"):
        st.markdown("""
        **Emergency reviews are triggered automatically when:**
        
        1. **Price Drop** - Stock drops > threshold (Large Cap: 5%, Mid Cap: 8%, Small Cap: 12%)
        2. **VIX Spike** - India VIX > μ₂₀ + 2σ₂₀ (Z-score spike detection)
        3. **Earnings** - Any holding reports quarterly results
        4. **Max Drawdown** - Portfolio drawdown exceeds 10% (Circuit Breaker)
        
        **Emergency Review Agents:**
        - Minimum: Sentiment + Technical + CIO
        - Risk and CIO agents always included
        - Skip: Fundamental (unchanged), Derivatives (unreliable during volatility)
        """)


def main():
    st.divider()
    
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Run Daily Event Loop", use_container_width=True, type="primary"):
            st.info("Running event loop...")
            result = run_event_loop_now()
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
                if "traceback" in result:
                    st.code(result["traceback"])
    
    with col2:
        if st.button("📊 Refresh Status", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    st.subheader("ℹ️ How It Works")
    st.markdown("""
    The **Event Loop** runs daily after market close:
    1. Evaluates all triggers (scheduled, VIX, price drop, etc.)
    2. Updates trailing stops for open positions
    3. Executes entries/exits based on signals
    4. Logs all decisions for audit trail
    
    Initially, no data exists. Run the event loop to start tracking!
    """)
    
    st.divider()
    
    status = get_system_status()
    
    if "error" in status:
        st.error(f"Error loading system status: {status['error']}")
        st.code(status["error"])
        return
    
    render_emergency_review_guide()
    
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⏰ Next Review")
        render_next_review(status.get("next_review", ("N/A", datetime.now())))
    
    with col2:
        render_execution_summary(status.get("execution_summary", {}))
    
    st.divider()
    
    tabs = st.tabs(["Signals", "Triggers", "Schedule", "Portfolio"])
    
    with tabs[0]:
        try:
            from src.trading.signal_store import SignalStore
            store = SignalStore()
            signals = store.get_all()
            render_signal_status(signals)
        except Exception as e:
            st.error(f"Could not load signals: {e}")
    
    with tabs[1]:
        render_trigger_summary(status.get("trigger_summary", {}))
        st.divider()
        render_trigger_history(status.get("recent_triggers", []))
    
    with tabs[2]:
        render_review_schedule(status.get("review_schedule", {}))
    
    with tabs[3]:
        render_portfolio_status()
    
    st.divider()
    
    with st.expander("🔧 System Configuration"):
        st.json({
            "scheduled_reviews": "Monthly (last trading day)",
            "price_drop_thresholds": {
                "large_cap": "5%",
                "mid_cap": "8%",
                "small_cap": "12%"
            },
            "vix_trigger": "Z-score (μ + 2σ)",
            "circuit_breaker": "10% drawdown",
            "f_o_expiry": "Skip derivatives agent"
        })


if __name__ == "__main__":
    main()
