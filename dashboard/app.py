"""
Phase 7.1: Dashboard Foundation
Main Streamlit Application with Sidebar and Session State Management

Tasks 7.1.1-7.1.5:
- Main app structure with pages
- Page navigation
- Sidebar with ticker input
- Risk parameter sliders
- Session state management
"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NIFTY50_PRESETS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "KOTAKBANK", "SBIN", "BHARTIARTL", "LT", "AXISBANK",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "TITAN", "NESTLEIND",
    "BAJFINANCE", "TATASTEEL"
]

DEFAULT_TICKER = "RELIANCE"

def init_session_state():
    """Initialize all session state variables (Task 7.1.5)"""
    if "ticker" not in st.session_state:
        st.session_state.ticker = DEFAULT_TICKER
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
    
    if "selected_strategy" not in st.session_state:
        st.session_state.selected_strategy = "conservative"
    
    if "risk_params" not in st.session_state:
        st.session_state.risk_params = {
            "stop_loss_atr": 2.0,
            "target_pct": 3.0,
            "max_position_pct": 20.0,
            "kelly_fraction": 0.25,
            "var_confidence": 0.95,
        }
    
    if "conviction_threshold" not in st.session_state:
        st.session_state.conviction_threshold = 6
    
    if "analysis_date" not in st.session_state:
        st.session_state.analysis_date = datetime.now().strftime("%Y-%m-%d")
    
    if "cache_enabled" not in st.session_state:
        st.session_state.cache_enabled = True
    
    if "ollama_connected" not in st.session_state:
        st.session_state.ollama_connected = None


def render_sidebar():
    """Render sidebar with ticker input and risk parameters (Tasks 7.1.3, 7.1.4)"""
    with st.sidebar:
        st.title("📊 Trading Dashboard")
        st.divider()
        
        st.subheader("Stock Selection")
        ticker_input = st.text_input(
            "Ticker Symbol",
            value=st.session_state.ticker,
            help="NSE stock symbol (e.g., RELIANCE, TCS)",
            placeholder="Enter symbol..."
        )
        ticker = ticker_input.upper().strip() if ticker_input else ""
        
        if ticker != st.session_state.ticker:
            st.session_state.ticker = ticker
        
        st.selectbox(
            "Quick Select (NIFTY 50)",
            options=[""] + NIFTY50_PRESETS,
            index=0,
            key="quick_select",
            on_change=lambda: _handle_quick_select(),
            help="Quickly select from NIFTY 50 stocks"
        )
        
        st.divider()
        
        st.subheader("⚙️ Risk Parameters")
        
        stop_loss_atr = st.slider(
            "Stop Loss (ATR multiples)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.risk_params["stop_loss_atr"],
            step=0.25,
            help="Stop loss distance in ATR multiples"
        )
        
        target_pct = st.slider(
            "Target Profit (%)",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.risk_params["target_pct"],
            step=0.5,
            help="Profit target as percentage of entry"
        )
        
        max_position_pct = st.slider(
            "Max Position Size (%)",
            min_value=5.0,
            max_value=50.0,
            value=st.session_state.risk_params["max_position_pct"],
            step=5.0,
            help="Maximum portfolio allocation per position"
        )
        
        st.divider()
        
        st.subheader("📈 Portfolio Settings")
        
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.1,
            max_value=0.5,
            value=st.session_state.risk_params["kelly_fraction"],
            step=0.05,
            help="Position sizing Kelly fraction (lower = safer)"
        )
        
        var_confidence = st.slider(
            "VaR Confidence Level",
            min_value=0.90,
            max_value=0.99,
            value=st.session_state.risk_params["var_confidence"],
            step=0.01,
            help="Value at Risk confidence interval"
        )
        
        conviction_threshold = st.slider(
            "Conviction Threshold",
            min_value=1,
            max_value=10,
            value=st.session_state.conviction_threshold,
            step=1,
            help="Minimum conviction score to trigger signal"
        )
        
        if st.button("💾 Save Risk Settings", use_container_width=True):
            st.session_state.risk_params = {
                "stop_loss_atr": stop_loss_atr,
                "target_pct": target_pct,
                "max_position_pct": max_position_pct,
                "kelly_fraction": kelly_fraction,
                "var_confidence": var_confidence,
            }
            st.session_state.conviction_threshold = conviction_threshold
            st.success("Settings saved!")
        
        st.divider()
        
        st.subheader("🔧 System Settings")
        
        cache_toggle = st.checkbox(
            "Enable Caching",
            value=st.session_state.cache_enabled,
            help="Cache market data for faster loading"
        )
        st.session_state.cache_enabled = cache_toggle
        
        analysis_date = st.date_input(
            "Analysis Date",
            value=datetime.now(),
            max_value=datetime.now(),
            help="Date for historical analysis"
        )
        st.session_state.analysis_date = analysis_date.strftime("%Y-%m-%d")
        
        st.divider()
        
        with st.expander("📋 Current Settings Summary"):
            st.json({
                "Ticker": st.session_state.ticker,
                "Risk Parameters": st.session_state.risk_params,
                "Conviction Threshold": st.session_state.conviction_threshold,
                "Cache Enabled": st.session_state.cache_enabled,
                "Analysis Date": st.session_state.analysis_date,
            })
        
        st.divider()
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def _handle_quick_select():
    """Handle quick select dropdown selection"""
    selected = st.session_state.get("quick_select", "")
    if selected and selected != st.session_state.ticker:
        st.session_state.ticker = selected
        st.session_state.quick_select = ""


def render_header():
    """Render main header"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title(f"📈 {st.session_state.ticker}")
        st.caption(f"Analysis Date: {st.session_state.analysis_date}")
    
    with col2:
        st.metric(
            label="System Status",
            value="Ready" if st.session_state.ollama_connected else "Checking...",
            delta="Connected" if st.session_state.ollama_connected else "..."
        )
    
    with col3:
        st.metric(
            label="Risk Profile",
            value=f"{st.session_state.risk_params['max_position_pct']:.0f}%",
            delta=f"SL: {st.session_state.risk_params['stop_loss_atr']}x ATR"
        )


def render_navigation():
    """Render navigation tabs (Task 7.1.2)"""
    tab_names = [
        "🎯 Today's Picks",
        "📊 Overview",
        "📰 Market Context",
        "🤖 Agent Reports",
        "💼 CIO Terminal",
        "🔬 Backtest Hub"
    ]
    
    return st.tabs(tab_names)


def check_ollama_connection():
    """Check if Ollama is connected"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        st.session_state.ollama_connected = response.status_code == 200
    except Exception:
        st.session_state.ollama_connected = False
    return st.session_state.ollama_connected


def main():
    """Main application entry point (Task 7.1.1)"""
    init_session_state()
    render_sidebar()
    
    st.title("AI Trading System Dashboard")
    st.markdown("---")
    
    is_connected = check_ollama_connection()
    
    if not is_connected:
        st.warning(
            "⚠️ Ollama not connected. LLM features will be disabled. "
            "Start Ollama with: `ollama serve`"
        )
    
    render_header()
    
    tabs = render_navigation()
    
    with tabs[0]:
        st.info("🎯 Go to Today's Picks page")
        st.page_link("pages/todays_picks.py", label="Go to Today's Picks", icon="🎯")
        st.markdown("---")
        st.markdown("Or use the Quick Analysis below:")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            quick_ticker = st.text_input(
                "Quick Analyze Stock",
                value=st.session_state.ticker,
                placeholder="Enter symbol...",
            )
        with col2:
            st.markdown("")
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                if quick_ticker:
                    st.session_state.ticker = quick_ticker.upper().strip()
                    result = run_analysis(st.session_state.ticker)
                    if result:
                        st.success(f"Analysis complete!")
                        st.session_state.analysis_results = result
                        st.rerun()
        
        if st.session_state.analysis_results:
            st.divider()
            st.subheader("Analysis Result")
            result = st.session_state.analysis_results
            verdict = result.get("verdict", "N/A")
            conviction = result.get("conviction", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Verdict", verdict)
            with col2:
                st.metric("Conviction", f"{conviction}/10")
            
            with st.expander("View Full Analysis"):
                st.page_link("pages/cio_terminal.py", label="Go to CIO Terminal", icon="💼")
    
    with tabs[1]:
        st.info("📰 Market Context page - See 'pages/market_context.py'")
        st.page_link("pages/market_context.py", label="Go to Market Context", icon="📰")
    
    with tabs[2]:
        st.info("🤖 Agent Reports page - See 'pages/agent_reports.py'")
        st.page_link("pages/agent_reports.py", label="Go to Agent Reports", icon="🤖")
    
    with tabs[3]:
        st.info("💼 CIO Terminal page - See 'pages/cio_terminal.py'")
        st.page_link("pages/cio_terminal.py", label="Go to CIO Terminal", icon="💼")
    
    with tabs[4]:
        st.info("🔬 Backtest Hub page - See 'pages/backtest_hub.py'")
        st.page_link("pages/backtest_hub.py", label="Go to Backtest Hub", icon="🔬")


def run_analysis(ticker: str):
    """Run the full analysis pipeline via orchestrator (Task 7.2.4 - Context sharing)"""
    try:
        with st.spinner(f"Analyzing {ticker}... This may take 1-2 minutes."):
            from src.pipeline.orchestrator import PipelineOrchestrator
            
            orchestrator = PipelineOrchestrator()
            result = orchestrator.analyze_single_stock(ticker)
            
            if result and "agent_outputs" in result:
                agent_outputs = result.get("agent_outputs", {})
                
                cio_output = agent_outputs.get("cio", {})
                verdict = cio_output.get("verdict", "HOLD")
                conviction = cio_output.get("conviction", 5)
                
                result["verdict"] = verdict
                result["conviction"] = conviction
                result["signal"] = f"{verdict}_{conviction}"
                result["analysis"] = cio_output.get("analysis", "")
                
                st.session_state.analysis_results = result
                return result
            else:
                st.error(f"Analysis failed: {result}")
                return None
    except Exception as e:
        import traceback
        st.error(f"Analysis error: {str(e)}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return None


def render_overview_tab():
    """Render the overview/home tab"""
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Ticker",
            value=st.session_state.ticker,
        )
    
    with col2:
        analysis_done = st.session_state.analysis_results is not None
        status = "Done" if analysis_done else "Pending"
        st.metric(
            label="Analysis Status",
            value=status,
        )
    
    with col3:
        strategy = st.session_state.selected_strategy.title()
        st.metric(
            label="Strategy",
            value=strategy,
        )
    
    with col4:
        conviction = st.session_state.conviction_threshold
        st.metric(
            label="Conviction Threshold",
            value=f"{conviction}/10",
        )
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Risk Parameters")
        st.json(st.session_state.risk_params)
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("🔄 Run Full Analysis", use_container_width=True, type="primary"):
            result = run_analysis(st.session_state.ticker)
            if result:
                st.success(f"Analysis complete for {st.session_state.ticker}!")
                st.rerun()
        
        if st.button("📊 Run Quick Backtest", use_container_width=True):
            st.info("Backtest triggered! See Backtest Hub page.")
        
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    if st.session_state.analysis_results:
        st.divider()
        st.subheader("Latest Analysis Result")
        
        result = st.session_state.analysis_results
        verdict = result.get("verdict", "N/A")
        conviction = result.get("conviction", 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Verdict", verdict)
        with col2:
            st.metric("Conviction", f"{conviction}/10")
        
        with st.expander("View Agent Reports"):
            st.page_link("pages/agent_reports.py", label="Go to Agent Reports", icon="🤖")


if __name__ == "__main__":
    main()
