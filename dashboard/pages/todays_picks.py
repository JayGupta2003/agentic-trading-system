"""
Today's Picks - Daily Trading Recommendations

Automatic screening and analysis of NIFTY stocks.
Shows BUY/HOLD/SELL recommendations with conviction scores.
"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

st.set_page_config(page_title="Today's Picks", page_icon="🎯", layout="wide")

st.title("🎯 Today's Picks")
st.markdown("**Automatic screening & AI analysis of NIFTY stocks**")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


def run_screening(universe: str = "NIFTY50", top_n: int = 50, 
                  min_fscore: int = 3, max_pe: float = 50, 
                  min_momentum: float = -20) -> Dict[str, Any]:
    """Run stock screening."""
    try:
        from src.screening.screener import StockShortlister
        
        screener = StockShortlister()
        
        with st.spinner("Screening stocks..."):
            result = screener.run_screening(
                universe=universe,
                min_fscore=min_fscore,
                max_pe=max_pe,
                min_momentum=min_momentum,
                top_n=top_n,
                save_output=True,
            )
        
        return result
    except Exception as e:
        return {"status": "error", "error": str(e), "stocks": []}


def run_full_analysis(symbols: List[str], progress_bar=None) -> List[Dict]:
    """Run full agent analysis on a list of symbols and store results."""
    results = []
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"Analyzing {symbol}...")
        
        try:
            from src.pipeline.orchestrator import PipelineOrchestrator
            from src.trading.signal_store import SignalStore
            from src.trading.types import TradingSignal
            
            orchestrator = PipelineOrchestrator()
            result = orchestrator.analyze_single_stock(symbol)
            
            agent_outputs = result.get("agent_outputs", {})
            cio_output = agent_outputs.get("cio", {})
            
            verdict = cio_output.get("verdict", "HOLD")
            conviction = cio_output.get("conviction_score", 5)
            thesis = cio_output.get("thesis", "")
            risk_params = cio_output.get("risk_params", {})
            
            entry_strategy = cio_output.get("entry_strategy", "")
            
            entry_price = risk_params.get("target", 0) or 0
            stop_loss = risk_params.get("stop_loss", 0) or 0
            
            signal = TradingSignal(
                symbol=symbol,
                signal=verdict,
                conviction=conviction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=risk_params.get("target", 0) or 0,
                thesis=thesis,
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=30),
                review_type="ai_analysis",
                cap_type="large",
                agent_outputs=agent_outputs,
                full_cio_analysis=cio_output.get("analysis", ""),
            )
            
            store = SignalStore()
            store.save(signal)
            
            results.append({
                "symbol": symbol,
                "verdict": verdict,
                "conviction": conviction,
                "thesis": thesis,
                "status": "success",
                "full_analysis": cio_output.get("analysis", ""),
                "entry_strategy": entry_strategy,
                "risk_params": risk_params,
                "all_agent_outputs": agent_outputs,
            })
            
        except Exception as e:
            results.append({
                "symbol": symbol,
                "verdict": "ERROR",
                "conviction": 0,
                "thesis": str(e),
                "status": "error",
            })
    
    return results


def render_results(results: List[Dict]):
    """Render analysis results."""
    
    if not results:
        st.info("No results to display.")
        return
    
    df_data = []
    for r in results:
        df_data.append({
            "Stock": r["symbol"],
            "Verdict": r["verdict"],
            "Conviction": f"{r['conviction']}/10",
            "Status": "✅" if r["status"] == "success" else "❌",
            "Summary": r.get("thesis", "")[:100] + "..." if r.get("thesis") else "",
        })
    
    df = pd.DataFrame(df_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buy_count = len([r for r in results if "BUY" in str(r["verdict"]).upper()])
        st.metric("BUY Signals", buy_count)
    
    with col2:
        hold_count = len([r for r in results if r["verdict"].upper() == "HOLD"])
        st.metric("HOLD Signals", hold_count)
    
    with col3:
        sell_count = len([r for r in results if "SELL" in str(r["verdict"]).upper()])
        st.metric("SELL Signals", sell_count)
    
    with col4:
        success_count = len([r for r in results if r["status"] == "success"])
        st.metric("Analyzed", f"{success_count}/{len(results)}")
    
    st.divider()
    
    st.subheader("📋 Results")
    
    col_filter = st.columns(3)
    with col_filter[0]:
        filter_verdict = st.multiselect(
            "Filter by Verdict",
            options=["BUY", "HOLD", "SELL", "STRONG_BUY", "STRONG_SELL"],
            default=["BUY", "STRONG_BUY"],
        )
    with col_filter[1]:
        min_conviction = st.slider("Min Conviction", 1, 10, 6)
    with col_filter[2]:
        sort_by = st.selectbox("Sort By", ["Conviction", "Symbol"])
    
    filtered = [r for r in results 
               if r["verdict"] in filter_verdict 
               and r["conviction"] >= min_conviction]
    
    if sort_by == "Conviction":
        filtered = sorted(filtered, key=lambda x: x["conviction"], reverse=True)
    else:
        filtered = sorted(filtered, key=lambda x: x["symbol"])
    
    if not filtered:
        st.warning("No stocks match your filters.")
        return
    
    st.dataframe(
        pd.DataFrame([
            {
                "Stock": r["symbol"],
                "Verdict": r["verdict"],
                "Conviction": r["conviction"],
                "Why": r.get("thesis", "")[:80] + "..." if r.get("thesis") else "N/A",
            }
            for r in filtered
        ]),
        use_container_width=True,
        hide_index=True,
    )
    
    st.divider()
    st.subheader("📊 Top BUY Picks (Detailed)")
    
    buy_results = [r for r in filtered if "BUY" in str(r["verdict"]).upper()]
    
    for r in buy_results[:10]:
        with st.expander(f"**{r['symbol']}** - {r['verdict']} (Conviction: {r['conviction']}/10)"):
            st.markdown(f"**Summary:** {r.get('thesis', 'No thesis available')}")
            if r.get("full_analysis"):
                st.markdown("---")
                st.markdown("**Full Analysis:**")
                st.text(r["full_analysis"][:2000] + "..." if len(r.get("full_analysis", "")) > 2000 else r.get("full_analysis", ""))


def main():
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        col_left, col_mid, col_right = st.columns(3)
        
        with col_left:
            universe = st.selectbox(
                "Stock Universe",
                options=["NIFTY50", "NIFTY100", "NIFTY500"],
                index=0,
            )
        
        with col_mid:
            min_fscore = st.slider("Min F-Score", 0, 9, 3)
        
        with col_right:
            max_pe = st.slider("Max P/E", 10, 100, 50)
        
        col_left2, col_mid2 = st.columns(2)
        
        with col_left2:
            min_momentum = st.slider("Min 3M Momentum (%)", -50, 0, -20)
        
        with col_mid2:
            max_stocks = st.selectbox(
                "Max Stocks to Analyze",
                options=[10, 20, 30, 50],
                index=3,
            )
        
        run_analysis = st.button(
            "🔍 Run Full Analysis (Screen + AI Analysis)",
            type="primary",
            use_container_width=True,
        )
    
    with col2:
        st.subheader("ℹ️ How It Works")
        st.markdown("""
        1. **Screen** stocks using F-Score, P/E, momentum
        2. **Analyze** with 6 AI agents
        3. **CIO** makes final decision
        4. **Rank** by conviction score
        """)
    
    st.divider()
    
    if run_analysis:
        result = run_screening(
            universe=universe, 
            top_n=max_stocks,
            min_fscore=min_fscore,
            max_pe=max_pe,
            min_momentum=min_momentum,
        )
        
        if result.get("status") == "error":
            st.error(f"Screening failed: {result.get('error')}")
            return
        
        screened_stocks = result.get("stocks", [])
        
        if not screened_stocks:
            st.warning("⚠️ No stocks passed screening criteria.")
            st.info("💡 Try relaxing the criteria: Lower F-Score, increase Max P/E, or increase Min Momentum.")
            st.markdown("""
            **Default criteria are strict:**
            - F-Score ≥ 3 (Quality)
            - P/E ≤ 50 (Valuation)
            - 3M Return ≥ -20% (Momentum)
            
            Try: F-Score=2, Max P/E=100, Min Momentum=-50
            """)
            return
        
        st.success(f"Found {len(screened_stocks)} stocks passing screening")
        
        symbols = [s["symbol"] for s in screened_stocks]
        
        st.info(f"Analyzing {len(symbols)} stocks with AI agents... This may take 5-10 minutes.")
        
        progress_bar = st.progress(0, text="Starting analysis...")
        
        results = run_full_analysis(symbols, progress_bar)
        
        progress_bar.empty()
        
        render_results(results)
    
    else:
        st.info("👆 Click 'Run Full Analysis' to get today's trading recommendations")
        
        st.divider()
        st.subheader("📜 Recent Results")
        
        try:
            from src.trading.signal_store import SignalStore
            
            store = SignalStore()
            signals = store.get_active()
            
            if signals:
                df_data = []
                for s in signals:
                    df_data.append({
                        "Stock": s.symbol,
                        "Signal": s.signal,
                        "Conviction": s.conviction,
                        "Valid Until": s.valid_until.strftime("%Y-%m-%d"),
                    })
                
                st.dataframe(
                    pd.DataFrame(df_data),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No saved signals. Run analysis to generate signals.")
        except Exception as e:
            st.error(f"Could not load signals: {e}")


if __name__ == "__main__":
    main()
