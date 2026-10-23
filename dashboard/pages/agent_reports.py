"""
Phase 7.2.2/7.2.3: Agent Reports Page

Displays individual agent analysis reports with:
- Tabbed interface for each agent
- Expandable sections with detailed metrics
- Context sharing between agents
- Full agent output display
"""

import streamlit as st
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Agent Reports", page_icon="🤖", layout="wide")

st.title("Agent Reports")
st.markdown("Individual analysis from each AI agent")


def get_sample_agent_data():
    """Generate sample agent data for demo purposes"""
    return {
        "fundamental": {
            "status": "success",
            "analysis": "Strong fundamentals with ROE of 18.5% and low debt-to-equity ratio of 0.3x. Quarterly results show consistent growth.",
            "metrics": {
                "ROE": "18.5%",
                "Debt/Equity": "0.3x",
                "P/E Ratio": "22.4",
                "Revenue Growth": "15.2% YoY",
                "F-Score": "7/9"
            }
        },
        "technical": {
            "status": "success",
            "analysis": "Stock trading above 50-day and 200-day moving averages. RSI at 58 indicates moderate bullish momentum.",
            "metrics": {
                "RSI(14)": "58.3",
                "SMA(50)": "1250",
                "SMA(200)": "1180",
                "Support": "1220",
                "Resistance": "1350"
            }
        },
        "sentiment": {
            "status": "success",
            "analysis": "Recent news coverage is positive with 12 bullish articles vs 3 bearish. Social sentiment slightly positive.",
            "metrics": {
                "Overall Sentiment": "POSITIVE",
                "Bullish Articles": "12",
                "Bearish Articles": "3",
                "Sentiment Score": "+0.65"
            }
        },
        "derivatives": {
            "status": "success",
            "analysis": "PCR at 0.95 indicates balanced sentiment. Max Pain at 1280 suggests consolidation.",
            "metrics": {
                "NIFTY PCR": "0.95",
                "Max Pain": "1280",
                "VIX": "14.2",
                "OI Buildup": "Bullish"
            }
        },
        "risk": {
            "status": "success",
            "analysis": "Moderate risk profile. Beta of 1.2 indicates slightly higher volatility than market.",
            "metrics": {
                "Beta": "1.2",
                "Volatility (20d)": "24.5%",
                "VaR (95%)": "-3.2%",
                "Max Drawdown": "-18.5%"
            }
        },
        "cio": {
            "status": "success",
            "analysis": "Based on comprehensive analysis, recommend BUY with HIGH conviction. Favorable risk-reward ratio.",
            "verdict": "BUY",
            "conviction": 8,
            "signal": "STRONG_BUY",
            "thesis_keywords": ["strong_fundamentals", "technical_uptrend", "low_interest_rate_environment"]
        }
    }


def render_agent_full_output(agent_name: str, agent_data: dict, icon: str):
    """Render a single agent tab with full output display"""
    status = agent_data.get("status", "unknown")
    status_color = "green" if status == "success" else "yellow" if status == "warning" else "red"
    
    with st.expander(f"{icon} {agent_name.replace('_', ' ').title()} ({status.upper()})", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Analysis")
            analysis = agent_data.get("analysis", "No analysis available")
            if isinstance(analysis, str):
                st.markdown(analysis)
            else:
                st.json(analysis)
        
        with col2:
            st.markdown("### Status")
            st.markdown(f"**Status:** `{status.upper()}`")
            
            conviction = agent_data.get("conviction")
            if conviction:
                st.markdown(f"**Conviction:** {conviction}/10")
            
            confidence = agent_data.get("confidence")
            if confidence:
                st.markdown(f"**Confidence:** {confidence}")
        
        metrics = agent_data.get("metrics", {})
        if metrics:
            st.divider()
            st.markdown("### Key Metrics")
            
            metric_cols = st.columns(min(4, len(metrics)))
            metric_items = list(metrics.items())
            
            for i, (key, value) in enumerate(metric_items):
                with metric_cols[i % len(metric_cols)]:
                    st.metric(label=str(key), value=str(value))
        
        thesis = agent_data.get("thesis", [])
        if thesis:
            st.divider()
            st.markdown("### Investment Thesis")
            if isinstance(thesis, list):
                for t in thesis:
                    st.markdown(f"- {t}")
            else:
                st.markdown(thesis)
        
        reasons = agent_data.get("reasons", [])
        if reasons:
            st.divider()
            st.markdown("### Key Reasons")
            if isinstance(reasons, list):
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown(reasons)
        
        concerns = agent_data.get("concerns", [])
        if concerns:
            st.divider()
            st.markdown("### ⚠️ Risk Factors")
            if isinstance(concerns, list):
                for c in concerns:
                    st.markdown(f"- {c}")
            else:
                st.markdown(concerns)
        
        st.divider()
        with st.expander("Raw Agent Output (JSON)"):
            st.json(agent_data, expanded=False)


def render_agent_tab(agent_name: str, agent_data: dict, icon: str):
    """Render a single agent tab with expandable sections"""
    status = agent_data.get("status", "unknown")
    status_color = "green" if status == "success" else "yellow" if status == "warning" else "red"
    
    with st.expander(f"{icon} {agent_name.replace('_', ' ').title()} ({status.upper()})", expanded=False):
        if status == "success":
            analysis = agent_data.get("analysis", "No analysis available")
            st.markdown(f"**Analysis:**\n\n{analysis}")
            
            metrics = agent_data.get("metrics", {})
            if metrics:
                st.divider()
                st.markdown("**Key Metrics:**")
                
                cols = st.columns(2)
                metric_items = list(metrics.items())
                
                for i, (key, value) in enumerate(metric_items):
                    with cols[i % 2]:
                        st.metric(label=key, value=value)
        else:
            st.warning(f"Agent status: {status}")
            if "error" in agent_data:
                st.error(f"Error: {agent_data['error']}")


def render_conviction_meter(conviction: int):
    """Render a visual conviction meter"""
    bars = "█" * conviction + "░" * (10 - conviction)
    color = "green" if conviction >= 7 else "yellow" if conviction >= 5 else "red"
    
    st.markdown(f"""
    <div style="background-color: #1a1a2e; padding: 15px; border-radius: 10px;">
        <h4 style="color: white; margin: 0;">Conviction Level: {conviction}/10</h4>
        <div style="font-family: monospace; font-size: 24px; color: {color}; margin-top: 10px;">
            {bars}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict_card(verdict: str, conviction: int):
    """Render the final verdict card"""
    if verdict == "BUY":
        bg_color = "#28a745"
        text_color = "white"
    elif verdict == "SELL":
        bg_color = "#dc3545"
        text_color = "white"
    else:
        bg_color = "#ffc107"
        text_color = "black"
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
        <h1 style="margin: 0; font-size: 48px;">{verdict}</h1>
        <h3 style="margin: 10px 0 0 0;">Conviction: {conviction}/10</h3>
    </div>
    """, unsafe_allow_html=True)


def main():
    ticker = st.session_state.get("ticker", "RELIANCE")
    st.caption(f"Analyzing: {ticker}")
    
    if "analysis_results" in st.session_state and st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        agent_outputs = results.get("agent_outputs", {})
        
        if agent_outputs:
            agent_name_mapping = {
                "fundamental_analyst": ("fundamental", "📊"),
                "technical_analyst": ("technical", "📈"),
                "sentiment_analyst": ("sentiment", "💬"),
                "derivatives_analyst": ("derivatives", "📉"),
                "risk_manager": ("risk", "⚠️"),
                "cio": ("cio", "💼"),
            }
            
            tab1, tab2, tab3, tab4 = st.tabs(["All Agents", "CIO Verdict", "Comparison", "Raw Data"])
            
            with tab1:
                st.subheader("All Agent Analyses")
                
                for orig_name, (display_name, icon) in agent_name_mapping.items():
                    if orig_name in agent_outputs:
                        agent_data = agent_outputs[orig_name]
                        render_agent_full_output(display_name, agent_data, icon)
            
            with tab2:
                st.subheader("CIO Final Decision")
                
                cio_data = agent_outputs.get("cio", {})
                verdict = results.get("verdict", "HOLD")
                conviction = results.get("conviction", 5)
                
                render_verdict_card(verdict, conviction)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Thesis Summary")
                    analysis = cio_data.get("analysis", results.get("analysis", "No analysis available"))
                    st.markdown(analysis)
                    
                    keywords = results.get("thesis_keywords", [])
                    if keywords:
                        st.markdown("### Key Themes")
                        for kw in keywords:
                            st.markdown(f"- {kw.replace('_', ' ').title()}")
                
                with col2:
                    st.markdown("### Quick Stats")
                    st.metric("Signal", results.get("signal", "NEUTRAL"))
                    st.metric("Verdict", verdict)
                    st.metric("Conviction", f"{conviction}/10")
                    st.metric("Duration", f"{results.get('duration_seconds', 0):.1f}s")
            
            with tab3:
                st.subheader("Agent Comparison")
                
                agent_statuses = {}
                display_names = {
                    "fundamental_analyst": "Fundamental",
                    "technical_analyst": "Technical",
                    "sentiment_analyst": "Sentiment",
                    "derivatives_analyst": "Derivatives",
                    "risk_manager": "Risk",
                    "cio": "CIO"
                }
                
                for orig_name, display_name in display_names.items():
                    if orig_name in agent_outputs:
                        agent_data = agent_outputs[orig_name]
                        agent_statuses[display_name] = agent_data.get("status", "unknown")
                
                col1, col2, col3 = st.columns(3)
                
                agents = list(agent_statuses.keys())
                for i, agent in enumerate(agents):
                    with [col1, col2, col3][i % 3]:
                        status = agent_statuses[agent]
                        status_icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌"
                        st.markdown(f"{status_icon} **{agent}**: {status}")
            
            with tab4:
                st.subheader("Raw Analysis Data")
                
                with st.expander("Full Results JSON"):
                    st.json(results, expanded=True)
                
                st.divider()
                
                st.markdown("### Agent Outputs")
                for agent_name, agent_data in agent_outputs.items():
                    with st.expander(f"{agent_name.title()} Output"):
                        st.json(agent_data, expanded=True)
        else:
            st.warning("No agent outputs in results. Run analysis first.")
    else:
        st.info("Select a stock in the sidebar and run analysis to see agent reports")
        
        st.divider()
        st.subheader("Demo Mode - Sample Data")
        st.warning("Illustrative sample data only. This is NOT output from a real analysis run.")
        
        demo_data = get_sample_agent_data()
        
        if st.checkbox("Show sample agent reports"):
            tab1, tab2 = st.tabs(["All Agents", "CIO Verdict"])
            
            with tab1:
                render_agent_full_output("fundamental", demo_data["fundamental"], "📊")
                render_agent_full_output("technical", demo_data["technical"], "📈")
                render_agent_full_output("sentiment", demo_data["sentiment"], "💬")
                render_agent_full_output("derivatives", demo_data["derivatives"], "📉")
                render_agent_full_output("risk", demo_data["risk"], "⚠️")
            
            with tab2:
                render_verdict_card(demo_data["cio"]["verdict"], demo_data["cio"]["conviction"])
                st.markdown("### Demo Thesis")
                st.markdown(demo_data["cio"]["analysis"])


if __name__ == "__main__":
    main()
