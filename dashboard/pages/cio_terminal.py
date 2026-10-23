"""
Phase 7.3.1: CIO Terminal Page

Enhanced CIO Terminal with:
- Color-coded verdict display (BUY/HOLD/SELL)
- Conviction meter visualization
- Trade setup details
- Risk assessment summary
- Agent alignment indicators
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="CIO Terminal", page_icon="💼", layout="wide")

st.title("💼 CIO Terminal")
st.markdown("Chief Investment Officer - Final Investment Decision")


def get_sample_cio_data():
    """Generate sample CIO data for demo"""
    return {
        "verdict": "BUY",
        "conviction": 8,
        "signal": "STRONG_BUY",
        "analysis": "Strong buy recommendation based on excellent fundamental metrics, positive technical setup, and favorable market conditions. The risk-reward ratio is attractive with limited downside risk.",
        "risk_assessment": "Moderate risk. Beta of 1.2 indicates slightly higher volatility than market. Stop loss recommended at 5% below entry.",
        "thesis_keywords": ["strong_fundamentals", "technical_uptrend", "low_interest_rate", "sector_leader"],
        "agent_alignments": {
            "fundamental": "POSITIVE",
            "technical": "BULLISH",
            "sentiment": "POSITIVE",
            "derivatives": "NEUTRAL",
            "risk": "MODERATE"
        },
        "trade_setup": {
            "entry_price": 1250.00,
            "stop_loss": 1187.50,
            "target_1": 1312.50,
            "target_2": 1375.00,
            "risk_reward": 2.0
        }
    }


def render_verdict_banner(verdict: str, conviction: int):
    """Render the main verdict banner with color coding"""
    if verdict == "BUY":
        bg_color = "#28a745"
        text_color = "white"
        icon = "✅"
    elif verdict == "SELL":
        bg_color = "#dc3545"
        text_color = "white"
        icon = "❌"
    else:
        bg_color = "#ffc107"
        text_color = "black"
        icon = "⏸️"
    
    conviction_bar = "█" * conviction + "░" * (10 - conviction)
    conviction_color = "#28a745" if conviction >= 7 else "#ffc107" if conviction >= 5 else "#dc3545"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {bg_color} 0%, {bg_color}aa 100%); 
                color: {text_color}; padding: 30px; border-radius: 15px; 
                text-align: center; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h1 style="margin: 0; font-size: 64px;">{icon} {verdict}</h1>
        <h3 style="margin: 15px 0 0 0; opacity: 0.9;">{verdict} with {conviction}/10 Conviction</h3>
        <div style="margin-top: 20px; font-family: monospace; font-size: 28px; 
                    color: {conviction_color}; background: rgba(0,0,0,0.2); 
                    padding: 10px 20px; border-radius: 8px;">
            {conviction_bar}
        </div>
        <div style="margin-top: 10px; font-size: 14px; opacity: 0.8;">
            Confidence Level: {conviction * 10}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_alignment(alignments: dict):
    """Render agent alignment indicators"""
    st.subheader("🔍 Agent Alignment")
    
    cols = st.columns(len(alignments))
    
    for idx, (agent, alignment) in enumerate(alignments.items()):
        with cols[idx]:
            if alignment in ["POSITIVE", "BULLISH", "STRONG_BUY"]:
                color = "#28a745"
                icon = "🟢"
            elif alignment in ["NEGATIVE", "BEARISH", "STRONG_SELL"]:
                color = "#dc3545"
                icon = "🔴"
            else:
                color = "#ffc107"
                icon = "🟡"
            
            st.markdown(f"""
            <div style="background: {color}22; border-left: 4px solid {color}; 
                        padding: 10px; border-radius: 5px; margin: 5px 0;">
                <div style="font-size: 24px;">{icon}</div>
                <div style="font-weight: bold;">{agent.title()}</div>
                <div style="font-size: 12px; color: {color};">{alignment}</div>
            </div>
            """, unsafe_allow_html=True)


def render_trade_setup(setup: dict):
    """Render the trade setup details"""
    st.subheader("📋 Trade Setup")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Entry Price",
            value=f"{setup.get('entry_price', 0):.2f}",
            delta="Current"
        )
    
    with col2:
        st.metric(
            label="Stop Loss",
            value=f"{setup.get('stop_loss', 0):.2f}",
            delta=f"-{(1 - setup.get('stop_loss', 0)/setup.get('entry_price', 1)) * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Target 1",
            value=f"{setup.get('target_1', 0):.2f}",
            delta=f"+{(setup.get('target_1', 0)/setup.get('entry_price', 1) - 1) * 100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Target 2",
            value=f"{setup.get('target_2', 0):.2f}",
            delta=f"+{(setup.get('target_2', 0)/setup.get('entry_price', 1) - 1) * 100:.1f}%"
        )
    
    st.divider()
    
    rr = setup.get('risk_reward', 0)
    st.markdown(f"**Risk:Reward Ratio:** 1:{rr:.1f}")
    
    if rr >= 2:
        st.success(f"Attractive R:R ratio of 1:{rr:.1f} - Good trade setup")
    elif rr >= 1.5:
        st.warning(f"Moderate R:R ratio of 1:{rr:.1f} - Acceptable trade setup")
    else:
        st.error(f"Poor R:R ratio of 1:{rr:.1f} - Not recommended")


def render_thesis_keywords(keywords: list):
    """Render investment thesis keywords"""
    st.subheader("💡 Investment Thesis")
    
    cols = st.columns(2)
    
    for i, keyword in enumerate(keywords):
        keyword_display = keyword.replace('_', ' ').title()
        with cols[i % 2]:
            st.markdown(f"- **{keyword_display}**")


def main():
    if "analysis_results" in st.session_state and st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        verdict = results.get("verdict", "HOLD")
        conviction = results.get("conviction", 5)
        analysis = results.get("analysis", "")
        risk_assessment = results.get("risk_assessment", "")
        keywords = results.get("thesis_keywords", [])
        agent_outputs = results.get("agent_outputs", {})
        
        alignments = {}
        for agent_name in ["fundamental", "technical", "sentiment", "derivatives", "risk"]:
            if agent_name in agent_outputs:
                status = agent_outputs[agent_name].get("status", "unknown")
                alignments[agent_name] = status.upper()
        
        trade_setup = results.get("trade_setup", {
            "entry_price": 1250,
            "stop_loss": 1187.50,
            "target_1": 1312.50,
            "target_2": 1375.00,
            "risk_reward": 2.0
        })
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            render_verdict_banner(verdict, conviction)
            
            st.markdown("### 📝 CIO Analysis")
            st.write(analysis)
            
            if risk_assessment:
                st.markdown("### ⚠️ Risk Assessment")
                st.write(risk_assessment)
        
        with col_right:
            render_agent_alignment(alignments)
            render_thesis_keywords(keywords)
        
        st.divider()
        render_trade_setup(trade_setup)
        
    else:
        st.info("👆 Run analysis on a stock to see the CIO verdict here")
        
        st.divider()
        st.subheader("Demo Mode - Sample CIO Decision")
        st.warning("Illustrative sample data only. This is NOT a real CIO decision.")
        
        demo = get_sample_cio_data()
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            render_verdict_banner(demo["verdict"], demo["conviction"])
            
            st.markdown("### 📝 CIO Analysis")
            st.write(demo["analysis"])
            
            st.markdown("### ⚠️ Risk Assessment")
            st.write(demo["risk_assessment"])
        
        with col_right:
            render_agent_alignment(demo["agent_alignments"])
            render_thesis_keywords(demo["thesis_keywords"])
        
        st.divider()
        render_trade_setup(demo["trade_setup"])


if __name__ == "__main__":
    main()
