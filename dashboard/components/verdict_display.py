"""
Phase 7.3.2: Verdict Display Component

Reusable verdict display component with:
- Color-coded verdict banner
- Conviction meter
- Signal indicator
"""

import streamlit as st
from typing import Optional, Dict, Any


def render_verdict_display(
    verdict: str,
    conviction: int,
    signal: Optional[str] = None,
    analysis: Optional[str] = None,
    expanded: bool = False
) -> None:
    """Render a complete verdict display"""
    
    if verdict == "BUY":
        bg_color = "#28a745"
        text_color = "white"
        verdict_icon = "✅"
    elif verdict == "SELL":
        bg_color = "#dc3545"
        text_color = "white"
        verdict_icon = "❌"
    else:
        bg_color = "#ffc107"
        text_color = "black"
        verdict_icon = "⏸️"
    
    conviction_bar = "█" * conviction + "░" * (10 - conviction)
    conviction_color = "#28a745" if conviction >= 7 else "#ffc107" if conviction >= 5 else "#dc3545"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {bg_color} 0%, {bg_color}cc 100%); 
                color: {text_color}; padding: 25px; border-radius: 12px; 
                text-align: center; margin: 15px 0; box-shadow: 0 3px 10px rgba(0,0,0,0.2);">
        <h2 style="margin: 0; font-size: 48px;">{verdict_icon} {verdict}</h2>
        <h4 style="margin: 10px 0 0 0; opacity: 0.9;">Conviction: {conviction}/10</h4>
        <div style="margin-top: 15px; font-family: monospace; font-size: 24px; 
                    color: {conviction_color}; background: rgba(0,0,0,0.2); 
                    padding: 8px 15px; border-radius: 6px; display: inline-block;">
            {conviction_bar}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if signal:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Signal", signal)
        with col2:
            st.metric("Confidence", f"{conviction * 10}%")
    
    if analysis and expanded:
        st.divider()
        st.markdown("### Analysis")
        st.write(analysis)


def render_verdict_badge(verdict: str, conviction: int) -> None:
    """Render a compact verdict badge"""
    
    if verdict == "BUY":
        color = "#28a745"
    elif verdict == "SELL":
        color = "#dc3545"
    else:
        color = "#ffc107"
    
    st.markdown(f"""
    <span style="background: {color}; color: white; padding: 5px 15px; 
                 border-radius: 20px; font-weight: bold; font-size: 14px;">
        {verdict} ({conviction}/10)
    </span>
    """, unsafe_allow_html=True)


def render_conviction_bar(conviction: int, show_label: bool = True) -> None:
    """Render a conviction progress bar"""
    
    pct = conviction * 10
    
    if conviction >= 7:
        color = "#28a745"
        label = "High"
    elif conviction >= 5:
        color = "#ffc107"
        label = "Medium"
    else:
        color = "#dc3545"
        label = "Low"
    
    st.markdown(f"""
    <div style="margin: 10px 0;">
        {"<span style='font-weight: bold;'>Conviction:</span> " if show_label else ""}
        <span style='color: {color}; font-weight: bold;'>{label}</span>
        <div style="background: #eee; border-radius: 10px; height: 20px; width: 100%; margin-top: 5px;">
            <div style="background: {color}; border-radius: 10px; height: 100%; width: {pct}%; 
                        display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">
                {pct}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_verdict_color(verdict: str) -> str:
    """Get hex color for verdict"""
    colors = {
        "BUY": "#28a745",
        "SELL": "#dc3545",
        "HOLD": "#ffc107",
        "STRONG_BUY": "#28a745",
        "STRONG_SELL": "#dc3545",
    }
    return colors.get(verdict.upper(), "#6c757d")


def format_verdict_for_display(verdict: str) -> str:
    """Format verdict string for display"""
    return verdict.replace("_", " ").title()
