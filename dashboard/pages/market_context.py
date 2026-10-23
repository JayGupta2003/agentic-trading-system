"""
Phase 7.2.1: Market Context Page

Displays market-wide context including:
- NIFTY 50 / Bank NIFTY indices
- India VIX
- Put-Call Ratio (PCR)
- FII/DII Activity
- Sector Performance
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional

st.set_page_config(page_title="Market Context", page_icon="📰", layout="wide")

st.title("📰 Market Context")
st.markdown("Real-time market overview and derivatives data")


@st.cache_data(ttl=300)
def get_index_data(ticker: str) -> dict:
    """Fetch index data with caching"""
    try:
        data = yf.Ticker(ticker)
        info = data.info
        hist = data.history(period="5d")
        
        if len(hist) >= 2:
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((current - prev) / prev) * 100
        else:
            current = info.get('regularMarketPrice', 0)
            change = info.get('regularMarketChangePercent', 0)
        
        return {
            "name": info.get('shortName', ticker),
            "price": current,
            "change": change,
            "high": info.get('fiftyTwoWeekHigh', 0),
            "low": info.get('fiftyTwoWeekLow', 0),
        }
    except Exception as e:
        return {"name": ticker, "price": 0, "change": 0, "error": str(e)}


@st.cache_data(ttl=300)
def get_vix_data() -> dict:
    """Fetch India VIX data"""
    try:
        vix = yf.Ticker("^INDIAVIX")
        info = vix.info
        hist = vix.history(period="5d")
        
        if len(hist) >= 2:
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = current - prev
        else:
            current = info.get('regularMarketPrice', 0)
            change = 0
        
        return {
            "value": current,
            "change": change,
            "description": "India VIX - Market Volatility Index"
        }
    except Exception as e:
        return {"value": 0, "change": 0, "error": str(e)}


@st.cache_data(ttl=300)
def get_pcr_data() -> dict:
    """Fetch NIFTY PCR data from pnsea"""
    try:
        from src.data.derivatives import PNSEMarketFetcher
        fetcher = PNSEMarketFetcher()
        data = fetcher.get_pcr()
        
        if data:
            return {
                "nifty_pcr": data.get("nifty_pcr", 0),
                "banknifty_pcr": data.get("banknifty_pcr", 0),
                "timestamp": data.get("timestamp", "")
            }
        return {"error": "No PCR data available"}
    except Exception as e:
        return {"error": str(e), "nifty_pcr": 0, "banknifty_pcr": 0}


@st.cache_data(ttl=3600)
def get_sector_data() -> list:
    """Fetch sector performance data"""
    sectors = {
        "^CNXAUTO": "Auto",
        "^CNXIT": "IT",
        "^CNXPHARMA": "Pharma",
        "^CNXFMCG": "FMCG",
        "^NSEBANK": "Bank",
        "^CNXREALTY": "Realty",
        "^CNXENERGY": "Energy",
        "^CNXMETAL": "Metal",
    }
    
    results = []
    for ticker, name in sectors.items():
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="5d")
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                results.append({
                    "sector": name,
                    "change": change,
                    "status": "up" if change >= 0 else "down"
                })
        except Exception:
            results.append({"sector": name, "change": 0, "status": "error"})
    
    return results


def render_index_cards():
    """Render index overview cards"""
    indices = {
        "^NSEI": "NIFTY 50",
        "^NSEMDCP50": "NIFTY MidCap",
        "^NSEBANK": "NIFTY Bank",
        "^CNXIT": "NIFTY IT",
        "^CNXAUTO": "NIFTY Auto",
    }
    
    cols = st.columns(len(indices))
    
    for idx, (ticker, name) in enumerate(indices.items()):
        with cols[idx]:
            data = get_index_data(ticker)
            if "error" not in data:
                delta_color = "normal" if data["change"] >= 0 else "inverse"
                st.metric(
                    label=name,
                    value=f"₹{data['price']:,.0f}",
                    delta=f"{data['change']:.2f}%",
                    delta_color=delta_color
                )
            else:
                st.metric(label=name, value="Error", delta=data.get("error", "N/A"))


def render_vix_panel():
    """Render VIX panel"""
    st.subheader("📊 Volatility Index (India VIX)")
    
    vix_data = get_vix_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current VIX",
            value=f"{vix_data['value']:.2f}",
            delta=f"{vix_data['change']:+.2f}" if vix_data.get('change') else "N/A"
        )
    
    with col2:
        risk_level = "Low" if vix_data['value'] < 15 else "Medium" if vix_data['value'] < 25 else "High"
        st.metric(label="Risk Level", value=risk_level)
    
    with col3:
        interpretation = (
            "Calm markets, favorable for momentum strategies"
            if vix_data['value'] < 15
            else "Moderate volatility, favor range-bound strategies"
            if vix_data['value'] < 25
            else "High volatility, consider defensive positioning"
        )
        st.caption(interpretation)


def render_pcr_panel():
    """Render Put-Call Ratio panel"""
    st.subheader("📈 Put-Call Ratio (PCR)")
    
    pcr_data = get_pcr_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        nifty_pcr = pcr_data.get("nifty_pcr", 0)
        pcr_color = "normal" if 0.7 <= nifty_pcr <= 1.2 else "inverse"
        st.metric(
            label="NIFTY 50 PCR",
            value=f"{nifty_pcr:.2f}",
            delta="Bullish" if nifty_pcr > 1 else "Bearish" if nifty_pcr < 0.7 else "Neutral",
            delta_color=pcr_color
        )
    
    with col2:
        banknifty_pcr = pcr_data.get("banknifty_pcr", 0)
        st.metric(
            label="Bank NIFTY PCR",
            value=f"{banknifty_pcr:.2f}",
            delta="Bullish" if banknifty_pcr > 1 else "Bearish" if banknifty_pcr < 0.7 else "Neutral"
        )
    
    with st.expander("PCR Interpretation"):
        st.markdown("""
        **Put-Call Ratio (PCR)** indicates market sentiment:
        - **PCR > 1.0**: More puts bought, potentially bullish (contrarian)
        - **PCR < 0.7**: More calls bought, potentially bearish (contrarian)
        - **PCR 0.7-1.0**: Neutral sentiment
        """)


def render_sector_performance():
    """Render sector performance"""
    st.subheader("🏭 Sector Performance")
    
    sectors = get_sector_data()
    
    if sectors:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for s in sectors[:4]:
                delta = f"{s['change']:+.2f}%"
                color = "🟢" if s['status'] == 'up' else "🔴"
                st.markdown(f"{color} **{s['sector']}**: {delta}")
        
        with col2:
            for s in sectors[4:]:
                delta = f"{s['change']:+.2f}%"
                color = "🟢" if s['status'] == 'up' else "🔴"
                st.markdown(f"{color} **{s['sector']}**: {delta}")
    else:
        st.info("Sector data unavailable")


def render_market_breadth():
    """Render market breadth section"""
    st.subheader("📊 Market Breadth")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**FII/DII Activity**")
        try:
            st.info("FII/DII data requires paid API (Breeze/FinEdge)")
        except Exception:
            st.info("FII/DII data unavailable")
        
        st.markdown("**Pending:**")
        st.markdown("- Advance/Decline Ratio")
        st.markdown("- 52-week Highs/Lows")
    
    with col2:
        st.markdown("**Market Status**")
        st.markdown("- FII/DII Net Buying: *Pending*")
        st.markdown("- Foreign Flows: *Pending*")
        st.markdown("- Sectoral Performance: *Above*")
    
    with st.expander("About Market Breadth"):
        st.markdown("""
        **Market Breadth Indicators:**
        - **Advance/Decline**: Ratio of rising vs falling stocks
        - **52-week Highs/Lows**: Number of stocks at extremes
        - **FII/DII Activity**: Foreign and Domestic Institutional flows
        """)


def main():
    render_index_cards()
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        render_vix_panel()
    with col_right:
        render_pcr_panel()
    
    st.divider()
    render_sector_performance()
    st.divider()
    render_market_breadth()


if __name__ == "__main__":
    main()
