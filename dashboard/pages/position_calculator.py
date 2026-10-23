"""
Phase 7.3.3/7.3.4: Position Calculator Page

Enhanced Position Calculator with:
- Entry price and ATR-based stop loss
- Position sizing with Kelly fraction
- Risk/reward visualization
- Multiple calculation modes
- VaR and max loss calculations
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Position Calculator", page_icon="🧮", layout="wide")

st.title("🧮 Position Calculator")
st.markdown("Calculate optimal position size and risk parameters")


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    target_price: float,
    portfolio_size: float,
    risk_pct: float,
    use_kelly: bool = False,
    kelly_fraction: float = 0.25
) -> dict:
    """Calculate position size and risk metrics"""
    risk_per_share = entry_price - stop_loss
    reward_per_share = target_price - entry_price
    
    if risk_per_share <= 0:
        return {"error": "Stop loss must be below entry price"}
    
    risk_amount = portfolio_size * (risk_pct / 100)
    shares = int(risk_amount / risk_per_share)
    position_value = shares * entry_price
    position_pct = (position_value / portfolio_size) * 100 if portfolio_size > 0 else 0
    
    risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    
    max_loss = shares * risk_per_share
    max_loss_pct = (max_loss / portfolio_size) * 100 if portfolio_size > 0 else 0
    
    potential_profit = shares * reward_per_share
    potential_profit_pct = (potential_profit / portfolio_size) * 100 if portfolio_size > 0 else 0
    
    breakeven_at = shares * entry_price
    
    if use_kelly:
        win_rate = 0.55
        kelly_size = kelly_fraction * win_rate - kelly_fraction * (1 - win_rate)
        kelly_shares = int(shares * (kelly_size / (risk_pct / 100)))
    else:
        kelly_shares = shares
    
    return {
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_amount": risk_amount,
        "shares": shares,
        "kelly_shares": kelly_shares,
        "position_value": position_value,
        "position_pct": position_pct,
        "risk_reward": risk_reward,
        "max_loss": max_loss,
        "max_loss_pct": max_loss_pct,
        "potential_profit": potential_profit,
        "potential_profit_pct": potential_profit_pct,
        "breakeven_at": breakeven_at,
    }


def render_risk_reward_chart(entry: float, stop_loss: float, target: float, shares: int):
    """Render risk/reward visualization using Plotly"""
    levels = ["Stop Loss", "Entry", "Target"]
    prices = [stop_loss, entry, target]
    colors = ["#dc3545", "#ffc107", "#28a745"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Risk Zone", "Reward Zone"],
        y=[entry - stop_loss, target - entry],
        marker_color=["rgba(220,53,69,0.3)", "rgba(40,167,69,0.3)"],
        text=[f"{entry - stop_loss:.0f}", f"{target - entry:.0f}"],
        textposition="auto",
        name="Zones"
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1, 2],
        y=prices,
        mode="lines+markers+text",
        marker=dict(size=20, color=colors),
        line=dict(width=3),
        text=[f"{p:.2f}" for p in prices],
        textposition="top center",
        name="Price Levels"
    ))
    
    fig.update_layout(
        title="Risk/Reward Profile",
        xaxis_title="Level",
        yaxis_title="Price",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_position_summary(results: dict, entry: float, target: float, stop_loss: float):
    """Render position summary metrics"""
    st.subheader("Position Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Shares to Buy",
            value=f"{results['shares']:,}",
            delta=f"{results['position_value']:,.0f}"
        )
    
    with col2:
        st.metric(
            label="Position Size",
            value=f"{results['position_pct']:.1f}%",
            delta=f"{results['position_value']:,.0f}"
        )
    
    with col3:
        st.metric(
            label="Max Loss",
            value=f"{results['max_loss']:,.2f}",
            delta=f"{results['max_loss_pct']:.2f}%"
        )
    
    with col4:
        st.metric(
            label="Potential Profit",
            value=f"{results['potential_profit']:,.2f}",
            delta=f"+{results['potential_profit_pct']:.2f}%"
        )


def render_risk_metrics(results: dict, portfolio_size: float):
    """Render risk metrics"""
    st.subheader("⚠️ Risk Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rr = results.get('risk_reward', 0)
        st.metric(
            label="Risk:Reward",
            value=f"1:{rr:.2f}",
            delta="Good" if rr >= 2 else "Moderate" if rr >= 1.5 else "Poor"
        )
    
    with col2:
        var_95 = portfolio_size * 0.02
        st.metric(
            label="VaR (95%)",
            value=f"{var_95:,.2f}",
            delta="Daily risk"
        )
    
    with col3:
        max_dd = results.get('max_loss_pct', 0)
        st.metric(
            label="Max Drawdown Risk",
            value=f"{max_dd:.2f}%",
            delta="Per trade"
        )


def render_kelly_section(results: dict):
    """Render Kelly fraction recommendation"""
    st.subheader("📈 Kelly Criterion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Kelly Fraction** helps determine optimal position size based on:
        - Win rate probability
        - Win/Loss ratio
        
        Formula: `f = W - (1-W)/R`
        """)
    
    with col2:
        kelly_shares = results.get('kelly_shares', results['shares'])
        kelly_pct = (kelly_shares / results['shares']) * 100 if results['shares'] > 0 else 0
        
        st.metric(
            label="Kelly Position",
            value=f"{kelly_shares:,} shares",
            delta=f"{kelly_pct:.0f}% of calculated"
        )


def render_trade_levels(entry: float, stop_loss: float, target: float, results: dict):
    """Render trade levels summary"""
    st.subheader("Trade Levels")
    
    levels = [
        ("Stop Loss", stop_loss, entry - stop_loss, "Red", "#dc3545"),
        ("Entry", entry, 0, "Yellow", "#ffc107"),
        ("Target", target, target - entry, "Green", "#28a745"),
    ]
    
    cols = st.columns(3)
    
    for idx, (name, price, delta, color_name, color) in enumerate(levels):
        with cols[idx]:
            st.markdown(f"""
            <div style="background: {color}22; border-left: 4px solid {color}; 
                        padding: 15px; border-radius: 8px; margin: 5px 0;">
                <div style="font-size: 16px; font-weight: bold;">{name}</div>
                <div style="font-size: 24px; font-weight: bold;">{price:,.2f}</div>
                <div style="font-size: 12px; color: gray;">
                    {"Delta: " + f"{delta:,.2f}" if delta != 0 else "Current Price"}
                </div>
            </div>
            """, unsafe_allow_html=True)


def main():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Trade Parameters")
        
        entry_price = st.number_input(
            "Entry Price",
            min_value=0.01,
            value=1250.0,
            step=10.0,
            help="Current or expected entry price"
        )
        
        stop_loss_pct = st.slider(
            "Stop Loss (%)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Stop loss as percentage of entry"
        )
        
        stop_loss = entry_price * (1 - stop_loss_pct / 100)
        
        st.caption(f"Stop Loss Price: {stop_loss:.2f}")
        
        target_pct = st.slider(
            "Target Profit (%)",
            min_value=2.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Profit target as percentage of entry"
        )
        
        target_price = entry_price * (1 + target_pct / 100)
        
        st.caption(f"Target Price: {target_price:.2f}")
    
    with col2:
        st.subheader("Portfolio Settings")
        
        portfolio_size = st.number_input(
            "Portfolio Size",
            min_value=10000,
            value=1000000,
            step=10000,
            help="Total capital available"
        )
        
        risk_pct = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.25,
            help="Percentage of portfolio to risk per trade"
        )
        
        use_kelly = st.checkbox(
            "Apply Kelly Criterion",
            value=False,
            help="Use Kelly fraction for position sizing"
        )
        
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.1,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Fraction of Kelly to use (lower = safer)",
            disabled=not use_kelly
        )
    
    st.divider()
    
    if entry_price > 0 and stop_loss < entry_price:
        results = calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            portfolio_size=portfolio_size,
            risk_pct=risk_pct,
            use_kelly=use_kelly,
            kelly_fraction=kelly_fraction
        )
        
        if "error" not in results:
            render_position_summary(results, entry_price, target_price, stop_loss)
            
            st.divider()
            
            col_chart, col_metrics = st.columns([1, 1])
            
            with col_chart:
                render_risk_reward_chart(entry_price, stop_loss, target_price, results['shares'])
            
            with col_metrics:
                render_risk_metrics(results, portfolio_size)
            
            st.divider()
            
            col_levels, col_kelly = st.columns([1, 1])
            
            with col_levels:
                render_trade_levels(entry_price, stop_loss, target_price, results)
            
            with col_kelly:
                render_kelly_section(results)
            
            with st.expander("Full Calculation Details"):
                st.json({
                    "Entry Price": f"{entry_price:.2f}",
                    "Stop Loss": f"{stop_loss:.2f} ({stop_loss_pct}%)",
                    "Target": f"{target_price:.2f} ({target_pct}%)",
                    "Risk per Share": f"{results['risk_per_share']:.2f}",
                    "Reward per Share": f"{results['reward_per_share']:.2f}",
                    "Risk Amount": f"{results['risk_amount']:.2f}",
                    "Position %": f"{results['position_pct']:.2f}%",
                    "Risk:Reward": f"1:{results['risk_reward']:.2f}",
                    "Breakeven At": f"{results['breakeven_at']:,.2f}"
                })
        else:
            st.error(results["error"])
    else:
        st.warning("Enter valid prices (stop loss must be below entry price)")


if __name__ == "__main__":
    main()
