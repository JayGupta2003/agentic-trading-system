"""
Phase 7.4: Backtest Hub Page

Enhanced backtest hub with:
- Backtest Hub page structure with tabs
- Strategy selection and configuration
- Results visualization with Plotly
- Historical backtest browser
- Integration with EventDrivenBacktester
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Backtest Hub", page_icon="🔬", layout="wide")

st.title("Backtest Hub")
st.markdown("Historical backtest results and strategy comparison using Event-Driven Backtesting")

BACKTEST_DIR = "backtest_results"

if not os.path.exists(BACKTEST_DIR):
    os.makedirs(BACKTEST_DIR, exist_ok=True)


def get_sample_backtest_data() -> Dict[str, Any]:
    """Generate sample backtest data for demo"""
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(90, 0, -1)]
    
    equity = 100000
    equity_curve = []
    prices = []
    
    import random
    random.seed(42)
    
    for i in range(90):
        daily_return = random.uniform(-0.02, 0.025)
        equity *= (1 + daily_return)
        equity_curve.append(equity)
        prices.append(100 * (1 + random.uniform(-0.1, 0.1)))
    
    return {
        "name": "Sample Backtest",
        "created_at": datetime.now().isoformat(),
        "config": {
            "symbols": ["RELIANCE", "TCS"],
            "strategy": "conservative",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "initial_capital": 100000,
            "max_positions": 5,
        },
        "metrics": {
            "total_return": 18.5,
            "sharpe_ratio": 1.42,
            "max_drawdown": -8.2,
            "win_rate": 62.5,
            "total_trades": 24,
            "profit_factor": 1.85,
            "sortino_ratio": 1.68,
            "calmar_ratio": 2.25,
            "avg_trade": 772.50,
            "best_trade": 3500,
            "worst_trade": -1200,
        },
        "equity_curve": {
            "dates": dates[:len(equity_curve)],
            "values": equity_curve
        },
        "trades": [
            {"date": "2024-01-15", "symbol": "RELIANCE", "type": "BUY", "price": 2450, "shares": 20, "pnl": 500},
            {"date": "2024-01-22", "symbol": "TCS", "type": "BUY", "price": 3850, "shares": 13, "pnl": -200},
            {"date": "2024-02-05", "symbol": "RELIANCE", "type": "SELL", "price": 2550, "shares": 20, "pnl": 2000},
            {"date": "2024-02-18", "symbol": "HDFCBANK", "type": "BUY", "price": 1680, "shares": 30, "pnl": 900},
            {"date": "2024-03-01", "symbol": "INFY", "type": "BUY", "price": 1780, "shares": 28, "pnl": 1400},
        ]
    }


def list_backtests() -> List[Dict[str, Any]]:
    """List all saved backtest results"""
    if not os.path.exists(BACKTEST_DIR):
        return []
    
    files = []
    for f in os.listdir(BACKTEST_DIR):
        if f.endswith('.json'):
            filepath = os.path.join(BACKTEST_DIR, f)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            files.append({
                'name': f.replace('.json', ''),
                'file': f,
                'modified': mtime
            })
    
    return sorted(files, key=lambda x: x['modified'], reverse=True)


def load_backtest(name: str) -> Optional[Dict[str, Any]]:
    """Load a backtest result"""
    filepath = os.path.join(BACKTEST_DIR, f"{name}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def save_backtest(name: str, data: Dict[str, Any]) -> bool:
    """Save a backtest result"""
    try:
        filepath = os.path.join(BACKTEST_DIR, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving backtest: {e}")
        return False


def render_metrics_grid(metrics: Dict[str, Any]):
    """Render metrics in a grid layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = metrics.get('total_return', 0)
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric(
            label="Total Return",
            value=f"{total_return:.2f}%",
            delta=f"{total_return:.2f}%",
            delta_color=delta_color
        )
    
    with col2:
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta="Good" if sharpe >= 1.5 else "Moderate" if sharpe >= 1 else "Poor"
        )
    
    with col3:
        max_dd = metrics.get('max_drawdown', 0)
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%"
        )
    
    with col4:
        win_rate = metrics.get('win_rate', 0)
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta="Good" if win_rate >= 60 else "Moderate" if win_rate >= 50 else "Poor"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
    
    with col2:
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    
    with col3:
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
    
    with col4:
        st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")


def render_equity_curve(data: Dict[str, Any]):
    """Render equity curve with Plotly"""
    equity = data.get('equity_curve', {})
    
    if isinstance(equity, list):
        if len(equity) > 0 and isinstance(equity[0], dict) and 'date' in equity[0]:
            dates = [e.get('date', '') for e in equity]
            values = [e.get('equity', 0) for e in equity]
        elif len(equity) > 0 and 'timestamp' in equity[0]:
            dates = [e.get('timestamp', '') for e in equity]
            values = [e.get('portfolio_value', 0) for e in equity]
        else:
            dates = []
            values = []
    elif isinstance(equity, dict):
        dates = equity.get('dates', [])
        values = equity.get('values', [])
    else:
        dates = []
        values = []
    
    if not dates or not values:
        st.info("No equity curve data available")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown")
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Equity',
            line=dict(color='#28a745', width=2),
            fill='tozeroy',
            fillcolor='rgba(40,167,69,0.2)'
        ),
        row=1, col=1
    )
    
    initial = values[0] if values else 100000
    drawdowns = [(v - initial) / initial * 100 if initial > 0 else 0 for v in values]
    peak = initial
    
    for i, v in enumerate(values):
        if v > peak:
            peak = v
        drawdowns[i] = (v - peak) / peak * 100 if peak > 0 else 0
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            line=dict(color='#dc3545', width=1),
            fill='tozeroy',
            fillcolor='rgba(220,53,69,0.3)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        template="plotly_white",
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def render_trades_table(trades: List[Dict[str, Any]]):
    """Render trades in a table format"""
    if not trades:
        st.info("No trades recorded")
        return
    
    for trade in trades:
        pnl = trade.get('pnl', 0)
        pnl_color = "#28a745" if pnl >= 0 else "#dc3545"
        trade_type = trade.get('type', 'BUY')
        type_color = "#28a745" if trade_type == "BUY" else "#dc3545"
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{trade.get('date', 'N/A')}**")
        with col2:
            st.markdown(f"{trade.get('symbol', 'N/A')}")
        with col3:
            st.markdown(f"<span style='color:{type_color}'>{trade_type}</span>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"{trade.get('price', 0):.2f}")
        with col5:
            st.markdown(f"<span style='color:{pnl_color}'>{pnl:+.2f}</span>", unsafe_allow_html=True)
        
        st.divider()


def render_strategy_config():
    """Render strategy configuration panel"""
    st.subheader("Strategy Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Strategy",
            options=["conservative", "aggressive", "diversified", "momentum", "value"],
            index=0,
            help="Select trading strategy"
        )
        
        symbols = st.multiselect(
            "Symbols",
            options=["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "AXISBANK", "LT"],
            default=["RELIANCE"],
            help="Select stocks to backtest"
        )
    
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1),
            help="Backtest start date"
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime(2024, 12, 31),
            help="Backtest end date"
        )
    
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=10000,
                value=100000,
                step=10000,
                help="Starting capital"
            )
            
            max_positions = st.slider(
                "Max Positions",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum concurrent positions"
            )
        
        with col2:
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Stop loss percentage"
            )
            
            target = st.slider(
                "Target (%)",
                min_value=2.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
                help="Profit target percentage"
            )
    
    return {
        "strategy": strategy,
        "symbols": symbols,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "stop_loss": stop_loss,
        "target": target,
    }


def run_event_driven_backtest(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run event-driven backtest using the EventDrivenBacktester."""
    try:
        with st.spinner("Running event-driven backtest... This may take a few minutes."):
            from src.backtesting.event_driven import EventDrivenBacktester, BacktestConfig as EDBConfig
            
            start_date = datetime.fromisoformat(config["start_date"])
            end_date = datetime.fromisoformat(config["end_date"])
            
            cap_types = {s: "large" for s in config["symbols"]}
            
            edb_config = EDBConfig(
                initial_capital=config["initial_capital"],
                start_date=start_date,
                end_date=end_date,
                symbols=config["symbols"],
                cap_types=cap_types,
                max_positions=config["max_positions"],
                max_position_pct=20.0,
                simulate_llm_signals=True,
                signal_generation_mode="rule_based",
            )
            
            backtester = EventDrivenBacktester(edb_config)
            result = backtester.run()
            
            trades_data = []
            for trade in result.trades:
                trades_data.append({
                    "date": trade.date.strftime("%Y-%m-%d"),
                    "symbol": trade.symbol,
                    "type": trade.action,
                    "price": trade.price,
                    "shares": trade.quantity,
                    "pnl": trade.pnl,
                })
            
            equity_dates = []
            equity_values = []
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                for idx, row in result.equity_curve.iterrows():
                    equity_dates.append(idx if isinstance(idx, str) else idx.strftime("%Y-%m-%d"))
                    equity_values.append(row.get("portfolio_value", row.get("value", 0)))
            
            return {
                "name": f"{'_'.join(config['symbols'][:2])}_event_driven",
                "created_at": datetime.now().isoformat(),
                "config": {
                    "symbols": config["symbols"],
                    "strategy": config["strategy"],
                    "start_date": config["start_date"],
                    "end_date": config["end_date"],
                    "initial_capital": config["initial_capital"],
                    "max_positions": config["max_positions"],
                    "backtest_type": "event_driven",
                    "data_quality_notes": result.data_quality_notes,
                },
                "metrics": {
                    "total_return": result.metrics.total_return_pct,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "sortino_ratio": result.metrics.sortino_ratio,
                    "max_drawdown": -result.metrics.max_drawdown_pct,
                    "win_rate": result.metrics.win_rate * 100,
                    "total_trades": result.metrics.total_trades,
                    "profit_factor": result.metrics.profit_factor,
                    "avg_trade": result.metrics.avg_win_pct,
                    "winning_trades": result.metrics.winning_trades,
                    "losing_trades": result.metrics.losing_trades,
                },
                "equity_curve": {
                    "dates": equity_dates,
                    "values": equity_values,
                },
                "trades": trades_data,
                "trigger_events": [
                    {
                        "date": e.date.strftime("%Y-%m-%d"),
                        "type": e.trigger_type.value,
                        "symbol": e.symbol,
                        "agents_run": e.agents_run,
                    } for e in result.trigger_events
                ],
            }
    except Exception as e:
        import traceback
        st.error(f"Error running event-driven backtest: {str(e)}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return None


def run_backtest_from_ui(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run backtest using the event-driven backtester."""
    use_event_driven = config.get("use_event_driven", True)
    
    if use_event_driven:
        return run_event_driven_backtest(config)
    
    st.error("Only the event-driven backtester is supported. Enable use_event_driven.")
    return None


def render_backtest_comparison(backtests: List[Dict[str, Any]]):
    """Render comparison of multiple backtests"""
    if len(backtests) < 2:
        return
    
    st.subheader("Strategy Comparison")
    
    names = [b.get('name', 'Unknown') for b in backtests]
    returns = [b.get('metrics', {}).get('total_return', 0) for b in backtests]
    sharpes = [b.get('metrics', {}).get('sharpe_ratio', 0) for b in backtests]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Returns (%)", "Sharpe Ratios"),
        shared_yaxes=False
    )
    
    colors = ['#28a745' if r >= 0 else '#dc3545' for r in returns]
    
    fig.add_trace(
        go.Bar(x=names, y=returns, marker_color=colors, name="Returns"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=names, y=sharpes, marker_color='#17a2b8', name="Sharpe"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    tab1, tab2, tab3 = st.tabs(["Saved Backtests", "Run New Backtest", "Compare Strategies"])
    
    with tab1:
        backtests = list_backtests()
        
        if backtests:
            selected = st.selectbox(
                "Select a backtest",
                options=[b['name'] for b in backtests],
                format_func=lambda x: f"{x}"
            )
            
            if selected:
                data = load_backtest(selected)
                if data:
                    st.subheader(f"Backtest: {selected}")
                    
                    if 'metrics' in data:
                        render_metrics_grid(data['metrics'])
                    
                    st.divider()
                    
                    if 'equity_curve' in data:
                        render_equity_curve(data)
                    
                    st.divider()
                    
                    st.subheader("Recent Trades")
                    render_trades_table(data.get('trades', []))
                    
                    with st.expander("Full Configuration"):
                        st.json(data.get('config', {}))
                    
                    with st.expander("All Metrics"):
                        st.json(data.get('metrics', {}))
        else:
            st.info("No saved backtests found. Run a backtest to see results here.")
            
            if st.checkbox("Show sample data"):
                st.divider()
                st.subheader("Sample Backtest Data")
                st.warning("Illustrative sample data only: a random equity curve and fixed example metrics. These are NOT results from this system.")
                
                sample = get_sample_backtest_data()
                render_metrics_grid(sample['metrics'])
                
                st.divider()
                render_equity_curve(sample)
                
                st.divider()
                st.subheader("Recent Trades")
                render_trades_table(sample['trades'])
    
    with tab2:
        st.subheader("Configure New Backtest")
        
        config = render_strategy_config()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            run_button = st.button("Run Backtest", type="primary", use_container_width=True)
        
        if run_button:
            if not config["symbols"]:
                st.error("Please select at least one symbol")
            else:
                result = run_backtest_from_ui(config)
                
                if result:
                    st.success("Backtest completed successfully!")
                    
                    backtest_name = f"{'_'.join(config['symbols'][:2])}_{config['strategy']}_{datetime.now().strftime('%Y%m%d')}"
                    
                    save_data = {
                        "name": backtest_name,
                        "created_at": datetime.now().isoformat(),
                        "config": config,
                        "metrics": result.get('metrics', {}),
                        "equity_curve": result.get('equity_curve', {}),
                        "trades": result.get('trades', []),
                    }
                    
                    if save_backtest(backtest_name, save_data):
                        st.info(f"Backtest saved as '{backtest_name}'")
                    
                    st.divider()
                    st.subheader("Results")
                    
                    if 'metrics' in save_data:
                        render_metrics_grid(save_data['metrics'])
                    
                    st.divider()
                    
                    if 'equity_curve' in save_data and save_data['equity_curve']:
                        render_equity_curve(save_data)
    
    with tab3:
        backtests = list_backtests()
        
        if len(backtests) >= 2:
            selected_backtests = st.multiselect(
                "Select backtests to compare",
                options=[b['name'] for b in backtests],
                default=[b['name'] for b in backtests[:2]],
                max_selections=5
            )
            
            if selected_backtests:
                comparison_data = []
                for name in selected_backtests:
                    data = load_backtest(name)
                    if data:
                        comparison_data.append(data)
                
                if comparison_data:
                    render_backtest_comparison(comparison_data)
        else:
            st.info("Run at least 2 backtests to compare strategies")


if __name__ == "__main__":
    main()
