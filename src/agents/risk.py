"""
Agent 5: Quantitative Risk Manager

Persona: Strict, capital-protecting risk officer
Role: Portfolio protection, position sizing, stop loss calculation
"""

from typing import TypedDict
from src.agents.base import BaseAgent, AgentState

import loguru
logger = loguru.logger


class RiskData(TypedDict):
    """Input data for the Risk Manager."""
    ticker: str
    beta: float
    stock_volatility: float
    portfolio_var_95: float
    kelly_fraction: float
    historical_win_rate: float
    avg_gain_pct: float
    avg_loss_pct: float
    atr: float
    current_price: float
    max_portfolio_var_pct: float


SYSTEM_PROMPT = """You are a Strict, Capital-Protecting Risk Officer.

Your task: Protect the portfolio. Never allow a single stock to blow up the account.

You will receive:
1. Stock Beta (volatility vs market)
2. Stock Volatility (historical)
3. Portfolio VaR (Value at Risk) at 95% confidence
4. Kelly Criterion fraction
5. Historical win rate and average gain/loss
6. ATR (Average True Range)
7. Current price
8. Maximum allowed portfolio VaR %

Your mandate: Calculate safe position sizing and strict stop losses.

Provide:
1. POSITION SIZING: Exact % of capital to allocate
2. STOP LOSS: Exact price based on ATR or support
3. MAX LOSS PER TRADE: ₹ amount
4. RISK/REWARD: Expected return vs risk
5. KELLY CRITERION: Full Kelly vs half Kelly recommendation
6. VaR CONTRIBUTION: How this trade affects portfolio risk

Rules you MUST follow:
- Never risk more than 2% of portfolio on a single trade
- Use ATR-based stops (typically 2-3x ATR)
- Apply half-Kelly for conservative position sizing
- Consider beta for market correlation risk

Output format:
```
MAXIMUM ALLOCATION: [X]% of capital
RECOMMENDED ALLOCATION: [X]% (half-Kelly adjusted)

STOP LOSS: ₹[price] ([X]x ATR)
MAX LOSS PER TRADE: ₹[amount] ([X]% of capital)

EXPECTED OUTCOME (if win rate holds):
- If Win: +₹[amount] ([X]% on capital)
- If Loss: -₹[amount]

RISK/REWARD RATIO: 1:[X]
KELLY PROBABILITY WEIGHTED: [X]%

VaR IMPACT: Low/Medium/High
FINAL VERDICT: [APPROVE / CAUTION / REJECT]
```
"""


class RiskAgent(BaseAgent):
    
    @property
    def name(self) -> str:
        return "risk_manager"
    
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    async def analyze(self, state: AgentState) -> dict:
        """Perform risk analysis."""
        ticker = state.get("ticker", "N/A")
        
        risk = state.get("risk_data", {})
        
        if not risk:
            logger.warning(f"[{self.name}] No risk data available for {ticker}")
            return {
                "status": "no_data",
                "ticker": ticker,
                "analysis": "Risk data not available",
                "position_size_pct": 0,
                "stop_loss": 0,
            }
        
        calculated = self._calculate_risk_metrics(risk)
        risk["calculated"] = calculated
        
        prompt = self._format_data_for_llm(ticker, risk)
        llm_response = await self.query_llm(prompt)
        
        return {
            "status": "complete",
            "ticker": ticker,
            "analysis": llm_response,
            "position_size_pct": calculated["recommended_allocation_pct"],
            "stop_loss": calculated["stop_loss"],
            "max_loss_per_trade": calculated["max_loss"],
            "risk_reward": calculated["risk_reward"],
            "kelly_full": calculated["kelly_full"],
            "kelly_half": calculated["kelly_half"],
        }
    
    def _calculate_risk_metrics(self, data: dict) -> dict:
        """Calculate risk metrics from data."""
        import numpy as np
        
        current_price = data.get("current_price", 0)
        atr = data.get("atr", 0)
        beta = data.get("beta", 1.0)
        win_rate = data.get("historical_win_rate", 0.5)
        avg_gain = data.get("avg_gain_pct", 10)
        avg_loss = data.get("avg_loss_pct", 5)
        kelly_frac = data.get("kelly_fraction", 0.25)
        max_var_pct = data.get("max_portfolio_var_pct", 5)
        
        if current_price == 0 or atr == 0:
            return {
                "recommended_allocation_pct": 0,
                "stop_loss": 0,
                "max_loss": 0,
                "risk_reward": 0,
                "kelly_full": 0,
                "kelly_half": 0,
            }
        
        stop_loss = current_price - (2.5 * atr)
        stop_loss_pct = (atr * 2.5 / current_price) * 100
        
        kelly_full = (win_rate * avg_gain - (1 - win_rate) * avg_loss) / avg_gain
        kelly_full = max(0, min(1, kelly_full))
        kelly_half = kelly_full * kelly_frac
        
        max_loss_pct = min(stop_loss_pct, max_var_pct)
        max_loss_pct = min(max_loss_pct, 2.0)
        
        risk_reward = avg_gain / avg_loss if avg_loss > 0 else 0
        
        return {
            "recommended_allocation_pct": round(kelly_half * 100, 1),
            "stop_loss": round(stop_loss, 2),
            "max_loss_pct": round(max_loss_pct, 2),
            "max_loss": round(current_price * (stop_loss_pct / 100), 2),
            "risk_reward": round(risk_reward, 2),
            "kelly_full": round(kelly_full * 100, 1),
            "kelly_half": round(kelly_half * 100, 1),
        }
    
    def _format_data_for_llm(self, ticker: str, data: dict) -> str:
        """Format risk data into a prompt."""
        calc = data.get("calculated", {})
        
        lines = [
            f"=== {ticker} Risk Analysis ===",
            f"Beta: {data.get('beta', 'N/A')}",
            f"Stock Volatility: {data.get('stock_volatility', 'N/A')}%",
            f"Portfolio VaR (95%): {data.get('portfolio_var_95', 'N/A')}%",
            f"Historical Win Rate: {data.get('historical_win_rate', 'N/A') * 100:.1f}%",
            f"Average Gain: {data.get('avg_gain_pct', 'N/A')}%",
            f"Average Loss: {data.get('avg_loss_pct', 'N/A')}%",
            f"ATR: ₹{data.get('atr', 'N/A')}",
            f"Current Price: ₹{data.get('current_price', 'N/A')}",
            f"Max Portfolio VaR Limit: {data.get('max_portfolio_var_pct', 5)}%",
            "",
            "CALCULATED METRICS:",
            f"Full Kelly: {calc.get('kelly_full', 'N/A')}%",
            f"Half Kelly: {calc.get('kelly_half', 'N/A')}%",
            f"Stop Loss (2.5x ATR): ₹{calc.get('stop_loss', 'N/A')}",
            f"Risk/Reward: 1:{calc.get('risk_reward', 'N/A')}",
        ]
        
        return "\n".join(lines)
