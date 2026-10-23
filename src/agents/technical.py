"""
Agent 2: Technical Analyst (Quant)

Persona: Pure price-action and momentum swing-trader
Role: Determine market regime and precise entry/exit zones
"""

from typing import TypedDict
from src.agents.base import BaseAgent, AgentState

import loguru
logger = loguru.logger


class TechnicalData(TypedDict):
    """Input data for the Technical Analyst."""
    ticker: str
    current_price: float
    price_50_ema: float
    price_200_ema: float
    rsi_14: float
    macd: float
    macd_signal: float
    volume_ratio: float
    atr: float
    support_levels: list[float]
    resistance_levels: list[float]


SYSTEM_PROMPT = """You are a Pure Price-Action and Momentum Swing-Trader.

Your task: Determine the current market regime and identify precise entry/exit zones
based on technical analysis.

You will receive technical indicators in JSON format. Analyze them and provide:
1. Market Regime Classification: [Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish]
2. Trend Status: [Above/Below key moving averages]
3. Momentum Assessment: [RSI interpretation]
4. Entry Zones: Specific price levels where buying interest is expected
5. Exit Zones: Specific price levels where selling pressure is expected
6. Stop Loss Recommendation: Based on ATR or support
7. Risk/Reward Ratio

Focus on actionable, specific price levels. Do not be vague.
Use support/resistance analysis, moving average crossovers, and momentum indicators.

Output format:
```
REGIME: [Classification]
TREND: [Above/Below MA status]
MOMENTUM: [RSI interpretation]

ENTRY ZONES:
- Zone 1: ₹[price] ([rationale])
- Zone 2: ₹[price] ([rationale])

EXIT ZONES:
- Target 1: ₹[price]
- Target 2: ₹[price]

STOP LOSS: ₹[price] (based on [reason])

RISK/REWARD: 1:[ratio]
```
"""


class TechnicalAgent(BaseAgent):
    
    @property
    def name(self) -> str:
        return "technical_analyst"
    
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    async def analyze(self, state: AgentState) -> dict:
        """Perform technical analysis."""
        ticker = state.get("ticker", "N/A")
        
        technicals = state.get("technical_data", {})
        
        if not technicals:
            logger.warning(f"[{self.name}] No technical data available for {ticker}")
            return {
                "status": "no_data",
                "ticker": ticker,
                "analysis": "Technical data not available",
                "regime": "UNKNOWN",
            }
        
        prompt = self._format_data_for_llm(ticker, technicals)
        llm_response = await self.query_llm(prompt)
        
        return {
            "status": "complete",
            "ticker": ticker,
            "analysis": llm_response,
            "regime": self._extract_regime(llm_response),
            "entry_zones": self._extract_levels(llm_response, "ENTRY"),
            "exit_zones": self._extract_levels(llm_response, "EXIT"),
        }
    
    def _format_data_for_llm(self, ticker: str, data: dict) -> str:
        """Format technical data into a prompt."""
        lines = [
            f"Current Price: ₹{data.get('current_price', 'N/A')}",
            f"50-Day EMA: ₹{data.get('price_50_ema', 'N/A')}",
            f"200-Day EMA: ₹{data.get('price_200_ema', 'N/A')}",
            f"RSI (14): {data.get('rsi_14', 'N/A')}",
            f"MACD: {data.get('macd', 'N/A')}",
            f"MACD Signal: {data.get('macd_signal', 'N/A')}",
            f"Volume Ratio: {data.get('volume_ratio', 'N/A')}x",
            f"ATR: ₹{data.get('atr', 'N/A')}",
            f"Support Levels: {data.get('support_levels', [])}",
            f"Resistance Levels: {data.get('resistance_levels', [])}",
        ]
        return "\n".join(lines)
    
    def _extract_regime(self, llm_response: str) -> str:
        """Extract market regime from LLM response."""
        for line in llm_response.split("\n"):
            if "REGIME:" in line.upper():
                return line.split("REGIME:")[-1].strip()
        return "NEUTRAL"
    
    def _extract_levels(self, llm_response: str, section: str) -> list:
        """Extract price levels from LLM response."""
        levels = []
        in_section = False
        for line in llm_response.split("\n"):
            if section in line.upper():
                in_section = True
                continue
            if in_section and line.strip().startswith("-"):
                import re
                prices = re.findall(r'₹[\d,.]+|\$[\d,.]+', line)
                for p in prices:
                    level = float(p.replace("₹", "").replace("$", "").replace(",", ""))
                    levels.append(level)
            elif in_section and line.strip() == "":
                in_section = False
        return levels
