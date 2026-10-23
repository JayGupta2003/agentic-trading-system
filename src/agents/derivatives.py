"""
Agent 4: Derivatives & Options Analyst (F&O Desk)

Persona: Options flow reader
Role: Identify where institutional option writers are placing their bets

IMPORTANT: This agent is SKIPPED during expiry week per Section 5.6 of REDESIGN_PLAN.md.
LLMs cannot reliably discount explicit OI collapse data during expiry week.
"""

from typing import TypedDict, Optional
from src.agents.base import BaseAgent, AgentState

import loguru
logger = loguru.logger


class OptionsData(TypedDict):
    """Input data for the Derivatives Analyst."""
    ticker: str
    pcr: float
    vix: float
    option_chain: dict
    max_pain: float
    oi_buildup_calls: dict
    oi_buildup_puts: dict
    is_expiry_week: bool


SYSTEM_PROMPT = """You are an Options Flow Reader and Derivatives Analyst.

Your task: Identify where the big option writers (institutions) are placing their bets.

In the Indian market, derivatives drive the cash market. Understanding options flow
is crucial for predicting stock direction.

You will receive:
1. Put-Call Ratio (PCR)
2. India VIX
3. Option Chain Open Interest buildup at specific strikes
4. Max Pain level
5. OI buildup for Calls and Puts

Analyze and provide:
1. Options Market Sentiment: [Bullish / Bearish / Neutral]
2. Institutional Resistance Level: Where call writing suggests institutional selling
3. Institutional Support Level: Where put writing suggests institutional buying
4. Max Pain Analysis: What level max pain suggests
5. PCR Interpretation: What the ratio tells us about overall sentiment

Key insights:
- High Call OI at a strike = Resistance (institutions shorting calls there)
- High Put OI at a strike = Support (institutions buying puts as hedge)
- PCR > 1 = More puts bought (bullish sentiment historically)
- PCR < 0.7 = More calls bought (bearish sentiment)

Output format:
```
OPTIONS SENTIMENT: [Bullish/Bearish/Neutral]
PCR INTERPRETATION: [What PCR of X.X means]

INSTITUTIONAL RESISTANCE: ₹[price] (Call writing at this level)
INSTITUTIONAL SUPPORT: ₹[price] (Put writing at this level)
MAX PAIN LEVEL: ₹[price]

VIX CONTEXT: [Current VIX level and implications]

SHORT-TERM OUTLOOK: [1-2 week directional bias]
```
"""

SYSTEM_PROMPT_EXPIRY = """You are an Options Flow Reader and Derivatives Analyst.

IMPORTANT: This is F&O expiry week. The data you receive may show artificial patterns
due to position squaring and contract rolling. Apply extra caution to your analysis.

Key considerations during expiry week:
- OI artificially collapses as institutions square positions
- PCR may be distorted
- Price action may be choppy due to derivative-related activity

Provide your analysis with these limitations in mind.
"""


class DerivativesAgent(BaseAgent):
    
    def __init__(self, is_expiry_week: bool = False):
        super().__init__()
        self._is_expiry_week = is_expiry_week
    
    @property
    def name(self) -> str:
        return "derivatives_analyst"
    
    @property
    def system_prompt(self) -> str:
        if self._is_expiry_week:
            return SYSTEM_PROMPT_EXPIRY + "\n\n" + SYSTEM_PROMPT
        return SYSTEM_PROMPT
    
    def set_expiry_week(self, is_expiry_week: bool) -> None:
        """Update expiry week flag."""
        self._is_expiry_week = is_expiry_week
    
    async def analyze(self, state: AgentState) -> dict:
        """Perform derivatives analysis."""
        ticker = state.get("ticker", "N/A")
        
        options = state.get("options_data", {})
        
        if state.get("is_expiry_week"):
            self._is_expiry_week = True
        
        if not options:
            logger.warning(f"[{self.name}] No options data available for {ticker}")
            return {
                "status": "no_data",
                "ticker": ticker,
                "analysis": "Options data not available",
                "sentiment": "NEUTRAL",
            }
        
        prompt = self._format_data_for_llm(ticker, options)
        llm_response = await self.query_llm(prompt)
        
        return {
            "status": "complete",
            "ticker": ticker,
            "analysis": llm_response,
            "sentiment": self._extract_sentiment(llm_response),
            "resistance_level": self._extract_price(llm_response, "RESISTANCE"),
            "support_level": self._extract_price(llm_response, "SUPPORT"),
            "max_pain": options.get("max_pain", 0),
            "is_expiry_week": self._is_expiry_week,
        }
    
    def _format_data_for_llm(self, ticker: str, data: dict) -> str:
        """Format options data into a prompt."""
        lines = [
            f"=== {ticker} Options Analysis ===",
            f"PCR (Put-Call Ratio): {data.get('pcr', 'N/A')}",
            f"India VIX: {data.get('vix', 'N/A')}",
            f"Max Pain: ₹{data.get('max_pain', 'N/A')}",
            "",
        ]
        
        if self._is_expiry_week:
            lines.insert(1, "[WARNING: This is F&O expiry week - expect artificial OI patterns]")
            lines.append("")
        
        if data.get("oi_buildup_calls"):
            lines.append("CALL OI BUILDUP (Top Strikes):")
            for strike, oi in list(data["oi_buildup_calls"].items())[:5]:
                lines.append(f"  ₹{strike}: {oi:,} OI")
            lines.append("")
        
        if data.get("oi_buildup_puts"):
            lines.append("PUT OI BUILDUP (Top Strikes):")
            for strike, oi in list(data["oi_buildup_puts"].items())[:5]:
                lines.append(f"  ₹{strike}: {oi:,} OI")
        
        return "\n".join(lines)
    
    def _extract_sentiment(self, llm_response: str) -> str:
        """Extract options sentiment from LLM response."""
        for line in llm_response.split("\n"):
            if "OPTIONS SENTIMENT:" in line.upper():
                return line.split("SENTIMENT:")[-1].strip()
        return "NEUTRAL"
    
    def _extract_price(self, llm_response: str, keyword: str) -> Optional[float]:
        """Extract price level from LLM response."""
        import re
        pattern = rf'{keyword}:.*?₹?([\d,]+\.?\d*)'
        match = re.search(pattern, llm_response, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
        return None
