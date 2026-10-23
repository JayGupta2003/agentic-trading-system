"""
Agent 1: Fundamental Analyst

Persona: Conservative value investor
Role: Evaluate corporate health, profitability trajectory, valuation vs sector peers
"""

from typing import TypedDict
from src.agents.base import BaseAgent, AgentState

import loguru
logger = loguru.logger


class FundamentalData(TypedDict):
    """Input data for the Fundamental Analyst."""
    ticker: str
    ttm_eps: float
    peg_ratio: float
    roe: float
    roce: float
    debt_to_equity: float
    piotroski_f_score: int
    sales_growth_yoy: float
    profit_growth_yoy: float
    sector_peers: list[str]


SYSTEM_PROMPT = """You are a Conservative Value Investor and Fundamental Analyst.

Your task: Evaluate the corporate health, profitability trajectory, and valuation 
of a company relative to its sector peers.

You will receive financial metrics in JSON format. Analyze them and provide:
1. Financial Health Summary (2-3 sentences)
2. Major RED FLAGS (concerns that warrant caution)
3. Major GREEN FLAGS (positive indicators)
4. Valuation Verdict: [Undervalued / Fair Value / Overvalued]

Be conservative and data-driven. Do not speculate beyond what the numbers show.
Focus on fundamentals that matter for long-term value: earnings quality, debt management,
profitability trends, and reasonable valuations.

Output format:
```
ANALYSIS:
[Your 2-3 sentence summary]

RED FLAGS:
- [Issue 1]
- [Issue 2]

GREEN FLAGS:
- [Positive 1]
- [Positive 2]

VALUATION VERDICT: [Undervalued/Fair Value/Overvalued]
```
"""


class FundamentalAgent(BaseAgent):
    
    @property
    def name(self) -> str:
        return "fundamental_analyst"
    
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    async def analyze(self, state: AgentState) -> dict:
        """Perform fundamental analysis."""
        ticker = state.get("ticker", "N/A")
        
        fundamentals = state.get("fundamental_data", {})
        
        if not fundamentals:
            logger.warning(f"[{self.name}] No fundamental data available for {ticker}")
            return {
                "status": "no_data",
                "ticker": ticker,
                "analysis": "Fundamental data not available",
                "verdict": "HOLD",
            }
        
        prompt = self._format_data_for_llm(ticker, fundamentals)
        llm_response = await self.query_llm(prompt)
        
        return {
            "status": "complete",
            "ticker": ticker,
            "analysis": llm_response,
            "verdict": self._extract_verdict(llm_response),
        }
    
    def _format_data_for_llm(self, ticker: str, data: dict) -> str:
        """Format fundamental data into a prompt."""
        return f"""Analyze the following fundamental data for {ticker}:

{self._to_readable_string(data)}

Provide your analysis following the output format."""
    
    def _to_readable_string(self, data: dict) -> str:
        """Convert data dict to readable string."""
        lines = []
        for key, value in data.items():
            if key == "sector_peers":
                lines.append(f"Sector Peers: {', '.join(str(p) for p in value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def _extract_verdict(self, llm_response: str) -> str:
        """Extract valuation verdict from LLM response."""
        for line in llm_response.split("\n"):
            if "VALUATION VERDICT:" in line.upper():
                verdict = line.split("VALUATION VERDICT:")[-1].strip()
                return verdict
        return "UNKNOWN"
