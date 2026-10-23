"""
Agent 3: Sentiment & Alternative Data Analyst

Persona: Macro-journalist and insider-tracker
Role: Identify catalysts and track Smart Money
"""

from typing import TypedDict
from src.agents.base import BaseAgent, AgentState

import loguru
logger = loguru.logger


class SentimentData(TypedDict):
    """Input data for the Sentiment Analyst."""
    ticker: str
    recent_headlines: list[str]
    corporate_announcements: list[str]
    bulk_deals: list[dict]
    block_deals: list[dict]
    mf_buying_recent: list[dict]


SYSTEM_PROMPT = """You are a Macro-Journalist and Insider-Tracker.

Your task: Identify potential catalysts and track "Smart Money" (institutional buying).

You will receive:
1. Recent news headlines
2. BSE/NSE corporate announcements
3. Recent Bulk and Block deals
4. Recent Mutual Fund buying activity

Analyze this data to find:
1. KEY CATALYSTS: Events that could move the stock (orders won, new products, management changes)
2. SMART MONEY TRACKING: Are promoters, mutual funds, or large institutional investors buying?
3. SENTIMENT ASSESSMENT: Overall market sentiment for this stock
4. RISK FACTORS: Negative news, insider selling, or concerning trends

If you see large block deals or significant MF buying, flag this as BULLISH.
If you see insider selling or negative news flow, flag as BEARISH.

Output format:
```
CATALYSTS:
- [Catalyst 1]
- [Catalyst 2]

SMART MONEY STATUS:
- Mutual Funds: [BUYING/SELLING/NETRAL]
- Promoters: [BUYING/SELLING/NETRAL]
- Block Deals: [Significant/None]

OVERALL SENTIMENT: [Positive/Negative/Neutral]

BEARISH FLAGS:
- [Risk 1]
- [Risk 2]
```
"""


class SentimentAgent(BaseAgent):
    
    @property
    def name(self) -> str:
        return "sentiment_analyst"
    
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    async def analyze(self, state: AgentState) -> dict:
        """Perform sentiment analysis."""
        ticker = state.get("ticker", "N/A")
        
        sentiment = state.get("sentiment_data", {})
        
        if not sentiment:
            logger.warning(f"[{self.name}] No sentiment data available for {ticker}")
            return {
                "status": "no_data",
                "ticker": ticker,
                "analysis": "Sentiment data not available",
                "sentiment": "NEUTRAL",
            }
        
        prompt = self._format_data_for_llm(ticker, sentiment)
        llm_response = await self.query_llm(prompt)
        
        return {
            "status": "complete",
            "ticker": ticker,
            "analysis": llm_response,
            "sentiment": self._extract_sentiment(llm_response),
            "smart_money": self._extract_smart_money(llm_response),
        }
    
    def _format_data_for_llm(self, ticker: str, data: dict) -> str:
        """Format sentiment data into a prompt."""
        sections = [f"=== {ticker} Sentiment Analysis ===\n"]
        
        if data.get("recent_headlines"):
            sections.append("RECENT HEADLINES:")
            for h in data["recent_headlines"][:10]:
                sections.append(f"- {h}")
            sections.append("")
        
        if data.get("corporate_announcements"):
            sections.append("CORPORATE ANNOUNCEMENTS:")
            for a in data["corporate_announcements"][:5]:
                sections.append(f"- {a}")
            sections.append("")
        
        if data.get("bulk_deals"):
            sections.append("RECENT BULK DEALS:")
            for d in data["bulk_deals"][:5]:
                sections.append(f"- {d}")
            sections.append("")
        
        if data.get("block_deals"):
            sections.append("RECENT BLOCK DEALS:")
            for d in data["block_deals"][:5]:
                sections.append(f"- {d}")
            sections.append("")
        
        if data.get("mf_buying_recent"):
            sections.append("RECENT MUTUAL FUND BUYING:")
            for mf in data["mf_buying_recent"][:5]:
                sections.append(f"- {mf}")
        
        return "\n".join(sections)
    
    def _extract_sentiment(self, llm_response: str) -> str:
        """Extract overall sentiment from LLM response."""
        for line in llm_response.split("\n"):
            if "OVERALL SENTIMENT:" in line.upper():
                return line.split("SENTIMENT:")[-1].strip()
        return "NEUTRAL"
    
    def _extract_smart_money(self, llm_response: str) -> dict:
        """Extract smart money tracking from LLM response."""
        smart_money = {"mf": "NEUTRAL", "promoters": "NEUTRAL", "block_deals": "None"}
        
        for line in llm_response.split("\n"):
            if "MUTUAL FUNDS:" in line.upper():
                smart_money["mf"] = line.split("FUNDS:")[-1].strip()
            elif "PROMOTERS:" in line.upper():
                smart_money["promoters"] = line.split("PROMOTERS:")[-1].strip()
            elif "BLOCK DEALS:" in line.upper():
                smart_money["block_deals"] = line.split("DEALS:")[-1].strip()
        
        return smart_money
