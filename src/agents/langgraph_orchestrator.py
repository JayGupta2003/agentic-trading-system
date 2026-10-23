"""
LangGraph Orchestrator - Proper Agent State Management

This module implements the trigger-based agent routing and state management
for the event-driven trading system.

Key features:
- VRAM-efficient state (only summaries, not raw data)
- Trigger-based agent routing (Section 5.3 of REDESIGN_PLAN.md)
- Checkpointer for state persistence
- Conditional graph edges based on trigger type
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from src.trading.types import TriggerType, AgentName, AGENT_ROUTING
from src.llm.ollama_client import create_agent_llm, clear_vram_for_next_agent

import loguru
logger = loguru.logger


class AgenticState(TypedDict):
    """VRAM-efficient state for LangGraph agents.
    
    IMPORTANT: Only store SUMMARIES, not raw data.
    Raw data stays in Python layer; only structured summaries enter LLM context.
    """
    symbol: str
    current_date: datetime
    cap_type: str
    
    price_summary: str
    fundamental_summary: str
    news_summary: str
    derivatives_summary: str
    
    fundamental_analysis: Optional[str]
    technical_analysis: Optional[str]
    sentiment_analysis: Optional[str]
    derivatives_analysis: Optional[str]
    risk_assessment: Optional[str]
    cio_synthesis: Optional[str]
    
    current_verdict: Optional[str]
    conviction_score: Optional[int]
    
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]
    trailing_stop_atr: Optional[float]
    
    trigger_type: str
    agents_to_run: List[str]
    
    errors: List[str]
    completed_agents: List[str]


@dataclass
class ExecutionParams:
    entry_price: float
    stop_loss: float
    target_price: float
    trailing_stop_pct: float
    thesis: str
    key_reasons: List[str]
    risks: List[str]
    conviction: int
    cap_type: str


class LangGraphOrchestrator:
    def __init__(self, checkpointer: Optional[Any] = None):
        self.checkpointer = checkpointer or MemorySaver()
        self._llm_cache: Dict[str, Any] = {}
        
    def _get_llm(self, agent_name: str):
        if agent_name not in self._llm_cache:
            self._llm_cache[agent_name] = create_agent_llm(agent_name)
        return self._llm_cache[agent_name]
    
    def _clear_llm_cache(self):
        for agent_name in list(self._llm_cache.keys()):
            clear_vram_for_next_agent(agent_name)
        self._llm_cache.clear()
    
    def build_graph(self, agents_to_include: Optional[List[str]] = None) -> StateGraph:
        workflow = StateGraph(AgenticState)
        
        all_agents = ["fundamental", "technical", "sentiment", "derivatives", "risk", "cio"]
        agents = agents_to_include or all_agents
        
        for agent in agents:
            workflow.add_node(agent, self._create_agent_node(agent))
        
        edge_map = {
            "fundamental": "technical",
            "technical": "sentiment",
            "sentiment": None,
            "derivatives": "risk",
            "risk": "cio",
            "cio": END,
        }
        
        for i, agent in enumerate(agents):
            if agent == "sentiment":
                next_agents = [a for a in ["derivatives", "risk", "cio"] if a in agents]
                if next_agents:
                    workflow.add_edge(agent, next_agents[0])
                else:
                    workflow.add_edge(agent, END)
            elif agent == "cio":
                workflow.add_edge(agent, END)
            elif agent in edge_map and edge_map[agent] in agents:
                workflow.add_edge(agent, edge_map[agent])
        
        workflow.set_entry_point(agents[0] if agents else END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def build_trigger_graph(self, trigger_type: TriggerType) -> StateGraph:
        agents_to_run = [a.value for a in AGENT_ROUTING.get(trigger_type, [AgentName.CIO])]
        
        workflow = StateGraph(AgenticState)
        
        for agent in agents_to_run:
            workflow.add_node(agent, self._create_agent_node(agent))
        
        for i, agent in enumerate(agents_to_run):
            if i < len(agents_to_run) - 1:
                workflow.add_edge(agent, agents_to_run[i + 1])
            else:
                workflow.add_edge(agent, END)
        
        workflow.set_entry_point(agents_to_run[0])
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _create_agent_node(self, agent_name: str):
        async def node(state: AgenticState) -> AgenticState:
            logger.info(f"[{agent_name}] Starting analysis...")
            state["completed_agents"].append(agent_name)
            
            try:
                result = await self._run_agent(agent_name, state)
                
                if agent_name == "fundamental":
                    state["fundamental_analysis"] = result
                elif agent_name == "technical":
                    state["technical_analysis"] = result
                elif agent_name == "sentiment":
                    state["sentiment_analysis"] = result
                elif agent_name == "derivatives":
                    state["derivatives_analysis"] = result
                elif agent_name == "risk":
                    state["risk_assessment"] = result
                elif agent_name == "cio":
                    state["cio_synthesis"] = result
                    self._parse_cio_output(state, result)
                
                clear_vram_for_next_agent(agent_name)
                
            except Exception as e:
                logger.error(f"[{agent_name}] Analysis failed: {e}")
                state["errors"].append(f"{agent_name}: {str(e)}")
            
            return state
        
        return node
    
    async def _run_agent(self, agent_name: str, state: AgenticState) -> str:
        llm = self._get_llm(agent_name)
        
        prompt = self._build_prompt(agent_name, state)
        system_prompt = self._get_system_prompt(agent_name)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        
        response = await llm.ainvoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    
    def _build_prompt(self, agent_name: str, state: AgenticState) -> str:
        symbol = state["symbol"]
        
        if agent_name == "fundamental":
            return f"Analyze {symbol} fundamentals:\n\n{state.get('fundamental_summary', 'Data not available')}"
        
        elif agent_name == "technical":
            context = f"Price Data:\n{state.get('price_summary', 'Data not available')}"
            if state.get("fundamental_analysis"):
                context += f"\n\nFundamental Analysis:\n{state['fundamental_analysis'][:500]}"
            return context
        
        elif agent_name == "sentiment":
            context = f"News Summary:\n{state.get('news_summary', 'Data not available')}"
            if state.get("fundamental_analysis"):
                context += f"\n\nFundamental Analysis:\n{state['fundamental_analysis'][:500]}"
            return context
        
        elif agent_name == "derivatives":
            context = f"Derivatives Data:\n{state.get('derivatives_summary', 'Data not available')}"
            if state.get("technical_analysis"):
                context += f"\n\nTechnical Analysis:\n{state['technical_analysis'][:500]}"
            return context
        
        elif agent_name == "risk":
            parts = []
            if state.get("fundamental_analysis"):
                parts.append(f"Fundamental: {state['fundamental_analysis'][:300]}")
            if state.get("technical_analysis"):
                parts.append(f"Technical: {state['technical_analysis'][:300]}")
            return "\n\n".join(parts) if parts else "No prior analysis available"
        
        elif agent_name == "cio":
            return self._build_cio_prompt(state)
        
        return f"Analyze {symbol}"
    
    def _build_cio_prompt(self, state: AgenticState) -> str:
        sections = [f"=== {state['symbol']} - ALL AGENT ANALYSIS ===\n"]
        
        if state.get("fundamental_analysis"):
            sections.append(f"FUNDAMENTAL ANALYSIS:\n{state['fundamental_analysis'][:1500]}\n")
        if state.get("technical_analysis"):
            sections.append(f"TECHNICAL ANALYSIS:\n{state['technical_analysis'][:1500]}\n")
        if state.get("sentiment_analysis"):
            sections.append(f"SENTIMENT ANALYSIS:\n{state['sentiment_analysis'][:1500]}\n")
        if state.get("derivatives_analysis"):
            sections.append(f"DERIVATIVES ANALYSIS:\n{state['derivatives_analysis'][:1500]}\n")
        if state.get("risk_assessment"):
            sections.append(f"RISK ASSESSMENT:\n{state['risk_assessment'][:1500]}\n")
        
        sections.append("\n" + "=" * 50)
        sections.append("Provide your final trading decision.")
        
        return "\n".join(sections)
    
    def _get_system_prompt(self, agent_name: str) -> str:
        prompts = {
            "fundamental": """You are a Conservative Value Investor and Fundamental Analyst.

Analyze financial metrics and provide:
1. Financial Health Summary (2-3 sentences)
2. Major RED FLAGS
3. Major GREEN FLAGS
4. Valuation Verdict: [Undervalued / Fair Value / Overvalued]

Be conservative and data-driven.""",

            "technical": """You are a Pure Price-Action and Momentum Swing-Trader.

Determine the current market regime and identify precise entry/exit zones:
1. Market Regime: [Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish]
2. Trend Status: [Above/Below key moving averages]
3. Momentum Assessment
4. Entry Zones with specific prices
5. Exit Zones with specific prices
6. Stop Loss Recommendation based on ATR

Focus on actionable, specific price levels.""",

            "sentiment": """You are a Macro-Journalist and Insider-Tracker.

Identify potential catalysts and track Smart Money:
1. KEY CATALYSTS: Events that could move the stock
2. SMART MONEY TRACKING: Are institutions buying?
3. SENTIMENT ASSESSMENT: [Positive/Negative/Neutral]
4. RISK FACTORS

If you see large block deals or significant MF buying, flag as BULLISH.""",

            "derivatives": """You are an Options Flow Reader and Derivatives Analyst.

Identify where institutions are placing their bets:
1. Options Market Sentiment: [Bullish / Bearish / Neutral]
2. Institutional Resistance Level
3. Institutional Support Level
4. Max Pain Analysis
5. PCR Interpretation

Key: High Call OI = Resistance, High Put OI = Support""",

            "risk": """You are a Strict, Capital-Protecting Risk Officer.

Calculate safe position sizing and strict stop losses:
1. POSITION SIZING: Exact % of capital to allocate
2. STOP LOSS: Exact price based on ATR
3. MAX LOSS PER TRADE
4. RISK/REWARD

Rules:
- Never risk more than 2% on a single trade
- Use ATR-based stops (2-3x ATR)
- Apply half-Kelly for conservative sizing""",

            "cio": """You are the Chief Investment Officer - the ultimate decision maker.

Review all agent analyses and formulate the final trading decision:
1. WEIGH CONFLICTING EVIDENCE
2. SYNTHESIZE A THESIS
3. ISSUE FINAL VERDICT: Strong Buy / Buy / Hold / Sell / Strong Sell
4. PROVIDE CONVICTION SCORE: 1-10

Decision Framework:
- 40% Fundamentals (long-term thesis)
- 25% Technicals (timing)
- 20% Risk Management (capital protection)
- 15% Sentiment/Options (catalysts)

If Risk Manager is negative, issue at least HOLD or worse.

Output format:
```
FINAL VERDICT: [Strong Buy / Buy / Hold / Sell / Strong Sell]
CONVICTION SCORE: [1-10]

THESIS:
[3-4 sentence synthesis]

ENTRY STRATEGY:
- Primary Entry: [details]
- Stop Loss: ₹[price]
- Target: ₹[price]
```"""
        }
        return prompts.get(agent_name, "")
    
    def _parse_cio_output(self, state: AgenticState, output: str) -> None:
        for line in output.split("\n"):
            upper = line.upper()
            if "FINAL VERDICT:" in upper:
                state["current_verdict"] = line.split(":", 1)[-1].strip()
            elif "CONVICTION SCORE:" in upper:
                import re
                match = re.search(r'\d+', line)
                if match:
                    state["conviction_score"] = int(match.group())
        
        if state.get("conviction_score"):
            base_allocation = state["conviction_score"] * 1.5
            state["entry_price"] = state.get("entry_price", 0)
            state["stop_loss"] = state.get("stop_loss", 0)
            state["target_price"] = state.get("target_price", 0)
    
    async def run(
        self,
        symbol: str,
        trigger_type: TriggerType,
        summaries: Dict[str, str],
        cap_type: str = "large",
        config: Optional[Dict] = None,
    ) -> AgenticState:
        initial_state: AgenticState = {
            "symbol": symbol,
            "current_date": datetime.now(),
            "cap_type": cap_type,
            "price_summary": summaries.get("price", ""),
            "fundamental_summary": summaries.get("fundamental", ""),
            "news_summary": summaries.get("news", ""),
            "derivatives_summary": summaries.get("derivatives", ""),
            "fundamental_analysis": None,
            "technical_analysis": None,
            "sentiment_analysis": None,
            "derivatives_analysis": None,
            "risk_assessment": None,
            "cio_synthesis": None,
            "current_verdict": None,
            "conviction_score": None,
            "entry_price": None,
            "stop_loss": None,
            "target_price": None,
            "trailing_stop_atr": None,
            "trigger_type": trigger_type.value,
            "agents_to_run": [a.value for a in AGENT_ROUTING.get(trigger_type, [AgentName.CIO])],
            "errors": [],
            "completed_agents": [],
        }
        
        thread_id = config.get("thread_id") if config else None
        thread = {"configurable": {"thread_id": thread_id or f"{symbol}_{datetime.now().timestamp()}"}}
        
        logger.info(f"Running LangGraph for {symbol} with trigger {trigger_type.value}")
        logger.info(f"Agents to run: {initial_state['agents_to_run']}")
        
        graph = self.build_trigger_graph(trigger_type)
        result = await graph.ainvoke(initial_state, thread)
        
        logger.info(f"LangGraph complete for {symbol}. Completed agents: {result.get('completed_agents', [])}")
        
        self._clear_llm_cache()
        
        return result


def create_orchestrator(checkpointer: Optional[Any] = None) -> LangGraphOrchestrator:
    return LangGraphOrchestrator(checkpointer=checkpointer)
