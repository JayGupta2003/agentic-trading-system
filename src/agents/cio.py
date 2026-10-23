"""
Agent 6: Chief Investment Officer (CIO) - The Decision Maker

Persona: Pragmatic portfolio manager
Role: Resolve conflicts, synthesize thesis, formulate final trade plan

NOTE: This agent uses Groq for LLM calls (better context, faster inference).
Other agents continue to use Ollama.
"""

from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseAgent, AgentState
from src.llm.groq_client import create_cio_llm

import loguru
logger = loguru.logger


SYSTEM_PROMPT = """You are the Chief Investment Officer (CIO) of a large institutional fund.

Analyze the agent outputs below and provide a detailed investment recommendation.

IMPORTANT: Your response must include these exact lines at the END:
VERDICT: BUY (or SELL or HOLD or STRONG BUY or STRONG SELL)
CONVICTION: 8 (a number from 1 to 10)

CRITICAL:
- Write at least 500 words of detailed analysis
- VERDICT must be one of: BUY, SELL, HOLD, STRONG BUY, STRONG SELL
- CONVICTION must be a single number from 1 to 10
- Both VERDICT and CONVICTION MUST appear as the LAST TWO lines of your response

Provide your detailed analysis and end with:
VERDICT: [BUY/SELL/HOLD/STRONG BUY/STRONG SELL]
CONVICTION: [1-10]
"""


class CIOAgent(BaseAgent):
    
    @property
    def name(self) -> str:
        return "cio"
    
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def __init__(self, groq_model: str = "groq/compound"):
        super().__init__()
        self._groq_llm = None
        self.groq_model = groq_model
    
    @property
    def groq_llm(self):
        """Lazy-loaded Groq LLM instance."""
        if self._groq_llm is None:
            self._groq_llm = create_cio_llm(model=self.groq_model)
        return self._groq_llm
    
    async def analyze(self, state: AgentState) -> dict:
        """Synthesize all agent outputs into final decision."""
        ticker = state.get("ticker", "N/A")
        agent_outputs = state.get("agent_outputs", {})
        
        if not agent_outputs:
            logger.error(f"[{self.name}] No agent outputs available for synthesis")
            return {
                "status": "error",
                "ticker": ticker,
                "verdict": "ERROR",
                "analysis": "No agent outputs to synthesize",
            }
        
        prompt = self._compile_mega_prompt(ticker, agent_outputs)
        llm_response = await self.query_llm(prompt)
        
        return {
            "status": "complete",
            "ticker": ticker,
            "verdict": self._extract_verdict(llm_response),
            "conviction_score": self._extract_conviction(llm_response),
            "thesis": self._extract_thesis(llm_response),
            "analysis": llm_response,
            "entry_strategy": self._extract_entry(llm_response),
            "risk_params": self._extract_risk_params(llm_response),
        }
    
    async def query_llm(self, prompt: str) -> str:
        """Query Groq LLM (overrides parent to use Groq instead of Ollama)."""
        logger.info(f"[{self.name}] Querying Groq LLM (model: {self.groq_model})...")
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        response = await self.groq_llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    
    def _compile_mega_prompt(self, ticker: str, outputs: dict) -> str:
        """Compile all agent outputs into a mega-prompt for the CIO."""
        sections = [
            f"=== {ticker} - ALL AGENT ANALYSIS ===\n",
            "Review all analyses below and issue your final trading decision.\n",
        ]
        
        agent_names = {
            "fundamental_analyst": "1. FUNDAMENTAL ANALYSIS",
            "technical_analyst": "2. TECHNICAL ANALYSIS",
            "sentiment_analyst": "3. SENTIMENT & ALTERNATIVE DATA",
            "derivatives_analyst": "4. DERIVATIVES & OPTIONS",
            "risk_manager": "5. RISK MANAGEMENT",
        }
        
        for agent_key, agent_title in agent_names.items():
            analysis = outputs.get(agent_key, {})
            sections.append(f"\n{agent_title}:")
            sections.append("-" * 40)
            
            if isinstance(analysis, dict):
                if "analysis" in analysis:
                    sections.append(analysis["analysis"])
                else:
                    sections.append(str(analysis))
            else:
                sections.append(str(analysis))
        
        return "\n".join(sections)
    
    def _extract_verdict(self, llm_response: str) -> str:
        """Extract final verdict from LLM response."""
        import re
        
        valid_verdicts = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
        
        lines = llm_response.split("\n")
        for line in reversed(lines):
            line_upper = line.upper().strip()
            if "VERDICT:" in line_upper:
                verdict = line.split("VERDICT:")[-1].strip().upper()
                for valid in valid_verdicts:
                    if valid in verdict:
                        return valid.title()
                if verdict:
                    return verdict.title()
        
        match = re.search(r"VERDICT:\s*([A-Z\s]+?)(?:\n|CONVICTION|$)", llm_response, re.IGNORECASE)
        if match:
            verdict = match.group(1).strip().upper()
            for valid in valid_verdicts:
                if valid in verdict:
                    return valid.title()
        
        return "HOLD"
    
    def _extract_conviction(self, llm_response: str) -> int:
        """Extract conviction score from LLM response."""
        import re
        
        lines = llm_response.split("\n")
        for line in reversed(lines):
            line_upper = line.upper().strip()
            if "CONVICTION:" in line_upper:
                match = re.search(r"CONVICTION:\s*(\d+)", line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        pattern = r"CONVICTION:\s*(\d+)"
        match = re.search(pattern, llm_response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return 5
    
    def _extract_thesis(self, llm_response: str) -> str:
        """Extract thesis from LLM response."""
        in_thesis = False
        thesis_lines = []
        
        for line in llm_response.split("\n"):
            if "THESIS:" in line.upper():
                in_thesis = True
                continue
            elif in_thesis:
                if line.strip().startswith("EVIDENCE") or line.strip().startswith("-" * 10):
                    break
                if line.strip():
                    thesis_lines.append(line.strip())
        
        return " ".join(thesis_lines[:4]) if thesis_lines else "No clear thesis"
    
    def _extract_entry(self, llm_response: str) -> str:
        """Extract entry strategy from LLM response."""
        in_entry = False
        entry_lines = []
        
        for line in llm_response.split("\n"):
            if "ENTRY STRATEGY:" in line.upper():
                in_entry = True
                continue
            elif in_entry:
                if line.strip().startswith("RISK") or line.strip().startswith("MONITORING"):
                    break
                if line.strip():
                    entry_lines.append(line.strip())
        
        return "\n".join(entry_lines) if entry_lines else "No specific entry strategy"
    
    def _extract_risk_params(self, llm_response: str) -> dict:
        """Extract risk parameters from LLM response."""
        import re
        
        params = {
            "position_size": None,
            "stop_loss": None,
            "target": None,
            "time_horizon": None,
        }
        
        for line in llm_response.split("\n"):
            if "position size" in line.lower() or "allocation:" in line.lower():
                match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
                if match:
                    params["position_size"] = float(match.group(1))
            elif "stop loss" in line.lower():
                match = re.search(r"₹?([\d,]+\.?\d*)", line)
                if match:
                    params["stop_loss"] = float(match.group(1).replace(",", ""))
            elif "target" in line.lower():
                match = re.search(r"₹?([\d,]+\.?\d*)", line)
                if match:
                    params["target"] = float(match.group(1).replace(",", ""))
            elif "time horizon" in line.lower():
                params["time_horizon"] = line.split(":")[-1].strip()
        
        return params
