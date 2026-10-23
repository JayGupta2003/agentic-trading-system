"""
Base Agent class for LangGraph integration.

All 6 agents (Fundamental, Technical, Sentiment, Derivatives, Risk, CIO) 
inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.llm.ollama_client import create_agent_llm

import loguru
logger = loguru.logger


class AgentState(TypedDict):
    """Shared state for all agents in the pipeline."""
    ticker: str
    agent_outputs: dict
    current_agent: str
    pipeline_stage: str
    error: str | None


class BaseAgent(ABC):
    """
    Abstract base class for all trading system agents.
    
    Each agent must implement:
    - name: str - Unique identifier
    - system_prompt: str - Role definition
    - analyze(state: AgentState) -> AgentState - Analysis logic
    """

    def __init__(self):
        self._llm = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent persona and task."""
        pass

    @property
    def description(self) -> str:
        """Brief description of agent's role."""
        return f"{self.name} Agent"

    @property
    def llm(self):
        """Lazy-loaded LLM instance."""
        if self._llm is None:
            self._llm = create_agent_llm(self.name)
        return self._llm

    def create_node(self):
        """Create a LangGraph node function for this agent."""
        async def node(state: AgentState) -> AgentState:
            logger.info(f"[{self.name}] Starting analysis...")
            state["current_agent"] = self.name
            state["pipeline_stage"] = self.name
            
            try:
                result = await self.analyze(state)
                state["agent_outputs"][self.name] = result
                logger.info(f"[{self.name}] Analysis complete")
            except Exception as e:
                logger.error(f"[{self.name}] Analysis failed: {e}")
                state["error"] = str(e)
                state["agent_outputs"][self.name] = {"error": str(e)}
            
            return state
        
        return node

    @abstractmethod
    async def analyze(self, state: AgentState) -> dict:
        """
        Perform analysis based on agent's specialty.
        
        Args:
            state: Current pipeline state containing ticker and prior agent outputs
            
        Returns:
            dict: Analysis results specific to this agent's specialty
        """
        pass

    def format_input_for_llm(self, state: AgentState) -> str:
        """Format the state data into a prompt for the LLM."""
        return f"Analyze ticker: {state.get('ticker', 'N/A')}"

    async def query_llm(self, prompt: str) -> str:
        """Query the LLM with the agent's system prompt and user input."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content if hasattr(response, "content") else str(response)


class SequentialPipeline:
    """
    Orchestrates multiple agents in sequential order.
    
    Executes agents one at a time to respect VRAM constraints:
    1 -> 2 -> 3 -> 4 -> 5 -> 6 (CIO)
    """

    def __init__(self, agents: list[BaseAgent]):
        self.agents = {agent.name: agent for agent in agents}
        self.graph = self._build_graph(agents)

    def _build_graph(self, agents: list[BaseAgent]) -> StateGraph:
        """Build the LangGraph state graph."""
        workflow = StateGraph(AgentState)
        
        for agent in agents:
            workflow.add_node(agent.name, agent.create_node())
        
        for i, agent in enumerate(agents):
            if i < len(agents) - 1:
                workflow.add_edge(agent.name, agents[i + 1].name)
            else:
                workflow.add_edge(agent.name, END)
        
        workflow.set_entry_point(agents[0].name)
        
        return workflow.compile()

    async def run(self, ticker: str) -> AgentState:
        """Execute the full pipeline for a given ticker."""
        initial_state: AgentState = {
            "ticker": ticker,
            "agent_outputs": {},
            "current_agent": "",
            "pipeline_stage": "",
            "error": None,
        }
        
        logger.info(f"Starting pipeline for {ticker}")
        result = await self.graph.ainvoke(initial_state)
        logger.info(f"Pipeline complete for {ticker}")
        
        return result


def create_agent_graph(agent: BaseAgent) -> StateGraph:
    """Create a standalone graph for a single agent (for testing)."""
    workflow = StateGraph(AgentState)
    workflow.add_node(agent.name, agent.create_node())
    workflow.set_entry_point(agent.name)
    workflow.add_edge(agent.name, END)
    return workflow.compile()
