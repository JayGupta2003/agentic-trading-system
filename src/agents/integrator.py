"""
Agent Integration Module - Connects Data Provider to Agents

This module orchestrates the flow of data from the UnifiedDataProvider
to each of the 6 agents, handling VRAM management between agent executions.
"""

from typing import Dict, List, Optional
from datetime import datetime

from src.data.data_provider import data_provider
from src.agents.base import AgentState
from src.llm.ollama_client import create_agent_llm, clear_vram_for_next_agent, get_vram_status

import loguru
logger = loguru.logger


class AgentIntegrator:
    """
    Integrates data from UnifiedDataProvider with the agent system.
    
    Responsible for:
    - Fetching data for a symbol
    - Formatting data for each agent
    - Running agents sequentially with VRAM management
    - Collecting and storing agent outputs
    """

    def __init__(self):
        self.data_provider = data_provider

    def analyze_stock(self, symbol: str, run_agents: List[str] = None, screening_context: Dict = None) -> Dict:
        """
        Run complete analysis for a stock symbol.

        Args:
            symbol: Stock ticker (e.g., "RELIANCE")
            run_agents: List of agent names to run. Defaults to all agents.
            screening_context: Optional screening data (quality, valuation, momentum scores)

        Returns:
            Dict with all agent outputs and final recommendation
        """
        if run_agents is None:
            run_agents = [
                "fundamental",
                "technical",
                "sentiment",
                "derivatives",
                "risk",
                "cio"
            ]

        logger.info(f"Starting analysis for {symbol}")
        
        all_data = self.data_provider.get_agent_data(symbol)
        market_context = self.data_provider.get_market_context()

        agent_outputs = {}

        if "fundamental" in run_agents:
            logger.info("Running Fundamental Analyst...")
            agent_outputs["fundamental_analyst"] = self._run_fundamental_agent(symbol, all_data, screening_context)
            clear_vram_for_next_agent("fundamental_analyst")

        if "technical" in run_agents:
            logger.info("Running Technical Analyst...")
            agent_outputs["technical_analyst"] = self._run_technical_agent(symbol, all_data, screening_context)
            clear_vram_for_next_agent("technical_analyst")

        if "sentiment" in run_agents:
            logger.info("Running Sentiment Analyst...")
            agent_outputs["sentiment_analyst"] = self._run_sentiment_agent(symbol, all_data, screening_context)
            clear_vram_for_next_agent("sentiment_analyst")

        if "derivatives" in run_agents:
            logger.info("Running Derivatives Analyst...")
            agent_outputs["derivatives_analyst"] = self._run_derivatives_agent(symbol, all_data, screening_context)
            clear_vram_for_next_agent("derivatives_analyst")

        if "risk" in run_agents:
            logger.info("Running Risk Manager...")
            agent_outputs["risk_manager"] = self._run_risk_agent(symbol, all_data, screening_context)
            clear_vram_for_next_agent("risk_manager")

        if "cio" in run_agents:
            logger.info("Running CIO Synthesis...")
            agent_outputs["cio"] = self._run_cio_agent(symbol, agent_outputs, screening_context)
            clear_vram_for_next_agent("cio")

        vram_status = get_vram_status()
        logger.info(f"VRAM Status: {vram_status}")

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_context": market_context,
            "screening_context": screening_context,
            "agent_outputs": agent_outputs,
            "vram_status": vram_status,
        }

    def _run_fundamental_agent(self, symbol: str, all_data: Dict, screening_context: Dict = None) -> Dict:
        """Run the Fundamental Analyst agent."""
        from src.agents.fundamental import FundamentalAgent, SYSTEM_PROMPT

        fundamental_data = all_data.get("fundamental_data", {})
        prompt = self._format_fundamental_prompt(symbol, fundamental_data, screening_context)

        try:
            llm = create_agent_llm("fundamental_analyst")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            response = llm.invoke(messages)
            output = response.content if hasattr(response, "content") else str(response)
            
            return {
                "status": "success",
                "analysis": output,
                "verdict": self._extract_verdict(output, "VALUATION VERDICT"),
                "input_data": self._summarize_fundamental_input(fundamental_data),
            }
        except Exception as e:
            logger.error(f"Fundamental agent error: {e}")
            return {"status": "error", "error": str(e)}

    def _format_fundamental_prompt(self, symbol: str, data: Dict, screening_context: Dict = None) -> str:
        """Format the prompt for the Fundamental Analyst."""
        lines = [
            f"=== {symbol} Fundamental Analysis ===",
            "",
            "FINANCIAL METRICS:",
            f"- Stock P/E: {data.get('stock_pe', 'N/A')}",
            f"- ROE: {data.get('roe', 'N/A')}%",
            f"- ROCE: {data.get('roce', 'N/A')}%",
            f"- Debt/Equity: {data.get('debt_to_equity', 'N/A')}",
            f"- Piotroski F-Score: {data.get('piotroski_f_score', 'N/A')}",
            f"- OPM: {data.get('opm', 'N/A')}%",
            f"- NPM: {data.get('npm', 'N/A')}%",
            f"- Current Ratio: {data.get('current_ratio', 'N/A')}",
            f"- Market Cap: ₹{data.get('market_cap_cr', 0):,.0f} Cr" if data.get('market_cap_cr') else "",
            f"- Dividend Yield: {data.get('dividend_yield', 'N/A')}%",
            f"- Promoter Holding: {data.get('promoter_holding', 'N/A')}%",
        ]

        if screening_context:
            lines.append("")
            lines.append("SCREENING SCORE CONTEXT:")
            lines.append(f"- Composite Score: {screening_context.get('composite_score', 'N/A')}")
            quality = screening_context.get('quality', {})
            if quality:
                lines.append(f"- Piotroski F-Score: {quality.get('piotroski_fscore', 'N/A')}")
                lines.append(f"- ROE: {quality.get('roe', 'N/A')}%")

        lines.append("")
        lines.append("ANALYST FLAGS FROM SCREENER.IN:")

        pros = data.get("pros", [])
        cons = data.get("cons", [])

        if pros:
            lines.append("POSITIVE FLAGS:")
            for pro in pros[:5]:
                lines.append(f"  + {pro}")

        if cons:
            lines.append("NEGATIVE FLAGS:")
            for con in cons[:5]:
                lines.append(f"  - {con}")

        lines.append("")
        lines.append("Provide your analysis following the output format.")

        return "\n".join(lines)

    def _summarize_fundamental_input(self, data: Dict) -> Dict:
        """Create a summary of key inputs for reference."""
        return {
            "pe": data.get("stock_pe"),
            "roe": data.get("roe"),
            "roce": data.get("roce"),
            "f_score": data.get("piotroski_f_score"),
            "debt_equity": data.get("debt_to_equity"),
        }

    def _run_technical_agent(self, symbol: str, all_data: Dict, screening_context: Dict = None) -> Dict:
        """Run the Technical Analyst agent."""
        from src.agents.technical import TechnicalAgent, SYSTEM_PROMPT

        technical_data = all_data.get("technical_data", {})
        prompt = self._format_technical_prompt(symbol, technical_data, screening_context)

        try:
            llm = create_agent_llm("technical_analyst")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            response = llm.invoke(messages)
            output = response.content if hasattr(response, "content") else str(response)
            
            return {
                "status": "success",
                "analysis": output,
                "regime": self._extract_verdict(output, "REGIME"),
                "input_data": self._summarize_technical_input(technical_data),
            }
        except Exception as e:
            logger.error(f"Technical agent error: {e}")
            return {"status": "error", "error": str(e)}

    def _format_technical_prompt(self, symbol: str, data: Dict, screening_context: Dict = None) -> str:
        """Format the prompt for the Technical Analyst."""
        support = data.get("support_levels", [])
        resistance = data.get("resistance_levels", [])

        lines = [
            f"=== {symbol} Technical Analysis ===",
            "",
            "PRICE & MOVING AVERAGES:",
            f"- Current Price: ₹{data.get('current_price', 0):.2f}",
            f"- 50-Day EMA: ₹{data.get('price_50_ema', 0):.2f}",
            f"- 200-Day EMA: ₹{data.get('price_200_ema', 0):.2f}",
            "",
            "MOMENTUM INDICATORS:",
            f"- RSI (14): {data.get('rsi_14', 50):.1f}",
            f"- MACD: {data.get('macd', 0):.2f}",
            f"- MACD Signal: {data.get('macd_signal', 0):.2f}",
            f"- Volume Ratio: {data.get('volume_ratio', 1):.2f}x",
            f"- ATR: ₹{data.get('atr', 0):.2f}",
            "",
            f"SUPPORT LEVELS: {', '.join(f'₹{s:.2f}' for s in support) if support else 'N/A'}",
            f"RESISTANCE LEVELS: {', '.join(f'₹{r:.2f}' for r in resistance) if resistance else 'N/A'}",
        ]

        if screening_context:
            lines.append("")
            lines.append("SCREENING MOMENTUM CONTEXT:")
            momentum = screening_context.get('momentum', {})
            if momentum:
                lines.append(f"- 3M Return: {momentum.get('return_3m', 'N/A')}%")
                lines.append(f"- 1Y Return: {momentum.get('return_1y', 'N/A')}%")
                lines.append(f"- Above SMA200: {momentum.get('above_sma200', 'N/A')}")
            technical = screening_context.get('technical', {})
            if technical:
                lines.append(f"- RSI: {technical.get('rsi_14', 'N/A')}")

        lines.append("")
        lines.append("Provide your analysis following the output format.")

        return "\n".join(lines)

    def _summarize_technical_input(self, data: Dict) -> Dict:
        """Create a summary of key inputs for reference."""
        return {
            "price": data.get("current_price"),
            "rsi": data.get("rsi_14"),
            "macd": data.get("macd"),
            "support": data.get("support_levels", [])[:3],
            "resistance": data.get("resistance_levels", [])[:3],
        }

    def _run_sentiment_agent(self, symbol: str, all_data: Dict, screening_context: Dict = None) -> Dict:
        """Run the Sentiment Analyst agent."""
        from src.agents.sentiment import SentimentAgent, SYSTEM_PROMPT

        sentiment_data = all_data.get("sentiment_data", {})
        prompt = self._format_sentiment_prompt(symbol, sentiment_data, screening_context)

        try:
            llm = create_agent_llm("sentiment_analyst")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            response = llm.invoke(messages)
            output = response.content if hasattr(response, "content") else str(response)
            
            return {
                "status": "success",
                "analysis": output,
                "sentiment": self._extract_verdict(output, "OVERALL SENTIMENT"),
                "input_data": self._summarize_sentiment_input(sentiment_data),
            }
        except Exception as e:
            logger.error(f"Sentiment agent error: {e}")
            return {"status": "error", "error": str(e)}

    def _format_sentiment_prompt(self, symbol: str, data: Dict, screening_context: Dict = None) -> str:
        """Format the prompt for the Sentiment Analyst."""
        lines = [
            f"=== {symbol} Sentiment & Alternative Data ===",
            "",
            "RECENT NEWS HEADLINES:",
        ]

        headlines = data.get("recent_headlines", [])
        if headlines:
            for h in headlines[:10]:
                lines.append(f"- [{h.get('sentiment', 'NEUTRAL')}] {h.get('title', '')[:100]}")
        else:
            lines.append("- No recent news available")

        lines.extend([
            "",
            f"OVERALL SENTIMENT: {data.get('overall_sentiment', 'NEUTRAL')}",
            f"SMART MONEY SIGNAL: {data.get('smart_money_signal', 'NEUTRAL')}",
            "",
            "BULK DEALS (Recent):",
        ])

        bulk_deals = data.get("bulk_deals", [])
        if bulk_deals:
            for deal in bulk_deals[:5]:
                lines.append(f"- {deal.get('buy_sell', 'N/A')}: ₹{deal.get('value', 0)/1e7:.2f}Cr @ ₹{deal.get('price', 0)}")
        else:
            lines.append("- No recent bulk deals")

        lines.append("")
        lines.append("Provide your analysis following the output format.")

        return "\n".join(lines)

    def _summarize_sentiment_input(self, data: Dict) -> Dict:
        """Create a summary of key inputs for reference."""
        return {
            "headlines_count": len(data.get("recent_headlines", [])),
            "sentiment": data.get("overall_sentiment"),
            "smart_money": data.get("smart_money_signal"),
        }

    def _run_derivatives_agent(self, symbol: str, all_data: Dict, screening_context: Dict = None) -> Dict:
        """Run the Derivatives Analyst agent."""
        from src.agents.derivatives import DerivativesAgent, SYSTEM_PROMPT

        options_data = all_data.get("options_data", {})
        prompt = self._format_derivatives_prompt(symbol, options_data, screening_context)

        try:
            llm = create_agent_llm("derivatives_analyst")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            response = llm.invoke(messages)
            output = response.content if hasattr(response, "content") else str(response)
            
            return {
                "status": "success",
                "analysis": output,
                "sentiment": self._extract_verdict(output, "OPTIONS SENTIMENT"),
                "input_data": self._summarize_options_input(options_data),
            }
        except Exception as e:
            logger.error(f"Derivatives agent error: {e}")
            return {"status": "error", "error": str(e)}

    def _format_derivatives_prompt(self, symbol: str, data: Dict, screening_context: Dict = None) -> str:
        """Format the prompt for the Derivatives Analyst."""
        call_oi = data.get("oi_buildup_calls", {})
        put_oi = data.get("oi_buildup_puts", {})

        lines = [
            f"=== {symbol} Derivatives & Options Analysis ===",
            "",
            "KEY METRICS:",
            f"- Put-Call Ratio (PCR): {data.get('pcr', 1.0):.3f}",
            f"- India VIX: {data.get('vix', 15):.2f}",
            f"- Max Pain Level: ₹{data.get('max_pain', 0):.2f}" if data.get('max_pain') else "- Max Pain: N/A",
            "",
            "CALL OI BUILDUP (Top Strikes):",
        ]

        if call_oi:
            for strike, oi in list(call_oi.items())[:5]:
                lines.append(f"  ₹{strike}: {oi:,} OI")
        else:
            lines.append("  N/A")

        lines.append("")
        lines.append("PUT OI BUILDUP (Top Strikes):")

        if put_oi:
            for strike, oi in list(put_oi.items())[:5]:
                lines.append(f"  ₹{strike}: {oi:,} OI")
        else:
            lines.append("  N/A")

        lines.extend([
            "",
            "Provide your analysis following the output format.",
        ])

        return "\n".join(lines)

    def _summarize_options_input(self, data: Dict) -> Dict:
        """Create a summary of key inputs for reference."""
        return {
            "pcr": data.get("pcr"),
            "vix": data.get("vix"),
            "max_pain": data.get("max_pain"),
        }

    def _run_risk_agent(self, symbol: str, all_data: Dict, screening_context: Dict = None) -> Dict:
        """Run the Risk Manager agent."""
        from src.agents.risk import RiskAgent, SYSTEM_PROMPT

        risk_data = all_data.get("risk_data", {})
        prompt = self._format_risk_prompt(symbol, risk_data, screening_context)

        try:
            llm = create_agent_llm("risk_manager")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            response = llm.invoke(messages)
            output = response.content if hasattr(response, "content") else str(response)
            
            return {
                "status": "success",
                "analysis": output,
                "position_size": self._extract_position_size(output),
                "stop_loss": self._extract_stop_loss(output),
                "input_data": self._summarize_risk_input(risk_data),
            }
        except Exception as e:
            logger.error(f"Risk agent error: {e}")
            return {"status": "error", "error": str(e)}

    def _format_risk_prompt(self, symbol: str, data: Dict, screening_context: Dict = None) -> str:
        """Format the prompt for the Risk Manager."""
        lines = [
            f"=== {symbol} Risk Management Analysis ===",
            "",
            "POSITION PARAMETERS:",
            f"- Current Price: ₹{data.get('current_price', 0):.2f}",
            f"- ATR: ₹{data.get('atr', 0):.2f}",
            f"- Stock Beta: {data.get('beta', 1.0):.2f}",
            f"- Stock Volatility: {data.get('stock_volatility', 0):.1f}%",
            "",
            "RISK PARAMETERS:",
            f"- Max Portfolio VaR: {data.get('max_portfolio_var_pct', 5)}%",
            f"- Kelly Fraction: {data.get('kelly_fraction', 0.25) * 100:.0f}%",
            f"- Historical Win Rate: {data.get('historical_win_rate', 0.55) * 100:.0f}%",
            f"- Avg Gain: {data.get('avg_gain_pct', 8)}%",
            f"- Avg Loss: {data.get('avg_loss_pct', 4)}%",
        ]

        if screening_context:
            lines.append("")
            lines.append("SCREENING SCORE CONTEXT:")
            lines.append(f"- Composite Score: {screening_context.get('composite_score', 'N/A')}")
            valuation = screening_context.get('valuation', {})
            if valuation:
                lines.append(f"- P/E Ratio: {valuation.get('pe_ratio', 'N/A')}")

        lines.append("")
        lines.append("Provide your risk analysis and position sizing.")

        return "\n".join(lines)

    def _summarize_risk_input(self, data: Dict) -> Dict:
        """Create a summary of key inputs for reference."""
        return {
            "price": data.get("current_price"),
            "atr": data.get("atr"),
            "kelly_fraction": data.get("kelly_fraction"),
        }

    def _run_cio_agent(self, symbol: str, agent_outputs: Dict, screening_context: Dict = None) -> Dict:
        """Run the CIO agent to synthesize all outputs."""
        from src.agents.cio import CIOAgent, SYSTEM_PROMPT

        prompt = self._format_cio_prompt(symbol, agent_outputs, screening_context)

        try:
            llm = create_agent_llm("cio")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            
            response = llm.invoke(messages)
            output = response.content if hasattr(response, "content") else str(response)
            
            return {
                "status": "success",
                "analysis": output,
                "verdict": self._extract_verdict(output, "FINAL VERDICT"),
                "conviction": self._extract_conviction(output),
            }
        except Exception as e:
            logger.error(f"CIO agent error: {e}")
            return {"status": "error", "error": str(e)}

    def _format_cio_prompt(self, symbol: str, agent_outputs: Dict, screening_context: Dict = None) -> str:
        """Format the mega-prompt for the CIO."""
        lines = [
            f"=== {symbol} - ALL AGENT ANALYSIS ===",
            "",
            "Review all analyses below and issue your FINAL TRADING DECISION.",
            "",
        ]

        if screening_context:
            lines.append("SCREENING CONTEXT:")
            lines.append(f"- Composite Score: {screening_context.get('composite_score', 'N/A')}")
            quality = screening_context.get('quality', {})
            if quality:
                lines.append(f"- Piotroski F-Score: {quality.get('piotroski_fscore', 'N/A')}")
            momentum = screening_context.get('momentum', {})
            if momentum:
                lines.append(f"- 3M Return: {momentum.get('return_3m', 'N/A')}%")
            lines.append("")

        agent_names = {
            "fundamental_analyst": "1. FUNDAMENTAL ANALYSIS",
            "technical_analyst": "2. TECHNICAL ANALYSIS",
            "sentiment_analyst": "3. SENTIMENT & ALTERNATIVE DATA",
            "derivatives_analyst": "4. DERIVATIVES & OPTIONS",
            "risk_manager": "5. RISK MANAGEMENT",
        }

        for agent_key, agent_title in agent_names.items():
            analysis = agent_outputs.get(agent_key, {})
            lines.append(f"\n{agent_title}:")
            lines.append("-" * 40)

            if isinstance(analysis, dict):
                if "analysis" in analysis:
                    lines.append(analysis["analysis"][:2000])
                elif "error" in analysis:
                    lines.append(f"Error: {analysis['error']}")
            else:
                lines.append(str(analysis)[:2000])

        lines.extend([
            "",
            "=" * 50,
            "Provide your final trading decision.",
        ])

        return "\n".join(lines)

    def _extract_verdict(self, text: str, keyword: str) -> str:
        """Extract a verdict/decision from agent output."""
        for line in text.split("\n"):
            if keyword.upper() in line.upper():
                return line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        return "UNKNOWN"

    def _extract_conviction(self, text: str) -> int:
        """Extract conviction score from CIO output."""
        import re
        pattern = r"CONVICTION\s*SCORE[:\s]*(\d+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 5

    def _extract_position_size(self, text: str) -> float:
        """Extract position size percentage from risk output."""
        import re
        pattern = r"(?:MAXIMUM\s+)?RECOMMENDED\s+ALLOCATION[:\s]*(\d+(?:\.\d+)?)\s*%"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0

    def _extract_stop_loss(self, text: str) -> float:
        """Extract stop loss price from risk output."""
        import re
        pattern = r"STOP\s+LOSS[:\s]*₹?([\d,]+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
        return 0.0


agent_integrator = AgentIntegrator()


def analyze_stock(symbol: str, run_agents: List[str] = None, screening_context: Dict = None) -> Dict:
    """
    Convenience function to analyze a stock.
    
    Usage:
        result = analyze_stock("RELIANCE")
        result = analyze_stock("RELIANCE", screening_context={"composite_score": 52.9})
    
    Args:
        symbol: Stock ticker (e.g., "RELIANCE")
        run_agents: List of agent names to run. Defaults to all agents.
        screening_context: Optional screening data (quality, valuation, momentum scores)
    
    Returns:
        Dict with all agent outputs and final recommendation
    """
    return agent_integrator.analyze_stock(symbol, run_agents, screening_context)


if __name__ == "__main__":
    print("=== Testing Agent Integration ===")
    print("Note: This will run all 6 agents sequentially.")
    print("VRAM will be cleared between agents.")
    print()
    
    result = analyze_stock("RELIANCE")
    
    print(f"\nAnalysis for {result['symbol']} complete!")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nAgents Run:")
    for agent, output in result["agent_outputs"].items():
        status = output.get("status", "unknown")
        print(f"  - {agent}: {status}")
    
    if "cio" in result["agent_outputs"]:
        cio_output = result["agent_outputs"]["cio"]
        print(f"\nFINAL VERDICT: {cio_output.get('verdict', 'N/A')}")
        print(f"CONVICTION SCORE: {cio_output.get('conviction', 'N/A')}/10")
