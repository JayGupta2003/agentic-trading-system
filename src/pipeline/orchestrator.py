"""
Pipeline Orchestrator - Connects Screening to Agent Pipeline

Orchestrates the full workflow from screening to agent analysis.

Phase 5.2: Batch Analysis Integration
- Supports analyzing multiple stocks in sequence
- Tracks per-stock timing and progress
- Provides batch summary statistics
- Error recovery: continues on individual stock failures
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import time
import json

from src.screening.screener import stock_shortlister, run_daily_screening
from src.agents.integrator import analyze_stock
from src.backtesting.interface import BacktesterInterface
from config.settings import settings

import loguru
logger = loguru.logger


class PipelineOrchestrator:
    """
    Orchestrates the complete trading pipeline:
    1. Screen stocks
    2. Run agent analysis on screened stocks (batch mode)
    3. Aggregate and rank results
    
    Phase 5.2 Features:
    - Batch processing with progress tracking
    - Per-stock timing and ETA calculation
    - Batch summary statistics
    - Error recovery (continue on individual failures)
    """

    def __init__(self):
        self.output_dir = Path(settings.data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backtester_interface = None

    def run_full_pipeline(
        self,
        universe: str = "NIFTY50",
        min_fscore: int = 4,
        max_pe: float = 40,
        min_momentum: float = -15,
        top_n: int = 5,
        run_agents: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        Run the complete pipeline from screening to agent analysis.

        Args:
            universe: Stock universe to screen (NIFTY50, NIFTY100, NIFTY500)
            min_fscore: Minimum Piotroski F-Score
            max_pe: Maximum P/E ratio
            min_momentum: Minimum 3-month momentum (%)
            top_n: Number of stocks to analyze
            run_agents: Whether to run agent analysis on screened stocks
            verbose: Print detailed progress

        Returns:
            Dict with screening results and agent analyses
        """
        logger.info("Starting full pipeline")
        pipeline_start = datetime.now()

        screening_result = stock_shortlister.run_screening(
            universe=universe,
            min_fscore=min_fscore,
            max_pe=max_pe,
            min_momentum=min_momentum,
            top_n=top_n,
            save_output=True,
        )

        output = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "pipeline_duration_seconds": None,
            "screening": screening_result,
            "analyses": [],
            "ranked_results": [],
        }

        if screening_result["status"] == "no_results" or not screening_result.get("stocks"):
            logger.warning("No stocks passed screening criteria")
            output["status"] = "no_stocks"
            return output

        if not run_agents:
            pipeline_end = datetime.now()
            output["pipeline_duration_seconds"] = (pipeline_end - pipeline_start).total_seconds()
            return output

        screened_stocks = screening_result["stocks"]

        for i, stock in enumerate(screened_stocks, 1):
            symbol = stock["symbol"]
            
            if verbose:
                logger.info(f"Running analysis for {symbol} ({i}/{len(screened_stocks)})")
            
            try:
                analysis = analyze_stock(symbol, screening_context=stock)
                
                agent_outputs = analysis.get("agent_outputs", {})
                cio_analysis = agent_outputs.get("cio", {})
                
                stock_result = {
                    "symbol": symbol,
                    "screening_score": stock["composite_score"],
                    "quality": stock.get("quality", {}),
                    "valuation": stock.get("valuation", {}),
                    "momentum": stock.get("momentum", {}),
                    "analysis": cio_analysis.get("analysis", ""),
                    "analysis_status": cio_analysis.get("status", "unknown"),
                }
                
                output["analyses"].append(stock_result)
                
                if verbose:
                    logger.info(f"Completed {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                output["analyses"].append({
                    "symbol": symbol,
                    "screening_score": stock["composite_score"],
                    "analysis_status": "error",
                    "error": str(e),
                })

        output["ranked_results"] = self._rank_results(output["analyses"])

        pipeline_end = datetime.now()
        output["pipeline_duration_seconds"] = (pipeline_end - pipeline_start).total_seconds()
        
        logger.info(f"Pipeline complete in {output['pipeline_duration_seconds']:.1f}s")
        
        self._save_pipeline_output(output)
        
        return output

    def analyze_batch(
        self,
        stocks: List[Dict],
        verbose: bool = True,
        skip_existing: bool = True,
    ) -> Dict:
        """
        Phase 5.2.1: Analyze a batch of stocks with progress tracking.
        
        This method provides enhanced batch processing with:
        - Progress tracking (current/total, ETA)
        - Per-stock timing
        - Batch summary statistics
        - Error recovery (continues on individual failures)

        Args:
            stocks: List of stock dicts with symbol, screening scores, etc.
                   Format: [{"symbol": "RELIANCE", "composite_score": 52.9, ...}, ...]
            verbose: Print detailed progress to logs
            skip_existing: Skip stocks that already have successful analysis (future feature)

        Returns:
            Dict with batch analysis results and summary statistics
            
        Example:
            stocks = [
                {"symbol": "RELIANCE", "composite_score": 52.9, "quality": {...}, "valuation": {...}},
                {"symbol": "TCS", "composite_score": 75.5, "quality": {...}, "valuation": {...}},
            ]
            result = orchestrator.analyze_batch(stocks)
            print(result["summary"]["total_stocks"])
            print(result["summary"]["successful"])
        """
        batch_start = datetime.now()
        total_stocks = len(stocks)
        
        if verbose:
            logger.info(f"Starting batch analysis: {total_stocks} stocks")
        
        results = []
        stock_timings = []
        errors = []
        
        for i, stock in enumerate(stocks, 1):
            symbol = stock["symbol"]
            stock_start = time.time()
            
            if verbose:
                logger.info(f"[{i}/{total_stocks}] Analyzing {symbol}...")
            
            try:
                analysis = analyze_stock(symbol, screening_context=stock)
                stock_end = time.time()
                stock_duration = stock_end - stock_start
                stock_timings.append(stock_duration)
                
                agent_outputs = analysis.get("agent_outputs", {})
                cio_output = agent_outputs.get("cio", {})
                cio_analysis_text = cio_output.get("analysis", "")
                
                cio_signals = self.extract_cio_signals(cio_analysis_text)
                
                stock_result = {
                    "symbol": symbol,
                    "screening_score": stock.get("composite_score", 0),
                    "quality": stock.get("quality", {}),
                    "valuation": stock.get("valuation", {}),
                    "momentum": stock.get("momentum", {}),
                    "technical": stock.get("technical", {}),
                    "analysis": cio_analysis_text,
                    "analysis_status": cio_output.get("status", "unknown"),
                    "verdict": cio_signals["verdict"],
                    "conviction": cio_signals["conviction"],
                    "signal": cio_signals["signal"],
                    "thesis_keywords": cio_signals["thesis_keywords"],
                    "all_agent_statuses": {
                        key: val.get("status", "unknown") 
                        for key, val in agent_outputs.items()
                    },
                    "duration_seconds": round(stock_duration, 2),
                    "agent_outputs": agent_outputs,
                }
                
                results.append(stock_result)
                
                if verbose:
                    avg_time = sum(stock_timings) / len(stock_timings)
                    remaining = total_stocks - i
                    eta_seconds = avg_time * remaining
                    eta_str = self._format_duration(eta_seconds)
                    logger.info(f"  -> {symbol} done ({stock_duration:.1f}s). ETA: {eta_str}")
                
            except Exception as e:
                stock_end = time.time()
                stock_duration = stock_end - stock_start
                
                logger.error(f"Failed to analyze {symbol}: {e}")
                errors.append({"symbol": symbol, "error": str(e)})
                
                results.append({
                    "symbol": symbol,
                    "screening_score": stock.get("composite_score", 0),
                    "analysis_status": "error",
                    "error": str(e),
                    "duration_seconds": round(stock_duration, 2),
                })
        
        batch_end = datetime.now()
        total_duration = (batch_end - batch_start).total_seconds()
        
        successful = [r for r in results if r.get("analysis_status") == "success"]
        failed = [r for r in results if r.get("analysis_status") == "error"]
        
        summary = {
            "total_stocks": total_stocks,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": round(len(successful) / total_stocks * 100, 1) if total_stocks > 0 else 0,
            "total_duration_seconds": round(total_duration, 2),
            "avg_time_per_stock": round(sum(stock_timings) / len(stock_timings), 2) if stock_timings else 0,
            "min_time": round(min(stock_timings), 2) if stock_timings else 0,
            "max_time": round(max(stock_timings), 2) if stock_timings else 0,
            "errors": errors,
        }
        
        ranked_results = self._rank_results(results)
        
        output = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "batch_duration_seconds": round(total_duration, 2),
            "results": results,
            "ranked_results": ranked_results,
            "summary": summary,
        }
        
        if verbose:
            logger.info("=" * 50)
            logger.info("BATCH ANALYSIS COMPLETE")
            logger.info(f"  Total: {summary['total_stocks']}")
            logger.info(f"  Successful: {summary['successful']}")
            logger.info(f"  Failed: {summary['failed']}")
            logger.info(f"  Success Rate: {summary['success_rate']}%")
            logger.info(f"  Total Time: {self._format_duration(total_duration)}")
            logger.info(f"  Avg per stock: {summary['avg_time_per_stock']}s")
            logger.info("=" * 50)
        
        self._save_batch_output(output)
        
        return output

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _save_batch_output(self, output: Dict) -> Path:
        """Save batch analysis output to file."""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_output_{timestamp}.json"
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved batch output to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save batch output: {e}")

        return filepath

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Phase 5.2.2: Aggregate batch analysis results into summary statistics.
        
        This method creates aggregated insights across all analyzed stocks:
        - Signal distribution (BULLISH/BEARISH/NEUTRAL counts)
        - Quality metrics aggregation (avg F-Score, avg P/E)
        - Momentum metrics aggregation (avg 3M return)
        - Recommended portfolio allocation
        - Top picks based on composite scoring

        Args:
            results: List of stock analysis results from analyze_batch

        Returns:
            Dict with aggregated statistics and recommendations
            
        Example:
            results = batch_result["results"]
            aggregation = orchestrator.aggregate_results(results)
            print(aggregation["signal_distribution"])
            print(aggregation["portfolio_allocation"])
        """
        if not results:
            return {"error": "No results to aggregate"}
        
        successful = [r for r in results if r.get("analysis_status") == "success"]
        failed = [r for r in results if r.get("analysis_status") == "error"]
        
        signal_counts = {"BULLISH": 0, "POSITIVE": 0, "NEUTRAL": 0, "BEARISH": 0, "UNKNOWN": 0}
        total_fscore = 0
        total_pe = 0
        total_momentum = 0
        valid_fscore_count = 0
        valid_pe_count = 0
        valid_momentum_count = 0
        
        for r in successful:
            analysis = r.get("analysis", "")
            signal = self._extract_ranking_signal(analysis)
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            quality = r.get("quality", {})
            valuation = r.get("valuation", {})
            momentum = r.get("momentum", {})
            
            fscore = quality.get("piotroski_fscore", 0)
            if fscore and isinstance(fscore, (int, float)):
                total_fscore += fscore
                valid_fscore_count += 1
            
            pe = valuation.get("pe_ratio", 0)
            if pe and isinstance(pe, (int, float)):
                total_pe += pe
                valid_pe_count += 1
            
            mom = momentum.get("return_3m", 0)
            if mom and isinstance(mom, (int, float)):
                total_momentum += mom
                valid_momentum_count += 1
        
        avg_fscore = total_fscore / valid_fscore_count if valid_fscore_count > 0 else 0
        avg_pe = total_pe / valid_pe_count if valid_pe_count > 0 else 0
        avg_momentum = total_momentum / valid_momentum_count if valid_momentum_count > 0 else 0
        
        allocation = self._calculate_portfolio_allocation(successful)
        
        top_picks = sorted(
            successful,
            key=lambda x: x.get("screening_score", 0),
            reverse=True
        )[:5]
        
        aggregation = {
            "total_stocks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": round(len(successful) / len(results) * 100, 1) if results else 0,
            
            "signal_distribution": signal_counts,
            "dominant_signal": max(signal_counts, key=signal_counts.get),
            
            "quality_metrics": {
                "avg_piotroski_fscore": round(avg_fscore, 2),
                "avg_pe_ratio": round(avg_pe, 2),
                "avg_3m_momentum": round(avg_momentum, 2),
            },
            
            "portfolio_allocation": allocation,
            "top_picks": [
                {
                    "symbol": p["symbol"],
                    "score": p.get("screening_score", 0),
                    "fscore": p.get("quality", {}).get("piotroski_fscore", 0),
                }
                for p in top_picks
            ],
            
            "market_summary": self._generate_market_summary(signal_counts, avg_fscore, avg_momentum),
        }
        
        return aggregation

    def _calculate_portfolio_allocation(self, successful_results: List[Dict]) -> Dict:
        """
        Calculate recommended portfolio allocation based on batch analysis.
        
        Uses composite scores to allocate capital across analyzed stocks.
        Higher allocation to higher-scoring stocks.
        """
        if not successful_results:
            return {"error": "No successful results"}
        
        total_score = sum(r.get("screening_score", 0) for r in successful_results)
        
        if total_score == 0:
            equal_weight = 100 / len(successful_results)
            return {
                "method": "equal_weight",
                "allocations": {
                    r["symbol"]: round(equal_weight, 2) for r in successful_results
                },
            }
        
        allocations = {}
        for r in successful_results:
            symbol = r["symbol"]
            score = r.get("screening_score", 0)
            weight = (score / total_score) * 100
            allocations[symbol] = round(weight, 2)
        
        return {
            "method": "score_weighted",
            "allocations": allocations,
            "max_single_allocation": max(allocations.values()),
            "min_single_allocation": min(allocations.values()),
        }

    def _generate_market_summary(
        self, 
        signal_counts: Dict, 
        avg_fscore: float, 
        avg_momentum: float
    ) -> str:
        """
        Generate a human-readable market summary based on aggregated metrics.
        """
        bullish_count = signal_counts.get("BULLISH", 0) + signal_counts.get("POSITIVE", 0)
        bearish_count = signal_counts.get("BEARISH", 0)
        total_signals = sum(signal_counts.values())
        
        if total_signals == 0:
            return "Insufficient data for market summary"
        
        bullish_pct = bullish_count / total_signals * 100
        bearish_pct = bearish_count / total_signals * 100
        
        if bullish_pct >= 60:
            bias = "BULLISH BIAS"
        elif bearish_pct >= 60:
            bias = "BEARISH BIAS"
        else:
            bias = "MIXED SIGNALS"
        
        if avg_fscore >= 6:
            quality = "HIGH QUALITY"
        elif avg_fscore >= 4:
            quality = "MODERATE QUALITY"
        else:
            quality = "LOW QUALITY"
        
        if avg_momentum >= 5:
            momentum = "STRONG MOMENTUM"
        elif avg_momentum >= 0:
            momentum = "STABLE MOMENTUM"
        else:
            momentum = "WEAK MOMENTUM"
        
        return f"{bias} | {quality} (Avg F-Score: {avg_fscore:.1f}) | {momentum} ({avg_momentum:.1f}%)"

    def _extract_conviction_from_text(self, text: str) -> int:
        """
        Phase 5.3.1: Extract conviction score (1-10) from CIO analysis text.
        
        The CIO agent produces detailed natural language analysis rather than structured
        markdown. This method uses keyword patterns to estimate conviction:
        - Strong buy indicators: 8-10
        - Buy/positive indicators: 6-7
        - Neutral/hold: 4-5
        - Negative/sell: 2-3
        - Strong sell: 1
        
        Args:
            text: Full CIO analysis text
            
        Returns:
            Conviction score 1-10, defaults to 5 (neutral)
            
        Example:
            text = "Strong buy recommended based on excellent fundamentals..."
            score = _extract_conviction_from_text(text)  # Returns 8-9
        """
        if not text:
            return 5
        
        text_lower = text.lower()
        
        strong_buy_indicators = [
            "strong buy", "highly recommended", "very bullish",
            "outperform", "strongly bullish", "excellent opportunity",
            "high conviction", "maximum conviction"
        ]
        buy_indicators = [
            "buy", "positive", "bullish", "favorable", "attractive",
            "good opportunity", "undervalued", "accumulate"
        ]
        neutral_indicators = [
            "neutral", "hold", "mixed", "cautious", "wait",
            "monitor", "under review", "no strong view"
        ]
        sell_indicators = [
            "sell", "negative", "bearish", "avoid", "overvalued",
            "reduce", "unfavorable", "underperform"
        ]
        strong_sell_indicators = [
            "strong sell", "strongly bearish", "very negative",
            "exit", "liquidate", "avoid at all costs"
        ]
        
        for phrase in strong_sell_indicators:
            if phrase in text_lower:
                return 2
        
        for phrase in sell_indicators:
            if phrase in text_lower:
                return 3
        
        for phrase in neutral_indicators:
            if phrase in text_lower:
                return 5
        
        for phrase in buy_indicators:
            if phrase in text_lower:
                return 7
        
        for phrase in strong_buy_indicators:
            if phrase in text_lower:
                return 9
        
        confidence_indicators = {
            "high confidence": 2,
            "confident": 1,
            "moderate confidence": 0,
            "low confidence": -1,
        }
        
        for phrase, adjustment in confidence_indicators.items():
            if phrase in text_lower:
                base = 5 + adjustment
                return max(1, min(10, base))
        
        return 5

    def _extract_verdict_from_text(self, text: str) -> str:
        """
        Phase 5.3.1: Extract structured verdict (BUY/HOLD/SELL) from CIO analysis text.
        
        Maps natural language to standardized verdict:
        - "buy", "accumulate", "outperform", "strong buy" -> BUY
        - "hold", "neutral", "monitor", "mixed" -> HOLD  
        - "sell", "reduce", "underperform", "exit" -> SELL
        
        Args:
            text: Full CIO analysis text
            
        Returns:
            Verdict: "BUY", "HOLD", "SELL", or "UNKNOWN"
            
        Example:
            text = "Recommend NEUTRAL position given mixed signals..."
            verdict = _extract_verdict_from_text(text)  # Returns "HOLD"
        """
        if not text:
            return "UNKNOWN"
        
        text_lower = text.lower()
        
        buy_patterns = [
            "recommend a buy", "recommend buy", "recommend buying",
            "buy position", "long position", "accumulate",
            "outperform", "overweight", "strong buy", "undervalued",
            "positive outlook", "bullish outlook", "good entry point"
        ]
        sell_patterns = [
            "recommend sell", "sell position", "short position",
            "underperform", "underweight", "strong sell",
            "negative outlook", "bearish outlook", "exit position",
            "reduce exposure", "avoid", "overvalued"
        ]
        hold_patterns = [
            "recommend hold", "hold position", "neutral position",
            "neutral", "mixed signals", "cautious", "monitor",
            "wait and see", "under review", "no strong view",
            "maintain current", "stay neutral"
        ]
        
        for pattern in buy_patterns:
            if pattern in text_lower:
                if "strong buy" in text_lower:
                    return "STRONG BUY"
                return "BUY"
        
        for pattern in sell_patterns:
            if pattern in text_lower:
                if "strong sell" in text_lower:
                    return "STRONG SELL"
                return "SELL"
        
        for pattern in hold_patterns:
            if pattern in text_lower:
                return "HOLD"
        
        if "buy" in text_lower:
            return "BUY"
        elif "sell" in text_lower:
            return "SELL"
        elif "neutral" in text_lower or "hold" in text_lower:
            return "HOLD"
        
        return "UNKNOWN"

    def _extract_thesis_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key investment thesis points from CIO analysis text.
        
        Identifies sentences containing investment rationale:
        - Growth indicators
        - Valuation indicators  
        - Risk factors
        - Catalysts
        
        Args:
            text: Full CIO analysis text
            max_keywords: Maximum number of thesis points to extract
            
        Returns:
            List of key thesis points/sentences
        """
        if not text:
            return []
        
        sentences = text.replace(".", ".\n").split("\n")
        thesis_sentences = []
        
        positive_keywords = [
            "growth", "profit", "revenue", "dividend", "roi", "roe",
            "undervalued", "bullish", "positive", "strong", "solid",
            "opportunity", "catalyst", "improving", "increasing"
        ]
        negative_keywords = [
            "risk", "debt", "loss", "decline", "weak", "concern",
            "overvalued", "bearish", "negative", "volatile", "uncertain"
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence = sentence.strip()
            
            if len(sentence) < 30:
                continue
            
            if any(kw in sentence_lower for kw in positive_keywords + negative_keywords):
                thesis_sentences.append(sentence)
        
        return thesis_sentences[:max_keywords]

    def extract_cio_signals(self, analysis_text: str) -> Dict:
        """
        Phase 5.3.1: Extract all signals from CIO analysis.
        
        Convenience method that extracts:
        - Verdict (BUY/HOLD/SELL)
        - Conviction score (1-10)
        - Thesis keywords
        - Signal category (BULLISH/BEARISH/NEUTRAL)
        
        Args:
            analysis_text: Full CIO analysis text
            
        Returns:
            Dict with extracted signals
            
        Example:
            result = extract_cio_signals(cio_analysis_text)
            print(result["verdict"])    # "BUY"
            print(result["conviction"])  # 7
            print(result["signal"])      # "BULLISH"
        """
        verdict = self._extract_verdict_from_text(analysis_text)
        conviction = self._extract_conviction_from_text(analysis_text)
        thesis = self._extract_thesis_keywords(analysis_text)
        
        verdict_to_signal = {
            "STRONG BUY": "BULLISH",
            "BUY": "BULLISH",
            "HOLD": "NEUTRAL",
            "SELL": "BEARISH",
            "STRONG SELL": "BEARISH",
            "UNKNOWN": "UNKNOWN"
        }
        
        return {
            "verdict": verdict,
            "conviction": conviction,
            "signal": verdict_to_signal.get(verdict, "UNKNOWN"),
            "thesis_keywords": thesis,
            "confidence": "HIGH" if conviction >= 7 or conviction <= 3 else "MEDIUM" if conviction != 5 else "LOW"
        }

    def analyze_single_stock(self, symbol: str, verbose: bool = False) -> Dict:
        """
        Run agent analysis on a single stock.

        Args:
            symbol: Stock ticker (e.g., "RELIANCE")
            verbose: Print detailed progress

        Returns:
            Dict with agent analysis results
        """
        if verbose:
            logger.info(f"Analyzing {symbol}")
        
        analysis = analyze_stock(symbol)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "agent_outputs": analysis.get("agent_outputs", {}),
            "vram_status": analysis.get("vram_status", {}),
        }

    def run_screener_only(
        self,
        universe: str = "NIFTY50",
        min_fscore: int = 4,
        max_pe: float = 40,
        min_momentum: float = -15,
        top_n: int = 5,
    ) -> Dict:
        """
        Run screening only without agent analysis.

        Args:
            universe: Stock universe to screen
            min_fscore: Minimum Piotroski F-Score
            max_pe: Maximum P/E ratio
            min_momentum: Minimum 3-month momentum (%)
            top_n: Number of stocks to return

        Returns:
            Dict with screening results
        """
        return stock_shortlister.run_screening(
            universe=universe,
            min_fscore=min_fscore,
            max_pe=max_pe,
            min_momentum=min_momentum,
            top_n=top_n,
            save_output=True,
        )

    def _rank_results(self, analyses: List[Dict]) -> List[Dict]:
        """
        Phase 5.3.2: Rank analyzed stocks by conviction and composite score.
        
        Ranking methodology:
        1. Primary: Verdict tier (BUY > HOLD > SELL)
        2. Secondary: Conviction score (1-10)
        3. Tertiary: Composite screening score
        
        This creates a multi-tier ranking where:
        - BUY stocks are ranked first, sorted by conviction
        - HOLD stocks follow, sorted by conviction
        - SELL stocks are ranked last (if any)
        
        Args:
            analyses: List of stock analyses from analyze_batch
            
        Returns:
            List of ranked stocks with tier, rank, and scores
            
        Example output:
            [
                {"symbol": "TCS", "tier": "BUY", "conviction": 8, "rank": 1, ...},
                {"symbol": "RELIANCE", "tier": "HOLD", "conviction": 7, "rank": 2, ...},
                {"symbol": "TATASTEEL", "tier": "SELL", "conviction": 3, "rank": 3, ...},
            ]
        """
        ranked = []

        for stock in analyses:
            if stock.get("analysis_status") == "success":
                score = stock.get("screening_score", 0)
                conviction = stock.get("conviction", 5)
                verdict = stock.get("verdict", "UNKNOWN")
                quality = stock.get("quality", {})
                momentum = stock.get("momentum", {})
                valuation = stock.get("valuation", {})
                
                ranking_signal = self._extract_ranking_signal(stock.get("analysis", ""))
                
                tier = self._verdict_to_tier(verdict)
                
                combined_rank_score = self._calculate_combined_rank_score(
                    tier=tier,
                    conviction=conviction,
                    screening_score=score
                )
                
                ranked.append({
                    "symbol": stock["symbol"],
                    "composite_score": score,
                    "conviction": conviction,
                    "verdict": verdict,
                    "tier": tier,
                    "fscore": quality.get("piotroski_fscore", 0),
                    "pe_ratio": valuation.get("pe_ratio", 0),
                    "momentum_3m": momentum.get("return_3m", 0),
                    "analysis_summary": ranking_signal,
                    "rank_score": combined_rank_score,
                    "thesis_keywords": stock.get("thesis_keywords", []),
                })

        ranked.sort(key=lambda x: x["rank_score"], reverse=True)
        
        for i, stock in enumerate(ranked, 1):
            stock["rank"] = i

        return ranked

    def _verdict_to_tier(self, verdict: str) -> str:
        """
        Convert verdict string to ranking tier.
        
        Tiers are used for primary sorting:
        - BUY: Recommended for purchase
        - HOLD: No strong action recommended
        - SELL: Recommended for exit/reduction
        
        Args:
            verdict: Verdict from extract_cio_signals (e.g., "BUY", "STRONG BUY", "HOLD")
            
        Returns:
            Tier string: "BUY", "HOLD", or "SELL"
        """
        verdict_upper = verdict.upper()
        
        if verdict_upper in ("STRONG BUY", "BUY"):
            return "BUY"
        elif verdict_upper in ("STRONG SELL", "SELL"):
            return "SELL"
        else:
            return "HOLD"

    def _calculate_combined_rank_score(
        self, 
        tier: str, 
        conviction: int, 
        screening_score: float
    ) -> float:
        """
        Calculate combined rank score for multi-criteria sorting.
        
        Formula:
        - BUY tier: 10000 + (conviction * 100) + screening_score
        - HOLD tier: 5000 + (conviction * 50) + screening_score
        - SELL tier: (conviction * 25) + screening_score
        
        This ensures:
        1. All BUY stocks rank above all HOLD stocks
        2. All HOLD stocks rank above all SELL stocks
        3. Within each tier, conviction score takes priority
        4. Screening score provides final tiebreaker
        
        Args:
            tier: Stock tier (BUY/HOLD/SELL)
            conviction: Conviction score (1-10)
            screening_score: Composite screening score
            
        Returns:
            Combined rank score (higher = better rank)
        """
        tier_multipliers = {
            "BUY": 10000 + (conviction * 100),
            "HOLD": 5000 + (conviction * 50),
            "SELL": conviction * 25,
        }
        
        base_score = tier_multipliers.get(tier, 5000)
        
        return base_score + screening_score

    def get_ranked_summary(self, ranked_results: List[Dict]) -> Dict:
        """
        Generate a summary from ranked results.
        
        Provides an at-a-glance summary of ranked stocks:
        - Top picks by tier
        - Portfolio recommendation
        - Risk summary
        
        Args:
            ranked_results: Output from _rank_results()
            
        Returns:
            Dict with summary statistics
        """
        if not ranked_results:
            return {"error": "No ranked results"}
        
        buy_stocks = [s for s in ranked_results if s.get("tier") == "BUY"]
        hold_stocks = [s for s in ranked_results if s.get("tier") == "HOLD"]
        sell_stocks = [s for s in ranked_results if s.get("tier") == "SELL"]
        
        return {
            "total_stocks": len(ranked_results),
            "tier_counts": {
                "buy": len(buy_stocks),
                "hold": len(hold_stocks),
                "sell": len(sell_stocks),
            },
            "top_buy": [
                {"symbol": s["symbol"], "conviction": s["conviction"]} 
                for s in buy_stocks[:3]
            ] if buy_stocks else [],
            "top_hold": [
                {"symbol": s["symbol"], "conviction": s["conviction"]} 
                for s in hold_stocks[:3]
            ] if hold_stocks else [],
            "portfolio_candidates": [s["symbol"] for s in ranked_results if s.get("tier") == "BUY"],
            "avoid_stocks": [s["symbol"] for s in ranked_results if s.get("tier") == "SELL"],
        }

    def _extract_ranking_signal(self, analysis_text: str) -> str:
        """
        Extract a short summary signal from analysis text.

        Args:
            analysis_text: Full analysis text from CIO

        Returns:
            Short signal string
        """
        if not analysis_text:
            return "No analysis"
        
        text_lower = analysis_text.lower()
        
        if any(word in text_lower for word in ["strong buy", "bullish", "outperform"]):
            return "BULLISH"
        elif any(word in text_lower for word in ["buy", "positive"]):
            return "POSITIVE"
        elif any(word in text_lower for word in ["neutral", "hold"]):
            return "NEUTRAL"
        elif any(word in text_lower for word in ["sell", "negative", "bearish"]):
            return "BEARISH"
        elif any(word in text_lower for word in ["strong sell", "underperform"]):
            return "BEARISH"
        
        return "UNKNOWN"

    def _save_pipeline_output(self, output: Dict) -> Path:
        """Save full pipeline output."""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_output_{timestamp}.json"
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved pipeline output to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pipeline output: {e}")

        return filepath


orchestrator = PipelineOrchestrator()


def run_pipeline(
    universe: str = "NIFTY50",
    min_fscore: int = 4,
    max_pe: float = 40,
    min_momentum: float = -15,
    top_n: int = 5,
) -> Dict:
    """
    Convenience function to run the full pipeline.

    Usage:
        result = run_pipeline(top_n=5)
    """
    return orchestrator.run_full_pipeline(
        universe=universe,
        min_fscore=min_fscore,
        max_pe=max_pe,
        min_momentum=min_momentum,
        top_n=top_n,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trading Pipeline Orchestrator")
    parser.add_argument("--stock", type=str, help="Analyze single stock")
    parser.add_argument("--screener", action="store_true", help="Run screener only")
    parser.add_argument("--batch", action="store_true", help="Run batch analysis (use with --stocks)")
    parser.add_argument("--stocks", type=str, help="Comma-separated list of stocks for batch mode")
    parser.add_argument("--universe", type=str, default="NIFTY50", help="Stock universe")
    parser.add_argument("--top-n", type=int, default=5, help="Number of stocks")
    parser.add_argument("--min-fscore", type=int, default=4, help="Min Piotroski F-Score")
    parser.add_argument("--max-pe", type=float, default=40, help="Max P/E ratio")
    parser.add_argument("--min-momentum", type=float, default=-15, help="Min 3M momentum (%%)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.stock:
        result = orchestrator.analyze_single_stock(args.stock, verbose=args.verbose)
        print(f"\n=== Analysis for {args.stock} ===")
        print(f"Status: {result['agent_outputs'].get('cio', {}).get('status', 'unknown')}")
    elif args.batch and args.stocks:
        stocks = [{"symbol": s.strip()} for s in args.stocks.split(",")]
        print(f"\n=== Batch Analysis: {len(stocks)} stocks ===")
        result = orchestrator.analyze_batch(stocks, verbose=True)
        print(f"\n=== Summary ===")
        print(f"Total: {result['summary']['total_stocks']}")
        print(f"Successful: {result['summary']['successful']}")
        print(f"Failed: {result['summary']['failed']}")
        print(f"Duration: {result['summary']['total_duration_seconds']}s")
    elif args.screener:
        result = orchestrator.run_screener_only(
            universe=args.universe,
            min_fscore=args.min_fscore,
            max_pe=args.max_pe,
            min_momentum=args.min_momentum,
            top_n=args.top_n,
        )
        print(f"\n=== Screening Results ===")
        print(f"Found: {result['stocks_count']} stocks")
        for i, stock in enumerate(result["stocks"], 1):
            print(f"{i}. {stock['symbol']} - Score: {stock['composite_score']:.1f}")
    else:
        result = orchestrator.run_full_pipeline(
            universe=args.universe,
            min_fscore=args.min_fscore,
            max_pe=args.max_pe,
            min_momentum=args.min_momentum,
            top_n=args.top_n,
            verbose=args.verbose,
        )
        print(f"\n=== Pipeline Complete ===")
        print(f"Duration: {result['pipeline_duration_seconds']:.1f}s")
        print(f"Stocks analyzed: {len(result['analyses'])}")
        print(f"\n=== Ranked Results ===")
        for stock in result.get("ranked_results", []):
            print(f"{stock['rank']}. {stock['symbol']} - Score: {stock['composite_score']:.1f} - {stock['analysis_summary']}")

    def get_cache_status(self) -> Dict:
        """
        Phase 5.4.2: Get current cache status for all data sources.
        
        Returns status of:
        - Fundamental data cache (24h TTL)
        - Market data cache
        - Universe cache (7 day TTL)
        
        Returns:
            Dict with cache status for each data source
        """
        from src.data.fundamental import _get_cache
        
        cache_dir = Path(settings.data_dir)
        status = {
            "timestamp": datetime.now().isoformat(),
            "fundamental_cache": {},
            "market_data_cache": {},
            "universe_cache": {},
        }
        
        fundamental_cache_file = cache_dir / "fundamental_cache.json"
        if fundamental_cache_file.exists():
            try:
                with open(fundamental_cache_file, "r") as f:
                    cache = json.load(f)
                
                entries = {}
                for symbol, entry in cache.items():
                    cached_time = datetime.fromisoformat(entry.get("timestamp", "2000-01-01"))
                    hours_old = (datetime.now() - cached_time).total_seconds() / 3600
                    is_fresh = hours_old < 24
                    entries[symbol] = {
                        "cached_at": entry.get("timestamp"),
                        "hours_old": round(hours_old, 1),
                        "is_fresh": is_fresh,
                    }
                
                status["fundamental_cache"] = {
                    "file": str(fundamental_cache_file),
                    "entries": len(entries),
                    "fresh_entries": sum(1 for e in entries.values() if e["is_fresh"]),
                    "stocks": entries,
                }
            except Exception as e:
                status["fundamental_cache"] = {"error": str(e)}
        else:
            status["fundamental_cache"] = {"exists": False}
        
        universe_cache_file = cache_dir / "nifty500_universe.csv"
        if universe_cache_file.exists():
            mtime = datetime.fromtimestamp(universe_cache_file.stat().st_mtime)
            age_days = (datetime.now() - mtime).total_seconds() / 86400
            status["universe_cache"] = {
                "file": str(universe_cache_file),
                "last_modified": mtime.isoformat(),
                "age_days": round(age_days, 1),
                "is_fresh": age_days < 7,
            }
        else:
            status["universe_cache"] = {"exists": False}
        
        status["market_data_cache"] = {
            "note": "yfinance handles its own caching",
            "refresh_on_each_run": True,
        }
        
        return status

    def clear_cache(self, cache_type: str = "all") -> Dict:
        """
        Phase 5.4.2: Clear cache for specific data source.
        
        Args:
            cache_type: Type of cache to clear
                - "fundamental": Clear fundamental data cache
                - "universe": Clear universe cache
                - "all": Clear all caches
        
        Returns:
            Dict with results of cache clearing
        """
        cache_dir = Path(settings.data_dir)
        results = {"cleared": [], "errors": []}
        
        if cache_type in ("fundamental", "all"):
            fund_cache = cache_dir / "fundamental_cache.json"
            if fund_cache.exists():
                try:
                    fund_cache.unlink()
                    results["cleared"].append("fundamental_cache")
                    logger.info("Cleared fundamental cache")
                except Exception as e:
                    results["errors"].append(f"fundamental_cache: {str(e)}")
        
        if cache_type in ("universe", "all"):
            uni_cache = cache_dir / "nifty500_universe.csv"
            if uni_cache.exists():
                try:
                    uni_cache.unlink()
                    results["cleared"].append("universe_cache")
                    logger.info("Cleared universe cache")
                except Exception as e:
                    results["errors"].append(f"universe_cache: {str(e)}")
        
        if cache_type == "all":
            for pattern in ["*.csv", "*.json"]:
                for f in cache_dir.glob(pattern):
                    if f.name not in results["cleared"]:
                        try:
                            if "cache" in f.name.lower() or "universe" in f.name.lower():
                                f.unlink()
                                results["cleared"].append(f.name)
                        except:
                            pass
        
        return results

    def get_data_freshness_report(self) -> str:
        """
        Phase 5.4.2: Generate a human-readable data freshness report.
        
        Returns:
            String with data freshness summary
        """
        status = self.get_cache_status()
        
        lines = [
            "=" * 50,
            "DATA FRESHNESS REPORT",
            "=" * 50,
            f"Generated: {status['timestamp']}",
            "",
        ]
        
        fund = status.get("fundamental_cache", {})
        if "error" not in fund:
            lines.append("FUNDAMENTAL DATA CACHE:")
            lines.append(f"  File: {fund.get('file', 'N/A')}")
            lines.append(f"  Total entries: {fund.get('entries', 0)}")
            lines.append(f"  Fresh entries (<24h): {fund.get('fresh_entries', 0)}")
            
            if fund.get("stocks"):
                lines.append("  Stock cache ages:")
                for symbol, info in list(fund["stocks"].items())[:5]:
                    fresh = "✓" if info["is_fresh"] else "✗"
                    lines.append(f"    {fresh} {symbol}: {info['hours_old']}h old")
                if len(fund["stocks"]) > 5:
                    lines.append(f"    ... and {len(fund['stocks']) - 5} more")
            lines.append("")
        
        uni = status.get("universe_cache", {})
        if "exists" in uni and uni["exists"] is False:
            lines.append("UNIVERSE CACHE: Not cached")
        else:
            fresh = "✓" if uni.get("is_fresh") else "✗"
            lines.append("UNIVERSE CACHE:")
            lines.append(f"  {fresh} Last updated: {uni.get('last_modified', 'N/A')}")
            lines.append(f"    Age: {uni.get('age_days', 0)} days")
        
        lines.append("")
        lines.append("CACHE TTL SETTINGS:")
        lines.append("  Fundamental data: 24 hours")
        lines.append("  Universe data: 7 days")
        lines.append("  Market data: Refreshed on each run")
        lines.append("")
        lines.append("TO FORCE REFRESH:")
        lines.append("  orchestrator.clear_cache('all')")
        lines.append("=" * 50)
        
        return "\n".join(lines)

    def benchmark_performance(self, stocks: List[Dict] = None) -> Dict:
        """
        Phase 5.4.3: Performance benchmarking for the pipeline.
        
        Measures and reports:
        - Per-stock timing
        - Per-agent timing (if detailed=True)
        - Estimated times for batch sizes
        - Total pipeline throughput
        
        Args:
            stocks: Optional list of stocks to benchmark. Uses RELIANCE if None.
            
        Returns:
            Dict with performance metrics
        """
        import time
        import json
        
        if stocks is None:
            stocks = [{"symbol": "RELIANCE", "composite_score": 50}]
        
        print("=" * 60)
        print("PERFORMANCE BENCHMARK")
        print("=" * 60)
        print()
        
        batch_result = self.analyze_batch(stocks, verbose=False)
        summary = batch_result["summary"]
        
        print()
        print("RESULTS:")
        print("-" * 40)
        print(f"Total Duration: {summary['total_duration_seconds']:.1f}s")
        print(f"Stocks Analyzed: {summary['successful']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Avg Time/Stock: {summary['avg_time_per_stock']:.1f}s")
        print(f"Min Time: {summary['min_time']:.1f}s")
        print(f"Max Time: {summary['max_time']:.1f}s")
        print()
        
        print("Per-Stock Breakdown:")
        for r in batch_result["results"]:
            print(f"  {r['symbol']}: {r['duration_seconds']}s")
        print()
        
        avg_time = summary["avg_time_per_stock"]
        print("Estimated Pipeline Times:")
        print("-" * 40)
        for n in [1, 3, 5, 10, 20, 50]:
            total = n * avg_time
            if total < 60:
                print(f"  {n:2d} stocks: ~{total:.0f}s")
            else:
                print(f"  {n:2d} stocks: ~{total/60:.0f}min ({total/3600:.1f}h)")
        
        print()
        print("=" * 60)
        
        return {
            "benchmark_timestamp": datetime.now().isoformat(),
            "stocks_tested": len(stocks),
            "total_duration": summary["total_duration_seconds"],
            "avg_per_stock": summary["avg_time_per_stock"],
            "min_per_stock": summary["min_time"],
            "max_per_stock": summary["max_time"],
            "success_rate": summary["success_rate"],
            "estimated_times": {
                n: round(n * avg_time, 1) for n in [1, 3, 5, 10, 20, 50]
            },
        }

    def get_pipeline_stats(self) -> Dict:
        """
        Get overall pipeline statistics from all batch runs.
        
        Returns aggregated statistics across all runs.
        """
        import json
        from pathlib import Path
        
        batch_files = list(self.output_dir.glob("batch_output_*.json"))
        
        if not batch_files:
            return {"error": "No batch results found"}
        
        all_results = []
        for f in batch_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    all_results.append(data)
            except:
                continue
        
        if not all_results:
            return {"error": "No valid batch results"}
        
        total_stocks = sum(r["summary"]["total_stocks"] for r in all_results)
        total_time = sum(r["summary"]["total_duration_seconds"] for r in all_results)
        total_successful = sum(r["summary"]["successful"] for r in all_results)
        
        return {
            "total_batch_runs": len(all_results),
            "total_stocks_analyzed": total_stocks,
            "total_time_seconds": round(total_time, 1),
            "total_time_minutes": round(total_time / 60, 1),
            "total_successful": total_successful,
            "overall_success_rate": round(total_successful / total_stocks * 100, 1) if total_stocks > 0 else 0,
            "avg_time_per_stock": round(total_time / total_stocks, 1) if total_stocks > 0 else 0,
        }