"""Test CIO agent verdict extraction with new format."""

import pytest
from src.agents.cio import CIOAgent


class TestCIOVerdictExtraction:
    """Test verdict and conviction extraction."""

    def setup_method(self):
        self.agent = CIOAgent()

    def test_extract_strong_buy(self):
        response = """
        Some analysis text here.
        
        VERDICT: Strong Buy
        CONVICTION: 9
        """
        verdict = self.agent._extract_verdict(response)
        assert verdict == "Strong Buy"

    def test_extract_buy(self):
        response = """
        Some analysis text here.
        
        VERDICT: Buy
        CONVICTION: 7
        """
        verdict = self.agent._extract_verdict(response)
        assert verdict == "Buy"

    def test_extract_hold(self):
        response = """
        Some analysis text here.
        
        VERDICT: Hold
        CONVICTION: 5
        """
        verdict = self.agent._extract_verdict(response)
        assert verdict == "Hold"

    def test_extract_sell(self):
        response = """
        Some analysis text here.
        
        VERDICT: Sell
        CONVICTION: 3
        """
        verdict = self.agent._extract_verdict(response)
        assert verdict == "Sell"

    def test_extract_conviction(self):
        response = """
        Some analysis text here.
        
        VERDICT: Strong Buy
        CONVICTION: 8
        """
        conviction = self.agent._extract_conviction(response)
        assert conviction == 8

    def test_extract_conviction_from_middle(self):
        response = """
        VERDICT: Buy
        CONVICTION: 7
        
        Some other text here.
        """
        conviction = self.agent._extract_conviction(response)
        assert conviction == 7

    def test_extract_from_full_analysis(self):
        response = """
        DETAILED ANALYSIS:
        The stock shows strong fundamentals with improving technicals.
        
        VERDICT: Strong Buy
        CONVICTION: 9
        
        THESIS:
        Strong buy based on multiple factors.
        
        VERDICT: Strong Buy
        CONVICTION: 9
        """
        verdict = self.agent._extract_verdict(response)
        conviction = self.agent._extract_conviction(response)
        
        assert verdict == "Strong Buy"
        assert conviction == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
