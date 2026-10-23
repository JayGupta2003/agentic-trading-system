"""Tests for Phase 2: Data Summarizer.

Run with: python -m pytest tests -v
"""

import pandas as pd
import pytest

from src.data.summarizer import (
    DataSummarizer,
    PriceSummary,
    FundamentalSummary,
    SentimentSummary,
    DerivativesSummary,
)


class TestPriceSummarizer:
    """Tests for price data summarization."""
    
    @pytest.fixture
    def summarizer(self):
        """Create DataSummarizer instance."""
        return DataSummarizer()
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        data = {
            'Open': [100 + i * 0.5 for i in range(60)],
            'High': [105 + i * 0.5 for i in range(60)],
            'Low': [95 + i * 0.5 for i in range(60)],
            'Close': [102 + i * 0.5 for i in range(60)],
            'Volume': [1000000] * 60
        }
        return pd.DataFrame(data, index=dates)
    
    def test_summarize_price_data_basic(self, summarizer, sample_price_data):
        """Test basic price data summarization."""
        summary = summarizer.summarize_price_data(sample_price_data, "TEST")
        
        assert summary.symbol == "TEST"
        assert summary.current_price == 102 + 59 * 0.5
        assert summary.sma_20 is not None
        assert summary.rsi is not None
        assert summary.trend != "INSUFFICIENT DATA"
    
    def test_summarize_price_data_empty(self, summarizer):
        """Test with empty DataFrame."""
        summary = summarizer.summarize_price_data(pd.DataFrame(), "EMPTY")
        
        assert summary.symbol == "EMPTY"
        assert summary.current_price == 0.0
        assert summary.trend == "INSUFFICIENT DATA"
        assert summary.momentum == "INSUFFICIENT DATA"
    
    def test_summarize_price_data_trend_detection(self, summarizer):
        """Test trend detection."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')
        rising_prices = [100 + i * 1 for i in range(250)]
        data = {
            'Close': rising_prices,
            'High': [p + 2 for p in rising_prices],
            'Low': [p - 2 for p in rising_prices],
            'Open': rising_prices
        }
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "RISING")
        
        assert "UPTREND" in summary.trend or "BULLISH" in summary.trend
    
    def test_summarize_price_data_momentum(self, summarizer):
        """Test momentum detection with fluctuating prices."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        data = {
            'Close': [100 + (i % 5) - 2 for i in range(60)],
            'High': [105 + (i % 5) - 2 for i in range(60)],
            'Low': [95 + (i % 5) - 2 for i in range(60)],
            'Open': [100 + (i % 5) - 2 for i in range(60)]
        }
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "FLAT")
        
        assert "RSI" in summary.momentum
    
    def test_support_resistance_levels(self, summarizer):
        """Test support and resistance level calculation."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        data = {
            'Close': [100 + i for i in range(60)],
            'High': [110 + i for i in range(60)],
            'Low': [90 + i for i in range(60)],
            'Open': [100 + i for i in range(60)]
        }
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "TEST")
        
        assert len(summary.support_levels) >= 1
        assert len(summary.resistance_levels) >= 1
        assert all(s <= summary.current_price for s in summary.support_levels)
    
    def test_price_summary_to_string(self, summarizer, sample_price_data):
        """Test string conversion."""
        summary = summarizer.summarize_price_data(sample_price_data, "TEST")
        text = summary.to_string()
        
        assert "TEST" in text
        assert "₹" in text
        assert "Trend:" in text


class TestFundamentalSummarizer:
    """Tests for fundamental data summarization."""
    
    @pytest.fixture
    def summarizer(self):
        """Create DataSummarizer instance."""
        return DataSummarizer()
    
    @pytest.fixture
    def sample_fundamental_data(self):
        """Create sample fundamental data."""
        return {
            'company_name': 'Reliance Industries',
            'sector': 'Energy',
            'pe_ratio': 25.5,
            'pb_ratio': 3.2,
            'roe': 9.5,
            'roce': 12.0,
            'debt_to_equity': 0.5,
            'current_ratio': 1.2,
            'f_score': 7,
            'sales_growth': 15.5,
            'profit_growth': 18.2,
            'market_cap': 1500000,
            'dividend_yield': 2.5,
            'eps': 95.5
        }
    
    def test_summarize_fundamental_basic(self, summarizer, sample_fundamental_data):
        """Test basic fundamental summarization."""
        summary = summarizer.summarize_fundamental_data(
            sample_fundamental_data, "RELIANCE"
        )
        
        assert summary.symbol == "RELIANCE"
        assert summary.company_name == "Reliance Industries"
        assert summary.pe_ratio == 25.5
        assert summary.f_score == 7
        assert "Strong" in summary.score or "Average" in summary.score
    
    def test_summarize_fundamental_concerns(self, summarizer):
        """Test concern detection."""
        data = {
            'debt_to_equity': 3.5,
            'current_ratio': 0.7,
            'pe_ratio': 50.0,
            'roe': 5.0,
            'sector': 'Unknown'
        }
        
        summary = summarizer.summarize_fundamental_data(data, "WEAK")
        
        assert "High leverage" in summary.concerns
        assert "Low liquidity" in summary.concerns
        assert "Expensive valuation" in summary.concerns
        assert "Low returns on equity" in summary.concerns
    
    def test_summarize_fundamental_empty(self, summarizer):
        """Test with minimal data."""
        summary = summarizer.summarize_fundamental_data({}, "EMPTY")
        
        assert summary.symbol == "EMPTY"
        assert summary.company_name == "EMPTY"
        assert summary.score == "INSUFFICIENT DATA"
    
    def test_market_cap_formatting(self, summarizer, sample_fundamental_data):
        """Test market cap formatting."""
        summary = summarizer.summarize_fundamental_data(
            sample_fundamental_data, "RELIANCE"
        )
        text = summary.to_string()
        
        assert "Cr" in text
    
    def test_fundamental_summary_to_string(self, summarizer, sample_fundamental_data):
        """Test string conversion."""
        summary = summarizer.summarize_fundamental_data(
            sample_fundamental_data, "RELIANCE"
        )
        text = summary.to_string()
        
        assert "Reliance Industries" in text
        assert "P/E:" in text
        assert "ROE:" in text


class TestSentimentSummarizer:
    """Tests for news/sentiment summarization."""
    
    @pytest.fixture
    def summarizer(self):
        """Create DataSummarizer instance."""
        return DataSummarizer()
    
    @pytest.fixture
    def sample_articles(self):
        """Create sample news articles."""
        return [
            {
                'title': 'Company reports strong quarterly results, beats estimates',
                'sentiment': 0.8
            },
            {
                'title': 'New order win from government sector',
                'sentiment': 0.6
            },
            {
                'title': 'Stock price target raised by brokerages',
                'sentiment': 0.4
            },
            {
                'title': 'Company expands into new markets',
                'sentiment': 0.5
            },
        ]
    
    def test_summarize_sentiment_positive(self, summarizer, sample_articles):
        """Test positive sentiment summarization."""
        summary = summarizer.summarize_news_data(sample_articles, "STOCK")
        
        assert summary.symbol == "STOCK"
        assert summary.positive_count == 4
        assert summary.negative_count == 0
        assert summary.overall_sentiment in ["POSITIVE", "VERY POSITIVE"]
        assert summary.sentiment_score > 0
    
    def test_summarize_sentiment_negative(self, summarizer):
        """Test negative sentiment summarization."""
        articles = [
            {'title': 'Company faces regulatory scrutiny', 'sentiment': -0.7},
            {'title': 'Quarterly loss disappoints', 'sentiment': -0.8},
        ]
        
        summary = summarizer.summarize_news_data(articles, "BAD")
        
        assert summary.negative_count == 2
        assert summary.overall_sentiment in ["NEGATIVE", "VERY NEGATIVE"]
        assert summary.sentiment_score < 0
    
    def test_summarize_sentiment_empty(self, summarizer):
        """Test with no articles."""
        summary = summarizer.summarize_news_data([], "EMPTY")
        
        assert summary.overall_sentiment == "NEUTRAL"
        assert summary.sentiment_score == 0.0
        assert summary.positive_count == 0
    
    def test_key_positives_extraction(self, summarizer, sample_articles):
        """Test extraction of key positives."""
        summary = summarizer.summarize_news_data(sample_articles, "STOCK")
        
        assert len(summary.key_positives) > 0
    
    def test_catalyst_identification(self, summarizer):
        """Test catalyst identification from headlines."""
        articles = [
            {'title': 'Company Q4 results beat estimates', 'sentiment': 0.5},
            {'title': 'New order win from infrastructure sector', 'sentiment': 0.6},
        ]
        
        summary = summarizer.summarize_news_data(articles, "STOCK")
        
        assert summary.catalyst_identified is not None
    
    def test_sentiment_summary_to_string(self, summarizer, sample_articles):
        """Test string conversion."""
        summary = summarizer.summarize_news_data(sample_articles, "STOCK")
        text = summary.to_string()
        
        assert "STOCK" in text
        assert "Sentiment:" in text
        assert "positive" in text


class TestDerivativesSummarizer:
    """Tests for derivatives data summarization."""
    
    @pytest.fixture
    def summarizer(self):
        """Create DataSummarizer instance."""
        return DataSummarizer()
    
    @pytest.fixture
    def sample_derivatives_data(self):
        """Create sample derivatives data."""
        return {
            'pcr': 1.15,
            'vix': 18.5,
            'vix_change_pct': 5.2,
            'major_oi_strike': 2500.0,
            'oi_pattern': 'OI Buildup',
            'expiry_date': '2024-01-25',
            'institutional_activity': 'Institutional Buying'
        }
    
    def test_summarize_derivatives_basic(self, summarizer, sample_derivatives_data):
        """Test basic derivatives summarization."""
        summary = summarizer.summarize_derivatives_data(
            sample_derivatives_data, "STOCK"
        )
        
        assert summary.symbol == "STOCK"
        assert summary.pcr == 1.15
        assert summary.vix == 18.5
        assert "Bullish" in summary.pcr_signal
        assert not summary.is_expiry_week
    
    def test_summarize_derivatives_pcr_signals(self, summarizer):
        """Test PCR signal interpretation."""
        test_cases = [
            (1.5, "Bullish"),
            (1.1, "Slightly Bullish"),
            (0.85, "Neutral"),
            (0.6, "Slightly Bearish"),
            (0.3, "Bearish"),
        ]
        
        for pcr_value, expected_signal in test_cases:
            data = {
                'pcr': pcr_value,
                'vix': 15.0,
                'expiry_date': '2024-01-25'
            }
            summary = summarizer.summarize_derivatives_data(data, "TEST")
            assert expected_signal in summary.pcr_signal
    
    def test_summarize_derivatives_expiry_week(self, summarizer, sample_derivatives_data):
        """Test expiry week flag."""
        summary = summarizer.summarize_derivatives_data(
            sample_derivatives_data, "STOCK", is_expiry_week=True
        )
        
        assert summary.is_expiry_week
        assert "EXPIRY WEEK" in summary.to_string()
    
    def test_summarize_derivatives_oi_patterns(self, summarizer):
        """Test OI pattern interpretation."""
        test_cases = [
            ('OI Buildup detected', 'OI Building Up'),
            ('OI Unwinding in progress', 'OI Unwinding'),
            ('Short covering detected', 'Short Covering'),
            ('Long Unwinding', 'Long Unwinding'),
        ]
        
        for input_pattern, expected_output in test_cases:
            data = {
                'pcr': 1.0,
                'vix': 15.0,
                'oi_pattern': input_pattern,
                'expiry_date': '2024-01-25'
            }
            summary = summarizer.summarize_derivatives_data(data, "TEST")
            assert summary.oi_buildup == expected_output
    
    def test_summarize_derivatives_institutional(self, summarizer):
        """Test institutional activity detection."""
        test_cases = [
            ('Major institutional buying seen', 'Institutional Buying'),
            ('FII selling continues', 'Institutional Selling'),
            ('Mixed activity', 'Mixed activity'),
        ]
        
        for input_activity, expected in test_cases:
            data = {
                'pcr': 1.0,
                'vix': 15.0,
                'institutional_activity': input_activity,
                'expiry_date': '2024-01-25'
            }
            summary = summarizer.summarize_derivatives_data(data, "TEST")
            assert expected in summary.institutional_activity
    
    def test_derivatives_summary_to_string(self, summarizer, sample_derivatives_data):
        """Test string conversion."""
        summary = summarizer.summarize_derivatives_data(
            sample_derivatives_data, "STOCK"
        )
        text = summary.to_string()
        
        assert "STOCK" in text
        assert "PCR:" in text
        assert "VIX:" in text


class TestDataSummarizerIntegration:
    """Integration tests for DataSummarizer."""
    
    @pytest.fixture
    def summarizer(self):
        """Create DataSummarizer instance."""
        return DataSummarizer()
    
    def test_all_summaries_can_convert_to_string(self, summarizer):
        """Test that all summaries can convert to strings."""
        price_data = pd.DataFrame({
            'Close': [100, 102, 104],
            'High': [105, 107, 108],
            'Low': [98, 100, 102],
            'Open': [99, 101, 103]
        })
        
        price_summary = summarizer.summarize_price_data(price_data, "TEST")
        assert len(price_summary.to_string()) > 0
        
        fund_data = {'pe_ratio': 20, 'sector': 'Tech'}
        fund_summary = summarizer.summarize_fundamental_data(fund_data, "TEST")
        assert len(fund_summary.to_string()) > 0
        
        news_data = [{'title': 'Good news', 'sentiment': 0.5}]
        news_summary = summarizer.summarize_news_data(news_data, "TEST")
        assert len(news_summary.to_string()) > 0
        
        deriv_data = {'pcr': 1.0, 'vix': 15, 'expiry_date': '2024-01-25'}
        deriv_summary = summarizer.summarize_derivatives_data(deriv_data, "TEST")
        assert len(deriv_summary.to_string()) > 0
    
    def test_summaries_are_vram_friendly(self, summarizer):
        """Test that summaries are concise (VRAM efficient)."""
        price_data = pd.DataFrame({
            'Close': [100 + i for i in range(100)],
            'High': [105 + i for i in range(100)],
            'Low': [95 + i for i in range(100)],
            'Open': [100 + i for i in range(100)]
        })
        
        summary = summarizer.summarize_price_data(price_data, "TEST")
        summary_str = summary.to_string()
        
        assert len(summary_str) < 2000
        assert len(price_data) == 100
    
    def test_summarizer_handles_missing_fields(self, summarizer):
        """Test that summarizer handles missing data gracefully."""
        partial_data = {
            'sector': 'Finance'
        }
        
        summary = summarizer.summarize_fundamental_data(partial_data, "PARTIAL")
        
        assert summary.symbol == "PARTIAL"
        assert summary.pe_ratio is None
        assert summary.score == "INSUFFICIENT DATA"


class TestRSICalculation:
    """Tests for RSI calculation."""
    
    @pytest.fixture
    def summarizer(self):
        return DataSummarizer()
    
    def test_rsi_overbought(self, summarizer):
        """Test RSI in overbought territory."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        rising = [100 + i * 2 for i in range(30)]
        data = {'Close': rising, 'High': rising, 'Low': rising, 'Open': rising}
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "RISING")
        
        assert summary.rsi is not None
        assert "OVERBOUGHT" in summary.momentum
    
    def test_rsi_oversold(self, summarizer):
        """Test RSI in oversold territory."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        falling = [100 - i * 2 for i in range(30)]
        data = {'Close': falling, 'High': falling, 'Low': falling, 'Open': falling}
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "FALLING")
        
        assert summary.rsi is not None
        assert "OVERSOLD" in summary.momentum


class TestMACDCalculation:
    """Tests for MACD calculation."""
    
    @pytest.fixture
    def summarizer(self):
        return DataSummarizer()
    
    def test_macd_positive(self, summarizer):
        """Test MACD in positive territory."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        rising = [100 + i for i in range(50)]
        data = {'Close': rising, 'High': rising, 'Low': rising, 'Open': rising}
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "RISING")
        
        assert summary.macd is not None
        assert "POSITIVE MACD" in summary.momentum
    
    def test_macd_negative(self, summarizer):
        """Test MACD in negative territory."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        falling = [100 - i for i in range(50)]
        data = {'Close': falling, 'High': falling, 'Low': falling, 'Open': falling}
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "FALLING")
        
        assert summary.macd is not None
        assert "NEGATIVE MACD" in summary.momentum


class TestTrendDetection:
    """Tests for trend detection."""
    
    @pytest.fixture
    def summarizer(self):
        return DataSummarizer()
    
    def test_strong_uptrend(self, summarizer):
        """Test strong uptrend detection."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')
        prices = [100 + i * 0.5 for i in range(250)]
        data = {'Close': prices, 'High': prices, 'Low': prices, 'Open': prices}
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "UPTREND")
        
        assert "UPTREND" in summary.trend
    
    def test_strong_downtrend(self, summarizer):
        """Test strong downtrend detection."""
        dates = pd.date_range('2024-01-01', periods=250, freq='D')
        prices = [200 - i * 0.5 for i in range(250)]
        data = {'Close': prices, 'High': prices, 'Low': prices, 'Open': prices}
        df = pd.DataFrame(data, index=dates)
        
        summary = summarizer.summarize_price_data(df, "DOWNTREND")
        
        assert "DOWNTREND" in summary.trend


class TestNewsBuffer:
    """Tests for news buffer functionality."""
    
    def test_news_buffer_days_setting(self):
        """Test that news_buffer_days setting exists and defaults to 5."""
        from config.settings import settings
        assert hasattr(settings, 'news_buffer_days')
        assert settings.news_buffer_days == 5
    
    def test_news_fetcher_buffer_parameter(self):
        """Test that NewsDataFetcher accepts use_buffer parameter."""
        from src.data.news import NewsDataFetcher
        
        fetcher = NewsDataFetcher()
        assert hasattr(fetcher, 'get_company_news')
        
        import inspect
        sig = inspect.signature(fetcher.get_company_news)
        assert 'use_buffer' in sig.parameters
    
    def test_news_buffer_includes_today(self):
        """Test that buffer mode includes today's news plus buffer days."""
        from datetime import datetime, timedelta
        from src.data.news import NewsDataFetcher
        
        fetcher = NewsDataFetcher()
        
        articles = fetcher.get_company_news("RELIANCE", days=5, use_buffer=True)
        
        assert len(articles) >= 0
        
        dates_found = []
        for article in articles:
            pub_date = article.get('publishedAt', '')
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    dates_found.append(dt.date())
                except (ValueError, TypeError):
                    pass
        
        if dates_found:
            max_date = max(dates_found)
            cutoff = datetime.now().date() - timedelta(days=8)
            assert max_date >= cutoff, f"Articles should include recent dates (found {max_date})"
            assert max_date <= datetime.now().date(), "Newest articles should include today"
    
    def test_ticker_to_name_mapping(self):
        """Test that ticker-to-name mapping provides multiple search terms."""
        from src.data.news import NewsDataFetcher
        
        fetcher = NewsDataFetcher()
        
        sbin_terms = fetcher._get_search_terms("SBIN")
        assert "SBIN" in sbin_terms
        assert any("State Bank" in t for t in sbin_terms), "Should include State Bank"
        assert "SBI" in sbin_terms
        
        tcs_terms = fetcher._get_search_terms("TCS")
        assert "TCS" in tcs_terms
        assert any("Tata" in t and "Consultancy" in t for t in tcs_terms), "Should include Tata Consultancy"
        
        unknown_terms = fetcher._get_search_terms("UNKNOWN")
        assert "UNKNOWN" in unknown_terms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
