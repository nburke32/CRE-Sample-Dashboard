"""
Tests for data/sec_fetcher.py — CIK lookup, HTML extraction, rate limiting, mock API.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from data.sec_fetcher import SECFetcher


class TestCIKLookup:
    """Test _get_cik ticker-to-CIK mapping."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        return SECFetcher(cache_dir=tmp_path / "sec_cache")

    def test_valid_ticker_returns_cik(self, fetcher):
        """Known tickers should return their CIK."""
        assert fetcher._get_cik("PLD") == "0001045609"
        assert fetcher._get_cik("AAPL") == "0000320193"

    def test_case_insensitive(self, fetcher):
        """Lookup should be case-insensitive."""
        assert fetcher._get_cik("pld") == "0001045609"
        assert fetcher._get_cik("aapl") == "0000320193"

    def test_invalid_ticker_returns_none(self, fetcher):
        """Unknown tickers should return None."""
        assert fetcher._get_cik("INVALID_TICKER") is None
        assert fetcher._get_cik("") is None

    def test_all_curated_companies_have_cik(self, fetcher):
        """Every company in COMPANIES should have a valid CIK."""
        for ticker in SECFetcher.COMPANIES:
            cik = fetcher._get_cik(ticker)
            assert cik is not None, f"Ticker {ticker} has no CIK"
            assert cik.startswith("0"), f"CIK for {ticker} should be zero-padded"

    def test_curated_company_count(self, fetcher):
        """Should have 9 curated companies (5 REITs + 4 Tech)."""
        assert len(SECFetcher.COMPANIES) == 9


class TestHTMLExtraction:
    """Test extract_text_from_html cleaning logic."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        return SECFetcher(cache_dir=tmp_path / "sec_cache")

    def test_basic_html_extraction(self, fetcher):
        """Should extract clean text from HTML."""
        html = "<html><body><p>Revenue was $10 billion.</p></body></html>"
        text = fetcher.extract_text_from_html(html)
        assert "Revenue was $10 billion." in text

    def test_script_and_style_removed(self, fetcher):
        """Script and style tags should be removed."""
        html = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <script>alert('xss')</script>
            <p>Important filing content.</p>
        </body>
        </html>
        """
        text = fetcher.extract_text_from_html(html)
        assert "Important filing content." in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_table_tags_removed(self, fetcher):
        """Table tags should be removed (financial tables are noisy)."""
        html = """
        <html><body>
            <p>Summary section.</p>
            <table><tr><td>Financial data</td></tr></table>
        </body></html>
        """
        text = fetcher.extract_text_from_html(html)
        assert "Summary section." in text
        assert "Financial data" not in text

    def test_truncation_at_max_length(self, fetcher):
        """Text exceeding max_length should be truncated with a message."""
        html = "<html><body><p>" + "A" * 1000 + "</p></body></html>"
        text = fetcher.extract_text_from_html(html, max_length=100)
        assert len(text) > 100  # Includes truncation message
        assert "[... Document truncated for length ...]" in text

    def test_no_truncation_under_limit(self, fetcher):
        """Text under max_length should not be truncated."""
        html = "<html><body><p>Short text.</p></body></html>"
        text = fetcher.extract_text_from_html(html, max_length=500000)
        assert "[... Document truncated" not in text

    def test_whitespace_cleaned(self, fetcher):
        """Excessive whitespace should be cleaned up."""
        html = "<html><body><p>Line one</p>\n\n\n\n<p>Line two</p></body></html>"
        text = fetcher.extract_text_from_html(html)
        # Should not have excessive blank lines
        assert "\n\n\n" not in text


class TestRateLimiting:
    """Test _rate_limit enforcement."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        return SECFetcher(cache_dir=tmp_path / "sec_cache")

    def test_rate_limit_delays_rapid_calls(self, fetcher):
        """Back-to-back calls should be delayed by min_request_interval."""
        fetcher.last_request_time = time.time()  # Simulate a recent request

        start = time.time()
        fetcher._rate_limit()
        elapsed = time.time() - start

        # Should have waited at least part of the interval
        assert elapsed >= 0.05  # Allow some tolerance

    def test_first_call_no_delay(self, fetcher):
        """First call (last_request_time=0) should not delay."""
        fetcher.last_request_time = 0  # Default

        start = time.time()
        fetcher._rate_limit()
        elapsed = time.time() - start

        assert elapsed < 0.05  # Should be nearly instant


class TestCompanySubmissions:
    """Test get_company_submissions with mocked HTTP."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        return SECFetcher(cache_dir=tmp_path / "sec_cache")

    def test_invalid_ticker_raises(self, fetcher):
        """Should raise ValueError for unknown tickers."""
        with pytest.raises(ValueError, match="not in curated company list"):
            fetcher.get_company_submissions("INVALID_XYZ")

    def test_successful_fetch(self, fetcher):
        """Should return parsed JSON from SEC API."""
        mock_data = {
            "cik": "0001045609",
            "name": "Prologis Inc",
            "filings": {"recent": {"form": ["10-K", "10-Q"]}}
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = MagicMock()

        with patch("data.sec_fetcher.requests.get", return_value=mock_response):
            result = fetcher.get_company_submissions("PLD")

        assert result["name"] == "Prologis Inc"
        assert "filings" in result

    def test_cache_hit(self, fetcher):
        """Should use cache when file exists and is fresh."""
        import json

        cache_file = fetcher.cache_dir / "PLD_submissions.json"
        cached_data = {"name": "Prologis (cached)", "filings": {"recent": {}}}
        cache_file.write_text(json.dumps(cached_data))

        # File was just written, so it's fresh
        result = fetcher.get_company_submissions("PLD")
        assert result["name"] == "Prologis (cached)"


class TestLatestFiling:
    """Test get_latest_filing form matching."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        return SECFetcher(cache_dir=tmp_path / "sec_cache")

    def test_finds_10k(self, fetcher):
        """Should find the first 10-K in the filings list."""
        mock_submissions = {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q", "10-K", "10-K"],
                    "filingDate": ["2024-12-01", "2024-09-01", "2024-02-14", "2023-02-15"],
                    "accessionNumber": ["0001-24-001", "0001-24-002", "0001-24-003", "0001-23-001"],
                    "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm"],
                }
            }
        }

        with patch.object(fetcher, "get_company_submissions", return_value=mock_submissions):
            result = fetcher.get_latest_filing("PLD", form_type="10-K")

        assert result is not None
        assert result["form"] == "10-K"
        assert result["filing_date"] == "2024-02-14"  # First 10-K found
        assert result["ticker"] == "PLD"

    def test_returns_none_when_not_found(self, fetcher):
        """Should return None when the requested form type doesn't exist."""
        mock_submissions = {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q"],
                    "filingDate": ["2024-12-01", "2024-09-01"],
                    "accessionNumber": ["0001-24-001", "0001-24-002"],
                    "primaryDocument": ["doc1.htm", "doc2.htm"],
                }
            }
        }

        with patch.object(fetcher, "get_company_submissions", return_value=mock_submissions):
            result = fetcher.get_latest_filing("PLD", form_type="10-K")

        assert result is None


class TestAvailableCompanies:
    """Test get_available_companies."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        return SECFetcher(cache_dir=tmp_path / "sec_cache")

    def test_returns_list_of_dicts(self, fetcher):
        """Should return a list of dicts with ticker and name."""
        companies = fetcher.get_available_companies()
        assert isinstance(companies, list)
        assert len(companies) == 9

        for company in companies:
            assert "ticker" in company
            assert "name" in company

    def test_includes_expected_tickers(self, fetcher):
        """Should include both REITs and tech companies."""
        companies = fetcher.get_available_companies()
        tickers = {c["ticker"] for c in companies}
        assert "PLD" in tickers
        assert "AAPL" in tickers
        assert "GOOGL" in tickers
