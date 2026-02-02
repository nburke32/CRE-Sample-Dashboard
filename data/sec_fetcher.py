"""
SEC Edgar Filing Fetcher
Retrieves SEC filings (10-K, 10-Q) for curated companies using the official SEC Edgar API.
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


class SECFetcher:
    """Fetches and caches SEC filings from Edgar."""

    BASE_URL = "https://data.sec.gov"
    HEADERS = {
        "User-Agent": "Portfolio Demo nolanburke@example.com",  # SEC requires User-Agent
        "Accept-Encoding": "gzip, deflate"
        # Don't set Host header - let requests handle it automatically
    }

    # Curated companies for the chatbot
    COMPANIES = {
        # REITs (align with dashboard data)
        "PLD": {"name": "Prologis Inc", "cik": "0001045609"},
        "EQIX": {"name": "Equinix Inc", "cik": "0001101239"},
        "DLR": {"name": "Digital Realty Trust Inc", "cik": "0001297996"},
        "SPG": {"name": "Simon Property Group Inc", "cik": "0001063761"},
        "O": {"name": "Realty Income Corp", "cik": "0000726728"},

        # Tech companies (for variety)
        "AAPL": {"name": "Apple Inc", "cik": "0000320193"},
        "MSFT": {"name": "Microsoft Corp", "cik": "0000789019"},
        "AMZN": {"name": "Amazon.com Inc", "cik": "0001018724"},
        "GOOGL": {"name": "Alphabet Inc", "cik": "0001652044"},
    }

    def __init__(self, cache_dir: Path):
        """Initialize SEC fetcher with cache directory."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting (SEC allows 10 req/sec max)
        self.last_request_time = 0
        self.min_request_interval = 0.11  # Slightly over 0.1s to stay under 10/sec

    def _rate_limit(self):
        """Enforce rate limiting to comply with SEC policy (10 req/sec max)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _get_cik(self, ticker: str) -> str | None:
        """Get CIK for a ticker from curated list."""
        company = self.COMPANIES.get(ticker.upper())
        return company["cik"] if company else None

    def get_company_submissions(self, ticker: str) -> dict:
        """
        Get all submissions for a company.

        Args:
            ticker: Stock ticker (e.g., 'AAPL')

        Returns:
            Dict with company info and filings list
        """
        cik = self._get_cik(ticker)
        if not cik:
            raise ValueError(f"Ticker {ticker} not in curated company list")

        # Check cache first
        cache_file = self.cache_dir / f"{ticker}_submissions.json"
        if cache_file.exists():
            # Use cache if less than 1 day old
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                with open(cache_file) as f:
                    return json.load(f)

        # Fetch from SEC
        self._rate_limit()
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            return data

        except requests.RequestException as e:
            raise Exception(f"Failed to fetch submissions for {ticker}: {e}")

    def get_latest_filing(self, ticker: str, form_type: str = "10-K") -> dict | None:
        """
        Get the latest filing of a specific type.

        Args:
            ticker: Stock ticker
            form_type: Filing type (10-K, 10-Q, 8-K, etc.)

        Returns:
            Dict with filing details or None if not found
        """
        submissions = self.get_company_submissions(ticker)

        recent_filings = submissions.get("filings", {}).get("recent", {})
        forms = recent_filings.get("form", [])
        filing_dates = recent_filings.get("filingDate", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        primary_documents = recent_filings.get("primaryDocument", [])

        # Find the latest filing of the requested type
        for i, form in enumerate(forms):
            if form == form_type:
                return {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": filing_dates[i],
                    "accession_number": accession_numbers[i],
                    "primary_document": primary_documents[i],
                    "cik": self._get_cik(ticker)
                }

        return None

    def download_filing(self, filing: dict) -> str:
        """
        Download the full text of a filing.

        Args:
            filing: Filing dict from get_latest_filing()

        Returns:
            Full text of the filing (HTML)
        """
        ticker = filing["ticker"]
        accession = filing["accession_number"].replace("-", "")  # Remove dashes for URL
        primary_doc = filing["primary_document"]
        cik = filing["cik"]

        # Check cache
        cache_file = self.cache_dir / f"{ticker}_{filing['form']}_{filing['filing_date']}.html"
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')

        # Download from SEC
        # URL format: https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION-WITH-DASHES/DOCUMENT
        self._rate_limit()
        url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            html_content = response.text

            # Cache it
            cache_file.write_text(html_content, encoding='utf-8')

            return html_content

        except requests.RequestException as e:
            raise Exception(f"Failed to download filing: {e}")

    def extract_text_from_html(self, html: str, max_length: int = 500000) -> str:
        """
        Extract clean text from SEC filing HTML.

        Args:
            html: Raw HTML content
            max_length: Maximum characters to return (for Claude context limits)

        Returns:
            Clean text content
        """
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for tag in soup(["script", "style", "table"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator='\n')

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[... Document truncated for length ...]"

        return text

    def get_filing_text(self, ticker: str, form_type: str = "10-K") -> str | None:
        """
        Convenience method: Get clean text of latest filing.

        Args:
            ticker: Stock ticker
            form_type: Filing type (10-K, 10-Q)

        Returns:
            Clean text of filing or None if not found
        """
        filing = self.get_latest_filing(ticker, form_type)
        if not filing:
            return None

        html = self.download_filing(filing)
        text = self.extract_text_from_html(html)

        return text

    def get_available_companies(self) -> list[dict[str, str]]:
        """Get list of available companies."""
        return [
            {"ticker": ticker, "name": info["name"]}
            for ticker, info in self.COMPANIES.items()
        ]
