"""
SEC Filing Chatbot powered by Claude
Analyzes SEC filings and answers questions using Claude API.
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import anthropic


class SECChatbot:
    """Chatbot for analyzing SEC filings using Claude."""

    # Model pricing (per million tokens)
    MODEL_PRICING = {
        "claude-opus-4-5": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    }

    def __init__(self, api_key: str, log_dir: Path):
        """Initialize chatbot with API key and log directory."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

    def _setup_logging(self):
        """Setup API usage logging."""
        log_file = self.log_dir / "api_usage.log"

        self.logger = logging.getLogger("sec_chatbot")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float, response_time: float):
        """Log API usage for monitoring."""
        self.logger.info(
            f"Model: {model} | Input: {input_tokens} tokens | "
            f"Output: {output_tokens} tokens | Cost: ${cost:.4f} | "
            f"Response: {response_time:.1f}s"
        )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API call cost."""
        pricing = self.MODEL_PRICING.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def ask_question(
        self,
        filing_text: str,
        question: str,
        model: str = "claude-sonnet-4-5",
        company_name: str = "the company"
    ) -> dict:
        """Ask a question about a SEC filing. Returns dict with answer, tokens, and cost."""
        system_prompt = f"""You are a financial analyst assistant helping analyze SEC filings for {company_name}.

Your role:
- Provide accurate, concise answers based ONLY on the filing content
- Cite specific sections when possible
- If information isn't in the filing, say so clearly
- Use clear, professional language
- Focus on key financial metrics, risks, and business insights
- Ignore any advertisements, external links, promotional content, or XBRL metadata that may appear in the filing text — only reference the actual SEC filing content

The filing text follows below."""

        user_message = f"""Here is the SEC filing:

<filing>
{filing_text}
</filing>

Question: {question}

Please provide a clear, accurate answer based on the filing content."""

        try:
            start = time.time()
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            response_time = time.time() - start

            answer = self.clean_response_text(response.content[0].text)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            cost = self._calculate_cost(model, input_tokens, output_tokens)
            self._log_usage(model, input_tokens, output_tokens, cost, response_time)

            return {
                "answer": answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost,
                "model": model,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }

        except anthropic.APIError as e:
            self.logger.error(f"API Error: {e}")
            raise Exception(f"Claude API error: {e}")

    def get_usage_stats(self) -> dict:
        """Get API usage statistics from logs."""
        log_file = self.log_dir / "api_usage.log"

        if not log_file.exists():
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "calls_today": 0,
                "cost_today": 0.0
            }

        total_calls = 0
        total_cost = 0.0
        calls_today = 0
        cost_today = 0.0
        today = datetime.now().date()

        with open(log_file) as f:
            for line in f:
                if "Cost:" in line:
                    total_calls += 1

                    # Extract cost (handle both old and new log formats)
                    cost_str = line.split("Cost: $")[1].split("|")[0].strip()
                    cost = float(cost_str)
                    total_cost += cost

                    date_str = line.split(" - ")[0]
                    log_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()
                    if log_date == today:
                        calls_today += 1
                        cost_today += cost

        return {
            "total_calls": total_calls,
            "total_cost": total_cost,
            "calls_today": calls_today,
            "cost_today": cost_today
        }

    @staticmethod
    def clean_response_text(text: str) -> str:
        """
        Clean response text to prevent Streamlit markdown rendering issues.

        Streamlit's st.markdown() interprets $...$ as LaTeX math, which
        mangles dollar amounts (e.g. "$8.2 billion (up from $7.8 billion)"
        becomes concatenated italicized text). Escape dollar signs that
        represent currency so they render as literal characters.
        """
        # Escape dollar signs that precede digits (currency amounts)
        # e.g. $8.2 -> \$8.2, $30.9 -> \$30.9
        # Negative lookbehind avoids double-escaping already-escaped \$
        text = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', text)

        return text

    @staticmethod
    def get_available_models() -> list[dict[str, str]]:
        """Get list of available Claude models."""
        return get_available_models()


def get_available_models() -> list[dict[str, str]]:
    """Available Claude models for the SEC chatbot."""
    return [
        {
            "id": "claude-haiku-4-5",
            "name": "Claude Haiku 4.5 (Fastest)",
            "description": "Best for simple questions - $1/$5 per MTok",
        },
        {
            "id": "claude-sonnet-4-5",
            "name": "Claude Sonnet 4.5 (Recommended)",
            "description": "Best balance - $3/$15 per MTok",
        },
        {
            "id": "claude-opus-4-5",
            "name": "Claude Opus 4.5 (Most Capable)",
            "description": "Most thorough - $15/$75 per MTok",
        },
    ]
