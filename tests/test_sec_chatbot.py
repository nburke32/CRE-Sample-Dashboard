"""
Tests for models/sec_chatbot.py — cost calculation, response cleaning, usage stats, mock API.
"""

from unittest.mock import MagicMock, patch

import pytest

from models.sec_chatbot import SECChatbot


class TestCostCalculation:
    """Test _calculate_cost pricing logic for each model tier."""

    @pytest.fixture
    def chatbot(self, tmp_path):
        """Create chatbot with mocked Anthropic client."""
        with patch("models.sec_chatbot.anthropic.Anthropic"):
            return SECChatbot(api_key="test_key", log_dir=tmp_path / "logs")

    def test_haiku_cost(self, chatbot):
        """Haiku: $1/MTok input, $5/MTok output."""
        cost = chatbot._calculate_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
        assert abs(cost - 6.00) < 0.001  # $1 input + $5 output

    def test_sonnet_cost(self, chatbot):
        """Sonnet: $3/MTok input, $15/MTok output."""
        cost = chatbot._calculate_cost("claude-sonnet-4-5", 1_000_000, 1_000_000)
        assert abs(cost - 18.00) < 0.001  # $3 input + $15 output

    def test_opus_cost(self, chatbot):
        """Opus: $15/MTok input, $75/MTok output."""
        cost = chatbot._calculate_cost("claude-opus-4-5", 1_000_000, 1_000_000)
        assert abs(cost - 90.00) < 0.001  # $15 input + $75 output

    def test_small_token_count(self, chatbot):
        """Typical usage: ~10k input, ~1k output with Sonnet."""
        cost = chatbot._calculate_cost("claude-sonnet-4-5", 10_000, 1_000)
        # (10000/1M * $3) + (1000/1M * $15) = $0.03 + $0.015 = $0.045
        assert abs(cost - 0.045) < 0.001

    def test_unknown_model_zero_cost(self, chatbot):
        """Unknown model should default to zero cost."""
        cost = chatbot._calculate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_zero_tokens_zero_cost(self, chatbot):
        """Zero tokens should result in zero cost."""
        cost = chatbot._calculate_cost("claude-sonnet-4-5", 0, 0)
        assert cost == 0.0


class TestResponseCleaning:
    """Test clean_response_text dollar sign escaping."""

    def test_dollar_amount_escaped(self):
        """Dollar amounts should be escaped for Streamlit markdown."""
        text = "Revenue was $8.2 billion (up from $7.8 billion)"
        result = SECChatbot.clean_response_text(text)
        assert "\\$8" in result
        assert "\\$7" in result

    def test_already_escaped_not_double_escaped(self):
        """Already-escaped dollar signs should not be double-escaped."""
        text = "Revenue was \\$8.2 billion"
        result = SECChatbot.clean_response_text(text)
        assert "\\\\$" not in result  # No double backslash
        assert "\\$8" in result

    def test_non_currency_dollar_untouched(self):
        """Dollar signs not followed by digits should be left alone."""
        text = "The $variable was set"
        result = SECChatbot.clean_response_text(text)
        # $v is not $digit, so should stay as-is
        assert "$variable" in result

    def test_multiple_amounts(self):
        """Multiple dollar amounts in one string."""
        text = "From $1.2M to $3.4B, costing $500K"
        result = SECChatbot.clean_response_text(text)
        assert "\\$1" in result
        assert "\\$3" in result
        assert "\\$5" in result

    def test_empty_string(self):
        """Empty string should return empty."""
        assert SECChatbot.clean_response_text("") == ""

    def test_no_dollars(self):
        """Text without dollar signs should be unchanged."""
        text = "The company reported strong growth in Q4."
        assert SECChatbot.clean_response_text(text) == text


class TestUsageStats:
    """Test get_usage_stats log parsing."""

    @pytest.fixture
    def chatbot_with_logs(self, tmp_path):
        """Create chatbot with a pre-populated log file."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        # Write sample log entries
        log_file = log_dir / "api_usage.log"
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        log_file.write_text(
            f"{today} 10:30:00 - INFO - Model: claude-sonnet-4-5 | Input: 5000 tokens | Output: 500 tokens | Cost: $0.0225 | Response: 2.3s\n"
            f"{today} 11:00:00 - INFO - Model: claude-haiku-4-5 | Input: 3000 tokens | Output: 300 tokens | Cost: $0.0045 | Response: 1.1s\n"
            f"2024-01-15 09:00:00 - INFO - Model: claude-sonnet-4-5 | Input: 10000 tokens | Output: 1000 tokens | Cost: $0.0450 | Response: 3.5s\n"
        )

        with patch("models.sec_chatbot.anthropic.Anthropic"):
            return SECChatbot(api_key="test_key", log_dir=log_dir)

    def test_total_calls_counted(self, chatbot_with_logs):
        """Should count total API calls."""
        stats = chatbot_with_logs.get_usage_stats()
        assert stats["total_calls"] == 3

    def test_total_cost_summed(self, chatbot_with_logs):
        """Should sum all costs correctly."""
        stats = chatbot_with_logs.get_usage_stats()
        expected = 0.0225 + 0.0045 + 0.0450
        assert abs(stats["total_cost"] - expected) < 0.001

    def test_today_calls_counted(self, chatbot_with_logs):
        """Should count only today's calls separately."""
        stats = chatbot_with_logs.get_usage_stats()
        assert stats["calls_today"] == 2

    def test_today_cost_summed(self, chatbot_with_logs):
        """Should sum only today's costs."""
        stats = chatbot_with_logs.get_usage_stats()
        expected_today = 0.0225 + 0.0045
        assert abs(stats["cost_today"] - expected_today) < 0.001

    def test_empty_log_returns_zeros(self, tmp_path):
        """No log file should return all zeros."""
        with patch("models.sec_chatbot.anthropic.Anthropic"):
            chatbot = SECChatbot(api_key="test_key", log_dir=tmp_path / "empty_logs")

        stats = chatbot.get_usage_stats()
        assert stats["total_calls"] == 0
        assert stats["total_cost"] == 0.0


class TestAvailableModels:
    """Test get_available_models structure."""

    def test_returns_three_models(self):
        """Should return exactly 3 model options."""
        models = SECChatbot.get_available_models()
        assert len(models) == 3

    def test_model_structure(self):
        """Each model should have id, name, and description."""
        for model in SECChatbot.get_available_models():
            assert "id" in model
            assert "name" in model
            assert "description" in model

    def test_model_ids_match_pricing(self):
        """Model IDs should correspond to entries in PRICING dict."""
        model_ids = {m["id"] for m in SECChatbot.get_available_models()}
        pricing_ids = set(SECChatbot.MODEL_PRICING.keys())
        assert model_ids == pricing_ids


class TestAskQuestion:
    """Test ask_question with mocked Anthropic client."""

    @pytest.fixture
    def chatbot(self, tmp_path):
        """Create chatbot with mocked API client."""
        with patch("models.sec_chatbot.anthropic.Anthropic") as mock_anthropic:
            chatbot = SECChatbot(api_key="test_key", log_dir=tmp_path / "logs")

            # Setup mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Revenue was $10 billion.")]
            mock_response.usage.input_tokens = 5000
            mock_response.usage.output_tokens = 200

            chatbot.client.messages.create.return_value = mock_response
            return chatbot

    def test_returns_expected_structure(self, chatbot):
        """ask_question should return dict with answer, tokens, cost, etc."""
        result = chatbot.ask_question(
            filing_text="Sample filing content.",
            question="What was the revenue?",
            model="claude-sonnet-4-5",
            company_name="Prologis"
        )

        assert "answer" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "total_tokens" in result
        assert "cost" in result
        assert "model" in result
        assert "response_time" in result
        assert "timestamp" in result

    def test_token_counts_correct(self, chatbot):
        """Token counts should match the mock response."""
        result = chatbot.ask_question(
            filing_text="Content.", question="Question?",
            model="claude-sonnet-4-5"
        )
        assert result["input_tokens"] == 5000
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 5200

    def test_cost_calculated(self, chatbot):
        """Cost should be calculated from model pricing."""
        result = chatbot.ask_question(
            filing_text="Content.", question="Question?",
            model="claude-sonnet-4-5"
        )
        # (5000/1M * $3) + (200/1M * $15) = $0.015 + $0.003 = $0.018
        assert abs(result["cost"] - 0.018) < 0.001

    def test_response_text_cleaned(self, chatbot):
        """Dollar signs in the response should be escaped."""
        result = chatbot.ask_question(
            filing_text="Content.", question="Question?",
            model="claude-sonnet-4-5"
        )
        # The mock returns "Revenue was $10 billion."
        # clean_response_text should escape the $10
        assert "\\$10" in result["answer"]

    def test_api_error_raises(self, chatbot):
        """API errors should be wrapped in a generic Exception."""
        import anthropic
        chatbot.client.messages.create.side_effect = anthropic.APIError(
            message="Rate limited",
            request=MagicMock(),
            body=None,
        )

        with pytest.raises(Exception, match="Claude API error"):
            chatbot.ask_question(
                filing_text="Content.", question="Question?",
                model="claude-sonnet-4-5"
            )
