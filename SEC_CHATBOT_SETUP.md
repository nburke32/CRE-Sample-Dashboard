# SEC Chatbot Setup Guide

## Quick Start

The SEC Chatbot is now built and ready to use! Here's how to set it up:

### 1. Install Dependencies

```bash
pip3 install anthropic beautifulsoup4
```

### 2. Add Your API Key

1. Copy the secrets template:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. Edit `.streamlit/secrets.toml` and add your Anthropic API key:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
   ```

3. Get your API key from: https://console.anthropic.com/

### 3. Run the Dashboard

```bash
streamlit run Home.py
```

Navigate to the "🤖 SEC Chatbot" page in the sidebar.

## How to Use

1. **Select a Company**: Choose from 9 pre-selected REITs and Tech companies
2. **Choose Filing Type**: 10-K (annual) or 10-Q (quarterly)
3. **Load Filing**: Click "📥 Load Filing" to fetch and cache the document
4. **Ask Questions**: Use the chat interface to ask about the filing

## Example Questions

- "What were total revenues for the year?"
- "Summarize the key risk factors"
- "What does the company say about AI strategy?"
- "What are the main business segments?"
- "Summarize management's discussion and analysis"

## Features

### ✅ What's Included

- **SEC Edgar API Integration**: Fetches filings directly from official SEC database
- **Claude AI Q&A**: Powered by Anthropic's Claude models (Haiku, Sonnet, Opus)
- **Local Caching**: Downloaded filings are cached to avoid redundant API calls
- **Usage Monitoring**: Real-time tracking of API calls and costs
- **Model Selection**: Choose between speed/cost (Haiku) and capability (Opus)
- **Security**: API keys stored in Streamlit secrets (local only)

### 📊 Curated Companies

**REITs** (aligned with dashboard data):
- PLD (Prologis) - Industrial/Logistics
- EQIX (Equinix) - Data Centers
- DLR (Digital Realty) - Data Centers
- SPG (Simon Property) - Retail Malls
- O (Realty Income) - Net Lease

**Tech Companies**:
- AAPL (Apple)
- MSFT (Microsoft)
- AMZN (Amazon)
- GOOGL (Alphabet/Google)

## Architecture

```
User Input
    ↓
[Streamlit UI]
    ↓
[SEC Fetcher] → SEC Edgar API (free, public)
    ↓
[Filing Cache] (local storage)
    ↓
[Claude Chatbot] → Anthropic API ($0.40-$15/MTok)
    ↓
[Usage Logger] → logs/api_usage.log
    ↓
Response to User
```

## Cost & Rate Limits

### SEC Edgar API
- **Cost**: Free
- **Rate Limit**: 10 requests/second (enforced in code)
- **Auth**: None required (just User-Agent header)

### Claude API
- **Haiku 4**: $0.40/MTok input, $2/MTok output (fastest, cheapest)
- **Sonnet 4**: $3/MTok input, $15/MTok output (balanced - recommended)
- **Opus 4**: $15/MTok input, $75/MTok output (most capable)

**Typical 10-K Q&A Cost**:
- Haiku: ~$0.12-$0.36 per question
- Sonnet: ~$0.90-$2.70 per question
- Opus: ~$4.50-$13.50 per question

**Recommendation**: Use Haiku for simple questions, Sonnet for detailed analysis.

## Security Notes

### Current Setup (Portfolio Demo)
- ✅ API keys in `.streamlit/secrets.toml` (gitignored)
- ✅ $20 spend limit on API key
- ✅ Usage monitoring enabled
- ✅ Local deployment only
- ⚠️ Suitable for portfolio/demo purposes

### Production Approach
In a commercial setting, you would:
1. **Use Key Vault**: Azure Key Vault / AWS Secrets Manager / GCP Secret Manager
2. **Add Authentication**: User login + session management
3. **Implement Rate Limiting**: Per-user quotas (not just SEC limits)
4. **Set Up Monitoring**: Email alerts on high usage
5. **Add Audit Logging**: Track who asked what, when
6. **Deploy Securely**: HTTPS, firewall rules, VPC

## Troubleshooting

### "No API key found in secrets"
- Make sure `.streamlit/secrets.toml` exists
- Check that `ANTHROPIC_API_KEY` is set correctly
- Restart Streamlit after adding the key

### "Failed to fetch submissions"
- Check internet connection
- Verify SEC isn't blocking your IP (rate limit exceeded)
- Wait 10 minutes if rate limited

### "Claude API error"
- Verify API key is valid
- Check you haven't hit spend limit
- Ensure API key has proper permissions

### Filing takes too long to load
- First load downloads from SEC (may take 10-30 seconds)
- Subsequent loads use cache (instant)
- Large filings are truncated to 500K characters

## Files & Structure

```
streamlit-dashboard/
├── pages/
│   └── 3_🤖_SEC_Chatbot.py      # Streamlit UI
├── data/
│   └── sec_fetcher.py            # SEC Edgar API client
├── models/
│   └── sec_chatbot.py            # Claude AI integration
├── data_store/
│   └── sec_filings/              # Cached filings (gitignored)
├── logs/
│   └── api_usage.log             # Usage tracking (gitignored)
└── .streamlit/
    ├── secrets.toml              # API key (gitignored)
    └── secrets.toml.example      # Template
```

## Limitations

This is a **portfolio demonstration**, not a production tool. Limitations include:

- ✅ Only 9 curated companies (not all 6,000+ public filers)
- ✅ Latest filing only (no historical analysis)
- ✅ 500K character limit per filing (Claude context limits)
- ✅ No full-text search across companies
- ✅ No comparative analysis features

These limitations are intentional to keep the demo:
- Fast and responsive
- Cost-controlled
- Focused on skill demonstration

## Next Steps

1. **Add your API key** to `.streamlit/secrets.toml`
2. **Test the chatbot** with a simple question
3. **Monitor usage** in the sidebar
4. **Try different models** to see the cost/quality trade-off

Enjoy exploring SEC filings with AI! 🚀
