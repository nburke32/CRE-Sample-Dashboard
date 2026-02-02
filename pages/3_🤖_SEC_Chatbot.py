"""
SEC Filing Chatbot
Powered by SEC Edgar API + Claude AI
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sec_fetcher import SECFetcher
from models.sec_chatbot import SECChatbot

st.set_page_config(page_title="SEC Chatbot", page_icon="🤖", layout="wide")

# =============================================================================
# PASSWORD GATE
# =============================================================================

def check_password():
    """Gate access to this page with a password stored in secrets."""
    if st.session_state.get("sec_chatbot_authenticated"):
        return True

    # Check top-level and [api] section (TOML sections absorb keys below them)
    password = st.secrets.get("SEC_CHATBOT_PASSWORD", "")
    if not password and "api" in st.secrets:
        password = st.secrets["api"].get("SEC_CHATBOT_PASSWORD", "")
    if not password:
        # No password configured — allow access (local dev)
        return True

    st.title("🔒 SEC Filing Chatbot")
    st.markdown("This page requires a password to access.")
    entered = st.text_input("Password", type="password", key="sec_pw_input")
    if st.button("Submit", type="primary"):
        if entered == password:
            st.session_state["sec_chatbot_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

check_password()

# =============================================================================
# INITIALIZATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data_store" / "sec_filings"
LOG_DIR = Path(__file__).parent.parent / "logs"

@st.cache_resource
def get_sec_fetcher():
    return SECFetcher(cache_dir=DATA_DIR)

sec_fetcher = get_sec_fetcher()

# =============================================================================
# SIDEBAR - CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # API Key (Production: use Key Vault instead of Streamlit secrets)
    api_key = None

    if "api" in st.secrets and "ANTHROPIC_API_KEY" in st.secrets["api"]:
        api_key = st.secrets["api"]["ANTHROPIC_API_KEY"]
        st.success("✅ API Key Loaded")
    elif "ANTHROPIC_API_KEY" in st.secrets:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        st.success("✅ API Key Loaded")
    else:
        st.warning("⚠️ No API key found in secrets")
        st.info("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml`")

    # Model selection
    models = SECChatbot.get_available_models()
    selected_model = st.selectbox(
        "Claude Model",
        options=[m["id"] for m in models],
        format_func=lambda x: next(m["name"] for m in models if m["id"] == x),
        index=1,  # Default to Sonnet (recommended)
        help="Haiku is fastest and cheapest. Sonnet is recommended for most analysis."
    )

    st.markdown("---")

    if api_key:
        chatbot = SECChatbot(api_key=api_key, log_dir=LOG_DIR)
        usage = chatbot.get_usage_stats()

        st.markdown("### 📊 API Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Calls Today", usage["calls_today"])
            st.metric("Total Calls", usage["total_calls"])
        with col2:
            st.metric("Cost Today", f"${usage['cost_today']:.4f}")
            st.metric("Total Cost", f"${usage['total_cost']:.4f}")

    st.markdown("---")

    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - *"What were total revenues for the year?"*
    - *"Summarize the key risk factors"*
    - *"What does the company say about AI strategy?"*
    - *"What are the main business segments?"*
    - *"Summarize management's discussion and analysis"*
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_filing = None
        st.rerun()

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("🤖 SEC Filing Chatbot")
st.markdown("Ask questions about SEC filings using Claude AI. Powered by official SEC Edgar API.")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    companies = sec_fetcher.get_available_companies()
    company_options = {c["ticker"]: c["name"] for c in companies}

    selected_ticker = st.selectbox(
        "Select Company",
        options=list(company_options.keys()),
        format_func=lambda x: f"{x} - {company_options[x]}"
    )

with col2:
    filing_type = st.selectbox(
        "Filing Type",
        options=["10-K", "10-Q"],
        help="10-K = Annual Report, 10-Q = Quarterly Report"
    )

if st.button("📥 Load Filing", type="primary"):
    with st.spinner(f"Fetching latest {filing_type} for {selected_ticker}..."):
        try:
            filing = sec_fetcher.get_latest_filing(selected_ticker, filing_type)

            if not filing:
                st.error(f"No {filing_type} found for {selected_ticker}")
            else:
                # Download and parse filing
                html = sec_fetcher.download_filing(filing)
                text = sec_fetcher.extract_text_from_html(html)

                # Store in session state
                st.session_state.current_filing = {
                    "ticker": selected_ticker,
                    "company": company_options[selected_ticker],
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "text": text,
                    "text_length": len(text)
                }

                st.success(f"✅ Loaded {filing_type} filed on {filing['filing_date']}")
                st.info(f"📄 Filing length: {len(text):,} characters")

        except Exception as e:
            st.error(f"Error loading filing: {e}")

if "current_filing" in st.session_state and st.session_state.current_filing:
    filing_info = st.session_state.current_filing
    st.info(
        f"📋 **Current Filing:** {filing_info['ticker']} - {filing_info['company']} "
        f"({filing_info['form']} filed {filing_info['filing_date']}) - "
        f"{filing_info['text_length']:,} characters"
    )

st.markdown("---")

# =============================================================================
# CHAT INTERFACE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "metadata" in message:
            # Show cost info
            meta = message["metadata"]
            st.caption(
                f"💰 {meta['total_tokens']:,} tokens | ${meta['cost']:.4f} | {meta['model'].split('-')[1].title()}"
            )

if prompt := st.chat_input("Ask a question about the filing..."):
    if "current_filing" not in st.session_state or not st.session_state.current_filing:
        st.error("⚠️ Please load a filing first using the 'Load Filing' button above.")
        st.stop()

    if not api_key:
        st.error("⚠️ No API key configured. Add ANTHROPIC_API_KEY to .streamlit/secrets.toml")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing filing..."):
            try:
                filing_info = st.session_state.current_filing
                chatbot = SECChatbot(api_key=api_key, log_dir=LOG_DIR)

                response = chatbot.ask_question(
                    filing_text=filing_info["text"],
                    question=prompt,
                    model=selected_model,
                    company_name=filing_info["company"]
                )

                st.markdown(response["answer"])
                st.caption(
                    f"💰 {response['total_tokens']:,} tokens | "
                    f"${response['cost']:.4f} | "
                    f"{response['model'].split('-')[1].title()}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": response
                })

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}"
                })

# =============================================================================
# DOCUMENTATION
# =============================================================================

with st.expander("📚 About This Chatbot"):
    st.markdown("""
    **How It Works:**

    1. **Fetch Filings**: Retrieves SEC filings (10-K, 10-Q) from official SEC Edgar API
    2. **Parse Content**: Extracts clean text from HTML filings
    3. **AI Analysis**: Uses Claude AI to answer questions based on filing content
    4. **Local Caching**: Filings are cached locally to avoid redundant API calls

    **Data Sources:**

    - **SEC Edgar API**: Official SEC database (free, public, no auth required)
    - **Claude API**: Anthropic's AI for natural language understanding

    **Curated Companies:**

    This demo includes 9 pre-selected companies (REITs + Tech):
    - **REITs**: PLD, EQIX, DLR, SPG, O
    - **Tech**: AAPL, MSFT, AMZN, GOOGL

    **Security Notes:**

    - ⚠️ **Portfolio Demo**: API keys stored in Streamlit secrets (local only)
    - ⚠️ **Production Approach**: Use Azure Key Vault / AWS Secrets Manager
    - ⚠️ **Rate Limiting**: SEC allows 10 req/sec max (enforced in code)
    - ⚠️ **Cost Controls**: $20 spend limit on API key, usage monitoring enabled

    **Limitations:**

    - Filings are truncated to 500K characters for Claude context limits
    - Only latest filing of each type is cached per company
    - No full-text search across all companies (curated set only)

    **Production Enhancements:**

    In a commercial setting, I would:
    1. Use paid SEC data provider (Bloomberg, FactSet) for broader coverage
    2. Implement proper authentication and user session management
    3. Add rate limiting per user (not just SEC rate limits)
    4. Set up monitoring/alerting (email on high usage)
    5. Use Key Vault for secrets (never Streamlit secrets)
    6. Add audit logging for compliance
    """)

with st.expander("⚠️ Data Limitations & Coverage"):
    st.markdown("""
    **Curated Company List:**

    This chatbot is designed as a **portfolio demonstration**, not a production tool.
    It includes a curated set of 9 companies to demonstrate:
    - SEC Edgar API integration
    - Filing parsing and text extraction
    - Claude AI integration for Q&A
    - Caching strategy
    - Usage monitoring

    **Why Curated?**

    - ✅ Manageable data size (no massive storage needs)
    - ✅ Fast demo experience (pre-cached filings)
    - ✅ Cost control (limited API usage)
    - ✅ Focus on skill demonstration, not comprehensive coverage

    **Production Approach:**

    In a real commercial application, I would:
    - Integrate with **Bloomberg Terminal / FactSet / S&P Capital IQ**
    - Support search across **all public companies** (~6,000+ filers)
    - Enable **historical filing analysis** (multiple years)
    - Add **comparative analysis** (compare companies side-by-side)
    - Implement **semantic search** across filing corpus
    - Build **alerting system** for new filings

    The goal here is demonstrating **technical capability**, not building a commercial product.
    """)
