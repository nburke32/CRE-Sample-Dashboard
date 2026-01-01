import streamlit as st
import time

st.set_page_config(page_title="SEC Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 SEC Chatbot")
st.markdown("Query SEC filings and regulatory documents via Snowflake.")
st.markdown("---")

# =============================================================================
# SNOWFLAKE MCP CONFIGURATION
# TODO: Update these settings once your Snowflake MCP server is configured
# =============================================================================
MCP_CONFIG = {
    "account": "your_account.us-west-2.aws",  # Your Snowflake account
    "mcp_endpoint": None,  # MCP server endpoint (set after configuration)
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema",
}

def query_mcp_server(prompt: str) -> str:
    """
    Query the Snowflake MCP server with a natural language prompt.
    
    TODO: Implement MCP client connection once server is configured.
    This function should:
    1. Connect to your Snowflake Managed MCP server
    2. Send the prompt to Cortex Analyst/Search
    3. Return the response from SEC filings data
    
    Example MCP tools available:
    - Cortex Analyst: For structured queries on SEC data
    - Cortex Search: For semantic search across filings
    """
    # Placeholder - replace with actual MCP client call
    # Example structure:
    # from snowflake.core import MCP  # hypothetical import
    # client = MCP.connect(MCP_CONFIG["mcp_endpoint"])
    # response = client.query(prompt)
    # return response.content
    
    time.sleep(1.5)  # Simulate network latency - remove when implementing
    return f"🔧 **MCP Connection Pending**\n\nYour query: *\"{prompt}\"*\n\nOnce your Snowflake MCP server is configured with SEC Marketplace data, this will return real results from 10-K, 10-Q, 8-K, and other SEC filings."


# =============================================================================
# CHAT INTERFACE
# =============================================================================

# Sidebar with connection status
with st.sidebar:
    st.markdown("### 🔗 Connection Status")
    if MCP_CONFIG["mcp_endpoint"]:
        st.success("Connected to MCP Server")
    else:
        st.warning("MCP Server not configured")
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    st.markdown("""
    - *"What were Apple's revenue trends in their latest 10-K?"*
    - *"Summarize risk factors from Tesla's most recent 10-Q"*
    - *"Show me 8-K filings for NVDA in the last 90 days"*
    """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about SEC filings..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query MCP server with spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching SEC filings..."):
            response = query_mcp_server(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

