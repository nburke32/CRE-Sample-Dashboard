import streamlit as st

st.set_page_config(
    page_title="Commercial Real Estate Dashboard",
    page_icon="🏢",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Logo", width=150)
    st.markdown("---")
    st.markdown("### Quick Links")
    st.page_link("pages/1_📊_Market_Analytics.py", label="📊 Market Analytics")
    st.page_link("pages/2_🗺️_Map_View.py", label="🗺️ Map View")
    st.page_link("pages/3_🤖_SEC_Chatbot.py", label="🤖 SEC Chatbot")
    st.page_link("pages/4_🔮_Predictive_Modeling.py", label="🔮 Predictive Modeling")
    st.markdown("---")
    st.caption("v0.1.0")

# Main content
st.title("🏢 Commercial Real Estate Dashboard")
st.markdown("Welcome back! Here's your overview.")

st.markdown("---")

# Navigation cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 📊 Market Analytics
    Commercial trends, cap rates, and market insights.
    """)
    st.page_link("pages/1_📊_Market_Analytics.py", label="Go to Analytics →")

with col2:
    st.markdown("""
    ### 🗺️ Map View
    Explore commercial properties on an interactive map.
    """)
    st.page_link("pages/2_🗺️_Map_View.py", label="Open Map →")

with col3:
    st.markdown("""
    ### 🤖 SEC Chatbot
    Query SEC filings and regulatory documents.
    """)
    st.page_link("pages/3_🤖_SEC_Chatbot.py", label="Chat with SEC →")

with col4:
    st.markdown("""
    ### 🔮 Predictive Modeling
    Forecast prices, cap rates, and NOI.
    """)
    st.page_link("pages/4_🔮_Predictive_Modeling.py", label="View Forecasts →")

st.markdown("---")

# Placeholder metrics
st.subheader("Quick Stats")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Listings", "142", "+12")
m2.metric("Avg. Price", "$485K", "-2.3%")
m3.metric("Days on Market", "28", "-5")
m4.metric("Showings This Week", "37", "+8")
