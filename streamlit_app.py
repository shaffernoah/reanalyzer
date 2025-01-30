import streamlit as st
from market_analyzer import MarketAnalyzer
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Real Estate Market Analyzer", layout="wide")

st.title("Real Estate Market Analyzer")

# Input form
with st.form("market_analysis_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        address = st.text_input("Street Address")
    with col2:
        city = st.text_input("City")
    with col3:
        state = st.text_input("State (2-letter code)")
    
    zip_code = st.text_input("ZIP Code (optional)")
    
    submitted = st.form_submit_button("Analyze Market")

if submitted:
    analyzer = MarketAnalyzer()
    
    try:
        results = analyzer.analyze_market(address, city, state, zip_code)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Metrics")
            st.metric("Current Metro Area Median Price", f"${results['current_price']:,.2f}")
            st.metric("Year-over-Year Appreciation", f"{results['yoy_appreciation']}%")
            st.metric("Market Health Score", f"{results['market_health']}/100")
        
        with col2:
            st.subheader("Market Dynamics")
            st.metric("Price Volatility", f"{results['volatility']}%")
            st.metric("Price Momentum", f"{results['momentum']}%")
            st.text(f"Last Updated: {results['last_updated']}")
        
        # Display price history chart if available
        if 'price_history' in results:
            st.subheader("Price History")
            fig = px.line(
                results['price_history'], 
                x='date', 
                y='price',
                title=f"Home Price Trends in {results['metro_area']}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error analyzing market: {str(e)}")
