
#  PROSCIO AI — Market Risk & Equity Trend Predictor
#  Author  : Puravi Pradhan
#  Stack   : Python · Streamlit · Plotly · NumPy · Anthropic
#  Metrics : VaR (95/99%), Annualized Volatility, RSI,
#            Sharpe Ratio, Momentum Signal, 14-day AI Forecast,
#            Monte Carlo Portfolio Simulation

# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from anthropic import Anthropic
import time, random
from datetime import datetime, timedelta

# ── Page config 
st.set_page_config(
    page_title="PROSCIO AI | Market Risk Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #070d14;
    color: #e2e8f0;
  }
  .stApp { background-color: #070d14; }

  section[data-testid="stSidebar"] {
    background-color: #0a1525;
    border-right: 1px solid #00d4ff22;
  }

  [data-testid="metric-container"] {
    background: #0d1b2a;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="metric-container"] label { color: #556677 !important; font-size: 10px; letter-spacing: 2px; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display: none; }

  .stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0066ff);
    color: #000;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 10px 24px;
    box-shadow: 0 0 20px #00d4ff44;
    transition: all 0.2s;
  }
  .stButton > button:hover { box-shadow: 0 0 30px #00d4ff88; transform: translateY(-1px); }

  .stTabs [data-baseweb="tab-list"] { background: #0a1525; border-bottom: 1px solid #1e2d40; gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    color: #556677; font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 2px; font-weight: 600;
    background: transparent; border: 1px solid transparent; border-radius: 4px;
  }
  .stTabs [aria-selected="true"] {
    background: #00d4ff22 !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff44 !important;
  }

  .stSelectbox > div > div {
    background: #0d1b2a; border: 1px solid #1e2d40;
    color: #e2e8f0; font-family: 'IBM Plex Mono', monospace;
  }
  .stSlider > div { padding: 0; }
  .stDataFrame { border: 1px solid #1e2d40; border-radius: 8px; }
  hr { border-color: #1e2d40; }

  .section-label {
    font-size: 10px; color: #00d4ff; letter-spacing: 3px;
    font-weight: 700; margin-bottom: 8px;
  }
  .signal-buy        { color:#22c55e; background:#22c55e22; border:1px solid #22c55e55; padding:2px 10px; border-radius:4px; }
  .signal-sell       { color:#ef4444; background:#ef444422; border:1px solid #ef444455; padding:2px 10px; border-radius:4px; }
  .signal-neutral    { color:#6b7280; background:#6b728022; border:1px solid #6b728055; padding:2px 10px; border-radius:4px; }
  .signal-overbought { color:#f59e0b; background:#f59e0b22; border:1px solid #f59e0b55; padding:2px 10px; border-radius:4px; }
  .signal-oversold   { color:#a78bfa; background:#a78bfa22; border:1px solid #a78bfa55; padding:2px 10px; border-radius:4px; }
  .live-dot { display:inline-block; width:8px; height:8px; background:#22c55e; border-radius:50%; box-shadow:0 0 6px #22c55e; margin-right:6px; }
</style>
""", unsafe_allow_html=True)
