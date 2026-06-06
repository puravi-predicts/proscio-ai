
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

# ── Stock Universe (10 stocks) 
STOCKS = {
    "AAPL":  {"name": "Apple Inc.",        "sector": "Technology",      "base": 189.5},
    "MSFT":  {"name": "Microsoft Corp.",   "sector": "Technology",      "base": 415.2},
    "GOOGL": {"name": "Alphabet Inc.",     "sector": "Technology",      "base": 178.3},
    "AMZN":  {"name": "Amazon.com Inc.",   "sector": "E-Commerce/Cloud","base": 192.4},
    "META":  {"name": "Meta Platforms",    "sector": "Social Media",    "base": 512.7},
    "JPM":   {"name": "JPMorgan Chase",    "sector": "Finance",         "base": 198.7},
    "TSLA":  {"name": "Tesla Inc.",        "sector": "EV/Auto",         "base": 248.6},
    "NVDA":  {"name": "NVIDIA Corp.",      "sector": "Semiconductors",  "base": 875.4},
    "NFLX":  {"name": "Netflix Inc.",      "sector": "Streaming",       "base": 628.3},
    "AMD":   {"name": "Advanced Micro Dev","sector": "Semiconductors",  "base": 162.8},
}

SECTOR_COLORS = {
    "Technology":       "#00d4ff",
    "E-Commerce/Cloud": "#06b6d4",
    "Social Media":     "#818cf8",
    "Finance":          "#22c55e",
    "EV/Auto":          "#f59e0b",
    "Semiconductors":   "#a78bfa",
    "Streaming":        "#f472b6",
}

PLOT_THEME = dict(
    paper_bgcolor="#0a1525", plot_bgcolor="#0a1525",
    font=dict(family="IBM Plex Mono", color="#8899aa", size=10),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#1e2d40", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2d40", showgrid=True, zeroline=False),
)


#  QUANT MATH ENGINE


def generate_price_history(base: float, days: int = 60, seed: int = None) -> np.ndarray:
    """Geometric Brownian Motion price simulation."""
    if seed is not None:
        np.random.seed(seed)
    prices = [base]
    for _ in range(days - 1):
        change = (np.random.random() - 0.48) * base * 0.025
        prices.append(max(prices[-1] + change, base * 0.5))
    return np.array(prices)


def calc_volatility(prices: np.ndarray) -> float:
    """Annualised Historical Volatility from log-returns (σ × √252)."""
    log_returns = np.log(prices[1:] / prices[:-1])
    return float(np.std(log_returns, ddof=1) * np.sqrt(252) * 100)


def calc_var(prices: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Simulation VaR at given confidence level."""
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    return float(abs(np.percentile(returns, (1 - confidence) * 100)) * 100)


def calc_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI — 14-period default."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-period - 1:])
    gains  = deltas[deltas > 0].sum() / period
    losses = abs(deltas[deltas < 0].sum()) / period
    if losses == 0:
        return 100.0
    return float(100 - 100 / (1 + gains / losses))


def calc_sharpe(prices: np.ndarray, risk_free: float = 0.05) -> float:
    """Annualised Sharpe Ratio (risk-free = 5%)."""
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    mean    = np.mean(returns) * 252
    std     = np.std(returns, ddof=1) * np.sqrt(252)
    return float((mean - risk_free) / std) if std != 0 else 0.0


def calc_macd(prices: np.ndarray):
    """MACD Line, Signal Line, and Histogram."""
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False).mean().values
    macd_line = ema(prices, 12) - ema(prices, 26)
    signal    = ema(macd_line, 9)
    return macd_line, signal, macd_line - signal


def calc_bollinger(prices: np.ndarray, period: int = 20):
    """Bollinger Bands: upper, middle (SMA), lower."""
    s   = pd.Series(prices)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std(ddof=1)
    return (mid + 2 * std).values, mid.values, (mid - 2 * std).values


def predict_trend(prices: np.ndarray):
    """
    Momentum + RSI fusion signal with 14-day stochastic price forecast.
    Returns: signal (str), confidence (float), future_prices (np.ndarray)
    """
    momentum   = (np.mean(prices[-10:]) - np.mean(prices[-20:-10])) / np.mean(prices[-20:-10])
    rsi        = calc_rsi(prices)
    vol        = calc_volatility(prices)

    if   momentum >  0.02 and rsi < 70: signal, conf = "BUY",        min(85, 60 + momentum * 500)
    elif momentum < -0.02 and rsi > 30: signal, conf = "SELL",       min(85, 60 + abs(momentum) * 500)
    elif rsi > 75:                       signal, conf = "OVERBOUGHT", 70.0
    elif rsi < 25:                       signal, conf = "OVERSOLD",   70.0
    else:                                signal, conf = "NEUTRAL",    50.0

    future, last = [], prices[-1]
    for _ in range(14):
        drift = momentum * 0.1 + (np.random.random() - 0.5) * vol * 0.003
        last  = last * (1 + drift)
        future.append(last)

    return signal, round(conf, 1), np.array(future)


