
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



#  MONTE CARLO ENGINE


def monte_carlo_simulation(
    start_price: float,
    mu: float,          
    sigma: float,       
    days: int = 252,
    n_simulations: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Geometric Brownian Motion Monte Carlo.
    dS = S · (μ dt + σ √dt · Z),   Z ~ N(0,1)
    Returns array shape (n_simulations, days+1).
    """
    np.random.seed(seed)
    dt       = 1 / 252
    paths    = np.zeros((n_simulations, days + 1))
    paths[:, 0] = start_price

    for t in range(1, days + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )
    return paths


def mc_portfolio_simulation(
    weights: dict,
    stock_data: dict,
    portfolio_value: float = 100_000,
    days: int = 252,
    n_simulations: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Weighted portfolio Monte Carlo using each stock's estimated drift & vol.
    Returns simulated portfolio value paths (n_simulations, days+1).
    """
    np.random.seed(seed)
    total_w = sum(weights.values())
    if total_w == 0:
        return np.zeros((n_simulations, days + 1))

    dt = 1 / 252
    port_paths = np.zeros((n_simulations, days + 1))
    port_paths[:, 0] = portfolio_value

    for sym, w in weights.items():
        if w == 0:
            continue
        frac   = w / total_w
        prices = stock_data[sym]["prices"]
        log_r  = np.log(prices[1:] / prices[:-1])
        mu     = float(np.mean(log_r) * 252)
        sigma  = float(np.std(log_r, ddof=1) * np.sqrt(252))

        stock_paths = monte_carlo_simulation(
            start_price    = portfolio_value * frac,
            mu             = mu,
            sigma          = sigma,
            days           = days,
            n_simulations  = n_simulations,
            seed           = seed + hash(sym) % 1000,
        )
        port_paths += stock_paths

    return port_paths


#  SESSION STATE — initialise once per session


if "stock_data" not in st.session_state:
    data = {}
    for sym, info in STOCKS.items():
        seed   = hash(sym) % 10000
        prices = generate_price_history(info["base"], 60, seed=seed)
        signal, confidence, future = predict_trend(prices)
        bb_upper, bb_mid, bb_lower = calc_bollinger(prices)
        macd_line, macd_sig, macd_hist = calc_macd(prices)
        data[sym] = {
            "prices":      prices,
            "future":      future,
            "signal":      signal,
            "confidence":  confidence,
            "volatility":  calc_volatility(prices),
            "var95":       calc_var(prices, 0.95),
            "var99":       calc_var(prices, 0.99),
            "rsi":         calc_rsi(prices),
            "sharpe":      calc_sharpe(prices),
            "price":       prices[-1],
            "change_pct":  (prices[-1] - prices[-2]) / prices[-2] * 100,
            "week_change": (prices[-1] - prices[-6]) / prices[-6] * 100,
            "bb_upper":    bb_upper,
            "bb_mid":      bb_mid,
            "bb_lower":    bb_lower,
            "macd_line":   macd_line,
            "macd_signal": macd_sig,
            "macd_hist":   macd_hist,
        }
    st.session_state.stock_data  = data
    # Default portfolio weights across 10 stocks (sum = 100)
    st.session_state.portfolio   = {
        "AAPL": 12, "MSFT": 12, "GOOGL": 10, "AMZN": 12,
        "META": 10, "JPM":  12, "TSLA":   8, "NVDA": 10,
        "NFLX":  7, "AMD":   7,
    }
    st.session_state.ai_analysis = {}
    st.session_state.vix         = round(15 + random.random() * 10, 1)
    st.session_state.fear_greed  = random.randint(40, 75)
    st.session_state.bullish_pct = random.randint(50, 70)
    st.session_state.last_tick   = time.time()

# ── Live price tick simulation (every 2 s) 
if time.time() - st.session_state.last_tick > 2:
    for sym in STOCKS:
        d     = st.session_state.stock_data[sym]
        nudge = (random.random() - 0.499) * d["price"] * 0.002
        d["price"]      = d["price"] + nudge
        d["change_pct"] = d["change_pct"] + (nudge / d["price"]) * 10
    st.session_state.last_tick = time.time()



#  SIDEBAR


with st.sidebar:
    # Logo
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
      <div style='font-size:28px'>⬡</div>
      <div style='font-size:18px; font-weight:700; letter-spacing:2px; color:#e2e8f0'>PROSCIO AI</div>
      <div style='font-size:9px; color:#00d4ff; letter-spacing:3px'>MARKET RISK INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Live status bar
    st.markdown(f"""
    <div style='display:flex; gap:16px; flex-wrap:wrap; padding:8px 0; align-items:center'>
      <div><span class='live-dot'></span><span style='color:#22c55e; font-size:10px; letter-spacing:2px'>LIVE</span></div>
      <div style='font-size:10px; color:#8899aa'>VIX <span style='color:#f59e0b'>{st.session_state.vix}</span></div>
      <div style='font-size:10px; color:#8899aa'>F/G <span style='color:#22c55e'>{st.session_state.fear_greed}</span></div>
      <div style='font-size:10px; color:#8899aa'>BULL <span style='color:#22c55e'>{st.session_state.bullish_pct}%</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Stock selector
    st.markdown("<div class='section-label'>▸ SELECT EQUITY</div>", unsafe_allow_html=True)
    selected = st.selectbox(
        "Stock",
        list(STOCKS.keys()),
        format_func=lambda s: f"{s}  —  {STOCKS[s]['name']}",
        label_visibility="collapsed",
    )
    st.divider()

    # Mini equity list
    st.markdown("<div class='section-label'>▸ EQUITY UNIVERSE</div>", unsafe_allow_html=True)
    for sym, info in STOCKS.items():
        d     = st.session_state.stock_data[sym]
        up    = d["change_pct"] >= 0
        arrow = "▲" if up else "▼"
        c     = "#22c55e" if up else "#ef4444"
        sc    = SECTOR_COLORS.get(info["sector"], "#00d4ff")
        is_sel = sym == selected
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:7px 10px; border-radius:6px; margin-bottom:4px;
                    background:{"#00d4ff14" if is_sel else "transparent"};
                    border:1px solid {"#00d4ff44" if is_sel else "transparent"}'>
          <div>
            <span style='font-size:12px; font-weight:700;
                         color:{"#00d4ff" if is_sel else "#e2e8f0"}'>{sym}</span>
            <span style='font-size:8px; color:{sc}; background:{sc}22;
                         padding:1px 4px; border-radius:3px; margin-left:5px'>{info["sector"]}</span>
            <div style='font-size:9px; color:#445566; margin-top:1px'>{info["name"]}</div>
          </div>
          <div style='text-align:right'>
            <div style='font-size:11px; font-weight:700'>${d["price"]:.1f}</div>
            <div style='font-size:10px; color:{c}'>{arrow}{abs(d["change_pct"]):.2f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # AI Signal card for selected stock
    d_sel = st.session_state.stock_data[selected]
    sig_class = {
        "BUY":"signal-buy","SELL":"signal-sell","NEUTRAL":"signal-neutral",
        "OVERBOUGHT":"signal-overbought","OVERSOLD":"signal-oversold"
    }.get(d_sel["signal"], "signal-neutral")

    st.markdown("<div class='section-label'>▸ AI SIGNAL</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; padding:4px 0'>
      <span class='{sig_class}' style='font-size:13px; font-weight:700; letter-spacing:1px'>
        {d_sel["signal"]}
      </span>
      <div style='text-align:right'>
        <div style='font-size:9px; color:#556677'>CONFIDENCE</div>
        <div style='font-size:26px; font-weight:700; color:#00d4ff'>{d_sel["confidence"]}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "<div style='font-size:9px; color:#334455; letter-spacing:2px; text-align:center'>"
        "PROSCIO AI v2.0 · PYTHON + STREAMLIT</div>",
        unsafe_allow_html=True
    )



#  MAIN AREA — HEADER


d    = st.session_state.stock_data[selected]
info = STOCKS[selected]

col_logo, col_ticker = st.columns([1, 3])

with col_logo:
    st.markdown("""
    <div style='padding:8px 0'>
      <span style='font-size:24px; font-weight:700; letter-spacing:2px'>⬡ PROSCIO AI</span><br>
      <span style='font-size:9px; color:#00d4ff; letter-spacing:3px'>
        MARKET RISK &amp; EQUITY INTELLIGENCE
      </span>
    </div>
    """, unsafe_allow_html=True)

with col_ticker:
    items = []
    for sym in STOCKS:
        sd = st.session_state.stock_data[sym]
        up = sd["change_pct"] >= 0
        c  = "#22c55e" if up else "#ef4444"
        a  = "▲" if up else "▼"
        items.append(
            f"<span style='color:#8899aa'>{sym}</span> "
            f"<span style='color:{c}; font-weight:700'>${sd['price']:.2f}</span> "
            f"<span style='color:{c}; font-size:10px'>{a}{abs(sd['change_pct']):.2f}%</span>"
        )
    st.markdown(
        "<div style='background:#0a1525; border:1px solid #1e2d40; border-radius:8px;"
        "padding:11px 16px; display:flex; gap:20px; flex-wrap:wrap; align-items:center'>"
        "<span style='font-size:9px; color:#00d4ff; letter-spacing:3px; flex-shrink:0'>LIVE FEED</span>"
        + "&nbsp;&nbsp;│&nbsp;&nbsp;".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Selected stock title row
price_col = "#22c55e" if d["change_pct"] >= 0 else "#ef4444"
sec_col   = SECTOR_COLORS.get(info["sector"], "#00d4ff")
st.markdown(f"""
<div style='display:flex; align-items:baseline; gap:14px; margin-bottom:4px; flex-wrap:wrap'>
  <span style='font-size:30px; font-weight:700'>{selected}</span>
  <span style='font-size:14px; color:#556677'>{info["name"]}</span>
  <span style='font-size:11px; color:{sec_col}; background:{sec_col}22;
               padding:2px 10px; border-radius:4px'>{info["sector"]}</span>
</div>
<div style='display:flex; align-items:baseline; gap:12px; margin-bottom:20px'>
  <span style='font-size:38px; font-weight:700'>${d["price"]:.2f}</span>
  <span style='font-size:16px; color:{price_col}'>
    {"▲" if d["change_pct"]>=0 else "▼"} {abs(d["change_pct"]):.2f}% today
  </span>
</div>
""", unsafe_allow_html=True)


#  TABS


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  OVERVIEW",
    "⚠️  RISK ANALYSIS",
    "💼  PORTFOLIO",
    "🎲  MONTE CARLO",
    "🤖  AI ANALYSIS",
])


