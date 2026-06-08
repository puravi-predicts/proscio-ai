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



#  TAB 1 — OVERVIEW

with tab1:

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("CURRENT PRICE",    f"${d['price']:.2f}",       f"{d['change_pct']:+.2f}% today")
    with c2: st.metric("RSI (14D)",         f"{d['rsi']:.1f}",          "Overbought" if d['rsi']>70 else "Oversold" if d['rsi']<30 else "Neutral")
    with c3: st.metric("SHARPE RATIO",      f"{d['sharpe']:.2f}",       "Excellent" if d['sharpe']>1 else "Moderate")
    with c4: st.metric("WEEKLY CHANGE",     f"{d['week_change']:+.2f}%","5-day return")
    with c5: st.metric("VOLATILITY (ANN)",  f"{d['volatility']:.1f}%", "Historical 60d")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price history + AI forecast chart ──
    st.markdown("<div class='section-label'>▸ PRICE HISTORY + 14-DAY AI FORECAST</div>",
                unsafe_allow_html=True)

    dates_hist   = [datetime.today() - timedelta(days=60-i) for i in range(60)]
    dates_future = [datetime.today() + timedelta(days=i+1)  for i in range(14)]

    fig = go.Figure()

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=dates_hist + dates_hist[::-1],
        y=list(d["bb_upper"]) + list(d["bb_lower"][::-1]),
        fill="toself", fillcolor="rgba(0,212,255,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="Bollinger Band",
    ))
    fig.add_trace(go.Scatter(x=dates_hist, y=d["bb_upper"],
        line=dict(color="#00d4ff", width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=dates_hist, y=d["bb_lower"],
        line=dict(color="#00d4ff", width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=dates_hist, y=d["bb_mid"],
        line=dict(color="#00d4ff", width=1), name="SMA-20"))

    # Historical price
    fig.add_trace(go.Scatter(
        x=dates_hist, y=d["prices"],
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
        name="Historical",
    ))

    # AI forecast
    fig.add_trace(go.Scatter(
        x=[dates_hist[-1]] + dates_future,
        y=[d["prices"][-1]] + list(d["future"]),
        line=dict(color="#f59e0b", width=2, dash="dash"),
        name="AI Forecast (14d)",
    ))

    fig.add_vline(x=dates_hist[-1], line_dash="dot", line_color="#ffffff")
    fig.add_annotation(
        x=dates_hist[-1], y=float(np.nanmax(d["prices"])) * 0.98,
        text="AI FORECAST →", font=dict(color="#ffffff", size=9),
        showarrow=False, xshift=10,
    )
    fig.update_layout(
        **PLOT_THEME, height=320, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── MACD ──
    st.markdown("<div class='section-label'>▸ MACD INDICATOR</div>", unsafe_allow_html=True)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(
        x=dates_hist, y=d["macd_hist"],
        marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in d["macd_hist"]],
        name="Histogram", opacity=0.6,
    ))
    fig_macd.add_trace(go.Scatter(x=dates_hist, y=d["macd_line"],
        line=dict(color="#00d4ff", width=1.5), name="MACD"))
    fig_macd.add_trace(go.Scatter(x=dates_hist, y=d["macd_signal"],
        line=dict(color="#f59e0b", width=1.5), name="Signal"))
    fig_macd.add_hline(y=0, line_color="#ffffff")
    fig_macd.update_layout(**PLOT_THEME, height=160,
                           legend=dict(orientation="h", font=dict(size=9)))
    st.plotly_chart(fig_macd, use_container_width=True)

    # ── Peer sparklines ──
    st.markdown("<div class='section-label'>▸ PEER COMPARISON (SPARKLINES)</div>",
                unsafe_allow_html=True)
    peers = [s for s in STOCKS if s != selected]
    cols  = st.columns(min(len(peers), 9))
    for col, sym in zip(cols, peers):
        sd = st.session_state.stock_data[sym]
        up = sd["week_change"] >= 0
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            y=sd["prices"][-20:], mode="lines",
            line=dict(color="#22c55e" if up else "#ef4444", width=1.5),
        ))
        fig_s.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a", height=70,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
        )
        with col:
            st.markdown(
                f"<div style='font-size:10px; font-weight:700; margin-bottom:2px'>{sym} "
                f"<span style='color:{'#22c55e' if up else '#ef4444'}; font-size:9px'>"
                f"{'▲' if up else '▼'}{abs(sd['week_change']):.1f}%</span></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_s, use_container_width=True)



#  TAB 2 — RISK ANALYSIS

with tab2:

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("VALUE AT RISK (95%)",    f"{d['var95']:.2f}%",      "Max loss 1-in-20 days",  delta_color="inverse")
    with c2: st.metric("VALUE AT RISK (99%)",    f"{d['var99']:.2f}%",      "Extreme tail risk",       delta_color="inverse")
    with c3: st.metric("ANNUALIZED VOLATILITY",  f"{d['volatility']:.1f}%", "Historical 60-day",       delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    col_gauge, col_dist = st.columns([1, 2])

    with col_gauge:
        st.markdown("<div class='section-label'>▸ RSI GAUGE</div>", unsafe_allow_html=True)
        bar_color = "#22c55e" if d["rsi"] < 30 else "#ef4444" if d["rsi"] > 70 else "#f59e0b"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=d["rsi"],
            number={"font": {"color": "#e2e8f0", "size": 32, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#556677", "tickfont": {"size": 9}},
                "bar":  {"color": bar_color},
                "steps": [
                    {"range": [0,  30],  "color": "#22c55e"},
                    {"range": [30, 70],  "color": "#1e2d40"},
                    {"range": [70, 100], "color": "#ef4422"},
                ],
                "threshold": {"line": {"color": "#00d4ff", "width": 2}, "value": d["rsi"]},
                "bgcolor": "#0a1525", "bordercolor": "#1e2d40",
            },
            title={"text": "RSI (14-day)", "font": {"color": "#556677", "size": 10}},
        ))
        fig_gauge.update_layout(paper_bgcolor="#0a1525", font_color="#8899aa",
                                height=220, margin=dict(l=20, r=20, t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        if d["rsi"] < 30:  st.success("✅ Oversold — potential buy zone")
        if d["rsi"] > 70:  st.warning("⚠️ Overbought — risk of reversal")

    with col_dist:
        st.markdown("<div class='section-label'>▸ RETURN DISTRIBUTION & VaR THRESHOLDS</div>",
                    unsafe_allow_html=True)
        returns = (d["prices"][1:] - d["prices"][:-1]) / d["prices"][:-1] * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=returns, nbinsx=20, name="Daily Returns",
            marker_color="#78bec9",
            marker_line_color="#063d46", marker_line_width=1,
        ))
        fig_dist.add_vline(x=-d["var95"], line_color="#f59e0b", line_dash="dash",
                           annotation_text="VaR 95%",
                           annotation_font_color="#f59e0b", annotation_font_size=9)
        fig_dist.add_vline(x=-d["var99"], line_color="#ef4444", line_dash="dash",
                           annotation_text="VaR 99%",
                           annotation_font_color="#ef4444", annotation_font_size=9)
        fig_dist.update_layout(**PLOT_THEME, height=220,
                               xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig_dist, use_container_width=True)

    # Full risk matrix
    st.markdown("<div class='section-label'>▸ RISK METRICS MATRIX — ALL EQUITIES</div>",
                unsafe_allow_html=True)
    rows = []
    for sym, sinfo in STOCKS.items():
        sd = st.session_state.stock_data[sym]
        rows.append({
            "Symbol":      sym,
            "Name":        sinfo["name"],
            "Sector":      sinfo["sector"],
            "Price":       f"${sd['price']:.2f}",
            "Vol %":       f"{sd['volatility']:.1f}",
            "VaR 95%":     f"{sd['var95']:.2f}",
            "VaR 99%":     f"{sd['var99']:.2f}",
            "RSI":         f"{sd['rsi']:.0f}",
            "Sharpe":      f"{sd['sharpe']:.2f}",
            "Signal":      sd["signal"],
            "Confidence":  f"{sd['confidence']}%",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style
          .map(
              lambda v: ("color: #22c55e" if v == "BUY"
                         else "color: #ef4444" if v == "SELL"
                         else "color: #f59e0b" if "OVER" in str(v)
                         else ""),
              subset=["Signal"]
          )
          .set_properties(**{
              "background-color": "#0d1b2a",
              "color": "#e2e8f0",
              "border": "1px solid #1e2d40",
          }),
        use_container_width=True,
        hide_index=True,
    )

    # Volatility bar chart
    st.markdown("<div class='section-label'>▸ VOLATILITY COMPARISON — ALL EQUITIES</div>",
                unsafe_allow_html=True)
    syms  = list(STOCKS.keys())
    vols  = [st.session_state.stock_data[s]["volatility"] for s in syms]
    bar_colors = ["#22c55e" if v < 20 else "#f59e0b" if v < 35 else "#ef4444" for v in vols]

    fig_vol = go.Figure(go.Bar(
        x=syms, y=vols,
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in vols],
        textposition="outside",
        textfont=dict(color="#8899aa", size=10),
    ))
    fig_vol.update_layout(**PLOT_THEME, height=230, yaxis_title="Annualized Volatility (%)")
    st.plotly_chart(fig_vol, use_container_width=True)



#  TAB 3 — PORTFOLIO

with tab3:

    st.markdown(
        "<div class='section-label'>▸ PORTFOLIO RISK ENGINE — DRAG SLIDERS TO REBALANCE</div>",
        unsafe_allow_html=True,
    )

    col_sliders, col_metrics = st.columns([2, 1])

    with col_sliders:
        new_weights = {}
        for sym in STOCKS:
            new_weights[sym] = st.slider(
                f"{sym}  —  {STOCKS[sym]['name']}",
                min_value=0, max_value=60,
                value=st.session_state.portfolio[sym],
                step=1, key=f"slider_{sym}",
            )
        st.session_state.portfolio = new_weights

    with col_metrics:
        total_var = sum(
            st.session_state.stock_data[s]["var95"] * w / 100
            for s, w in new_weights.items()
        )
        total_w = sum(new_weights.values())
        weighted_sharpe = sum(
            st.session_state.stock_data[s]["sharpe"] * new_weights[s] / 100
            for s in STOCKS
        )
        st.metric("PORTFOLIO VaR (95%)", f"{total_var:.3f}%", "Weighted average")
        st.metric("TOTAL ALLOCATION",    f"{total_w}%",
                  "✅ Balanced" if total_w == 100 else f"{'Over' if total_w>100 else 'Under'} by {abs(total_w-100)}%",
                  delta_color="off")
        st.metric("WEIGHTED SHARPE",     f"{weighted_sharpe:.2f}", "Portfolio quality")

    # Pie + sector bar
    col_pie, col_sector = st.columns(2)

    with col_pie:
        st.markdown("<div class='section-label'>▸ POSITION ALLOCATION</div>",
                    unsafe_allow_html=True)
        labels      = [s for s in STOCKS if new_weights[s] > 0]
        values      = [new_weights[s] for s in labels]
        pie_colors  = [SECTOR_COLORS.get(STOCKS[s]["sector"], "#00d4ff") for s in labels]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=pie_colors, line=dict(color="#070d14", width=2)),
            hole=0.5,
            textfont=dict(family="IBM Plex Mono", size=11, color="#e2e8f0"),
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0a1525", showlegend=True, height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(font=dict(family="IBM Plex Mono", size=9, color="#8899aa")),
            annotations=[dict(
                text=f"{total_w}%<br>TOTAL",
                font=dict(size=16, color="#e2e8f0", family="IBM Plex Mono"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_sector:
        st.markdown("<div class='section-label'>▸ SECTOR CONCENTRATION</div>",
                    unsafe_allow_html=True)
        sector_totals: dict = {}
        for sym, w in new_weights.items():
            sec = STOCKS[sym]["sector"]
            sector_totals[sec] = sector_totals.get(sec, 0) + w

        fig_sec = go.Figure(go.Bar(
            x=list(sector_totals.values()),
            y=list(sector_totals.keys()),
            orientation="h",
            marker_color=[SECTOR_COLORS.get(s, "#00d4ff") for s in sector_totals],
            text=[f"{v}%" for v in sector_totals.values()],
            textposition="outside",
            textfont=dict(size=10, color="#8899aa"),
        ))
        fig_sec.update_layout(**PLOT_THEME, height=300, xaxis_title="Portfolio Weight (%)")
        st.plotly_chart(fig_sec, use_container_width=True)

    # Risk contribution per position
    st.markdown("<div class='section-label'>▸ VaR RISK CONTRIBUTION PER POSITION</div>",
                unsafe_allow_html=True)
    contrib_rows = [
        {
            "Symbol":          sym,
            "Weight %":        w,
            "VaR Contribution": round(st.session_state.stock_data[sym]["var95"] * w / 100, 4),
            "Signal":          st.session_state.stock_data[sym]["signal"],
        }
        for sym, w in new_weights.items() if w > 0
    ]
    if contrib_rows:
        cdf = pd.DataFrame(contrib_rows)
        fig_contrib = go.Figure(go.Bar(
            x=cdf["Symbol"],
            y=cdf["VaR Contribution"],
            marker_color=[SECTOR_COLORS.get(STOCKS[s]["sector"], "#00d4ff") for s in cdf["Symbol"]],
            text=[f"{v:.4f}%" for v in cdf["VaR Contribution"]],
            textposition="outside",
            textfont=dict(size=9, color="#8899aa"),
        ))
        fig_contrib.update_layout(**PLOT_THEME, height=230, yaxis_title="VaR Contribution (%)")
        st.plotly_chart(fig_contrib, use_container_width=True)



#  TAB 4 — MONTE CARLO SIMULATION

with tab4:

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a1525,#0d1b2a);
                border:1px solid #a78bfa33; border-radius:10px;
                padding:20px; margin-bottom:20px'>
      <div class='section-label' style='color:#a78bfa'>▸ MONTE CARLO SIMULATION ENGINE</div>
      <div style='font-size:12px; color:#556677; line-height:1.7'>
        Geometric Brownian Motion (GBM) — the industry-standard stochastic model for equity prices.<br>
        Run thousands of randomised future price paths and visualise the probability distribution of outcomes.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls 
    mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)

    with mc_col1:
        mc_mode = st.selectbox(
            "Simulation Mode",
            ["Single Stock", "Portfolio"],
            help="Single Stock: simulate one equity. Portfolio: simulate entire weighted portfolio.",
        )

    with mc_col2:
        mc_sims = st.selectbox(
            "Number of Simulations",
            [100, 250, 500, 1000],
            index=2,
            help="More paths = more accurate distribution, but slower render.",
        )

    with mc_col3:
        mc_days = st.selectbox(
            "Forecast Horizon",
            [30, 63, 126, 252],
            index=3,
            format_func=lambda x: {30: "1 Month", 63: "3 Months", 126: "6 Months", 252: "1 Year"}[x],
        )

    with mc_col4:
        mc_capital = st.number_input(
            "Portfolio Value ($)",
            min_value=1_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            help="Starting capital for portfolio simulation.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run simulation 
    if mc_mode == "Single Stock":
        mc_stock = st.selectbox(
            "Select Stock to Simulate",
            list(STOCKS.keys()),
            format_func=lambda s: f"{s} — {STOCKS[s]['name']}",
        )
        sd_mc   = st.session_state.stock_data[mc_stock]
        log_r   = np.log(sd_mc["prices"][1:] / sd_mc["prices"][:-1])
        mc_mu   = float(np.mean(log_r) * 252)
        mc_sig  = float(np.std(log_r, ddof=1) * np.sqrt(252))
        paths   = monte_carlo_simulation(
            start_price   = sd_mc["price"],
            mu            = mc_mu,
            sigma         = mc_sig,
            days          = mc_days,
            n_simulations = mc_sims,
        )
        start_val = sd_mc["price"]
        label     = mc_stock
        currency  = "$"

    else:  # Portfolio mode
        weights_used = {s: st.session_state.portfolio[s] for s in STOCKS}
        paths = mc_portfolio_simulation(
            weights         = weights_used,
            stock_data      = st.session_state.stock_data,
            portfolio_value = mc_capital,
            days            = mc_days,
            n_simulations   = mc_sims,
        )
        start_val = mc_capital
        label     = "Portfolio"
        currency  = "$"

    # ── Key statistics 
    final_vals     = paths[:, -1]
    pct_returns    = (final_vals - start_val) / start_val * 100
    mc_mean        = float(np.mean(final_vals))
    mc_median      = float(np.median(final_vals))
    mc_p5          = float(np.percentile(final_vals, 5))
    mc_p95         = float(np.percentile(final_vals, 95))
    mc_var_mc      = float(np.percentile(pct_returns, 5))   # 95% VaR via MC
    prob_profit    = float(np.mean(final_vals > start_val) * 100)
    prob_loss_10   = float(np.mean(pct_returns < -10) * 100)

    # ── Stats row 
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1: st.metric("MEAN OUTCOME",    f"{currency}{mc_mean:,.0f}", f"{(mc_mean-start_val)/start_val*100:+.1f}%")
    with s2: st.metric("MEDIAN OUTCOME",  f"{currency}{mc_median:,.0f}", f"{(mc_median-start_val)/start_val*100:+.1f}%")
    with s3: st.metric("5th PERCENTILE",  f"{currency}{mc_p5:,.0f}",    f"{(mc_p5-start_val)/start_val*100:+.1f}%", delta_color="inverse")
    with s4: st.metric("95th PERCENTILE", f"{currency}{mc_p95:,.0f}",   f"{(mc_p95-start_val)/start_val*100:+.1f}%")
    with s5: st.metric("PROB. OF PROFIT", f"{prob_profit:.1f}%",         "Price > Start")
    with s6: st.metric("PROB. LOSS > 10%",f"{prob_loss_10:.1f}%",        "Downside risk", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Paths chart 
    st.markdown(
        f"<div class='section-label'>▸ {mc_sims} SIMULATED PRICE PATHS — {label} ({mc_days}D HORIZON)</div>",
        unsafe_allow_html=True
    )

    future_dates = [datetime.today() + timedelta(days=i) for i in range(mc_days + 1)]
    fig_mc = go.Figure()

    # Draw a sample of paths (max 200 for performance)
    sample_idx = np.random.choice(mc_sims, min(200, mc_sims), replace=False)
    for i in sample_idx:
        fig_mc.add_trace(go.Scatter(
            x=future_dates, y=paths[i],
            mode="lines",
            line=dict(color="rgba(0,212,255,0.06)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Percentile bands
    p5_path   = np.percentile(paths, 5,  axis=0)
    p25_path  = np.percentile(paths, 25, axis=0)
    p50_path  = np.percentile(paths, 50, axis=0)
    p75_path  = np.percentile(paths, 75, axis=0)
    p95_path  = np.percentile(paths, 95, axis=0)

    # P5–P95 fill
    fig_mc.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(p95_path) + list(p5_path[::-1]),
        fill="toself", fillcolor="rgba(167,139,250,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th Percentile",
    ))
    # P25–P75 fill
    fig_mc.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(p75_path) + list(p25_path[::-1]),
        fill="toself", fillcolor="rgba(167,139,250,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th Percentile",
    ))

    fig_mc.add_trace(go.Scatter(x=future_dates, y=p5_path,
        line=dict(color="#ef4444", width=1.5, dash="dot"), name="5th pct (Worst)"))
    fig_mc.add_trace(go.Scatter(x=future_dates, y=p50_path,
        line=dict(color="#a78bfa", width=2.5), name="Median"))
    fig_mc.add_trace(go.Scatter(x=future_dates, y=p95_path,
        line=dict(color="#22c55e", width=1.5, dash="dot"), name="95th pct (Best)"))

    # Start price reference line
    fig_mc.add_hline(y=start_val, line_color="#ffff22", line_dash="dash",
                     annotation_text="Start Price",
                     annotation_font_color="#fffff4", annotation_font_size=9)

    fig_mc.update_layout(
        **PLOT_THEME, height=380, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        yaxis_title=f"{'Price ($)' if mc_mode == 'Single Stock' else 'Portfolio Value ($)'}",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # ── Final value distribution 
    col_hist, col_cone = st.columns(2)

    with col_hist:
        st.markdown("<div class='section-label'>▸ FINAL VALUE DISTRIBUTION</div>",
                    unsafe_allow_html=True)
        fig_mchist = go.Figure()
        fig_mchist.add_trace(go.Histogram(
            x=final_vals, nbinsx=40,
            marker_color="rgba(167,139,250,0.4)",
            marker_line_color="#a78bfa", marker_line_width=1,
            name="Simulation Outcomes",
        ))
        fig_mchist.add_vline(x=start_val,  line_color="#fff444", line_dash="dash",
                             annotation_text="Start", annotation_font_color="#ffff88", annotation_font_size=9)
        fig_mchist.add_vline(x=mc_mean,    line_color="#00d4ff",   line_dash="dash",
                             annotation_text="Mean",  annotation_font_color="#00d4ff",  annotation_font_size=9)
        fig_mchist.add_vline(x=mc_p5,      line_color="#ef4444",   line_dash="dot",
                             annotation_text="VaR 95%", annotation_font_color="#ef4444", annotation_font_size=9)
        fig_mchist.update_layout(**PLOT_THEME, height=280,
                                 xaxis_title="Final Value ($)", yaxis_title="Count")
        st.plotly_chart(fig_mchist, use_container_width=True)

    with col_cone:
        st.markdown("<div class='section-label'>▸ CONFIDENCE CONE (MEAN ± σ BANDS)</div>",
                    unsafe_allow_html=True)
        mean_path = np.mean(paths, axis=0)
        std_path  = np.std(paths,  axis=0)

        fig_cone = go.Figure()
        # ±2σ
        fig_cone.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(mean_path + 2*std_path) + list((mean_path - 2*std_path)[::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.04)",
            line=dict(color="rgba(0,0,0,0)"), name="±2σ Band",
        ))
        # ±1σ
        fig_cone.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(mean_path + std_path) + list((mean_path - std_path)[::-1]),
            fill="toself", fillcolor="rgba(0,212,255,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="±1σ Band",
        ))
        fig_cone.add_trace(go.Scatter(x=future_dates, y=mean_path,
            line=dict(color="#00d4ff", width=2.5), name="Mean Path"))
        fig_cone.add_hline(y=start_val, line_color="#fffff2", line_dash="dash")

        fig_cone.update_layout(
            **PLOT_THEME, height=280, showlegend=True,
            legend=dict(orientation="h", font=dict(size=9)),
        )
        st.plotly_chart(fig_cone, use_container_width=True)

    # ── Monte Carlo explainer 
    st.divider()
    st.markdown("<div class='section-label'>▸ WHAT IS MONTE CARLO SIMULATION?</div>",
                unsafe_allow_html=True)

    with st.expander("📖 Monte Carlo — Full Explanation (click to expand)", expanded=False):
        st.markdown(f"""
### 🎲 Monte Carlo Simulation in Finance

**Monte Carlo simulation** is a computational technique that uses repeated random sampling
to model the probability distribution of an uncertain outcome — in this case, future stock prices.

---

#### The Math: Geometric Brownian Motion (GBM)

This dashboard uses **GBM**, the foundation of the Black-Scholes options pricing model:

```
S(t+dt) = S(t) · exp[ (μ - ½σ²)·dt + σ·√dt·Z ]
```

| Symbol | Meaning |
|--------|---------|
| `S(t)` | Stock price at time t |
| `μ` (mu) | Estimated annualised drift (mean log-return × 252) |
| `σ` (sigma) | Estimated annualised volatility (std log-return × √252) |
| `dt` | Time step = 1/252 (one trading day) |
| `Z` | Standard normal random draw: Z ~ N(0,1) |

For **{label}**, the estimated parameters are:
if mc_mode == "Single Stock":
    st.markdown(f"- **μ (drift):** {mc_mu*100:.2f}%")
else:
    st.markdown(f"- **μ (drift):** Weighted composite")
- **σ (volatility):** {mc_sig*100 if mc_mode == "Single Stock" else "Weighted composite":.2f}{'%' if mc_mode == 'Single Stock' else ''}

---

#### How It Works — Step by Step

1. **Estimate parameters** from 60 days of historical prices (daily log-returns → μ, σ)
2. **Sample randomness** — draw Z from N(0,1) for each day of each simulation path
3. **Propagate prices** — apply the GBM formula repeatedly across {mc_days} trading days
4. **Repeat {mc_sims:,} times** to get a distribution of possible futures
5. **Analyse outcomes** — extract percentiles, VaR, probability of profit, etc.

---

#### What the Charts Tell You

| Chart | Insight |
|-------|---------|
| **Spaghetti paths** | Each thin line = one possible future. Wide spread = high uncertainty |
| **Confidence cone** | ±1σ and ±2σ bands show where ~68% and ~95% of outcomes land |
| **Final distribution** | Histogram of outcomes at day {mc_days}; right skew = upside potential |
| **5th percentile (VaR)** | The worst 5% scenario — your Monte Carlo VaR |

---

#### Current Simulation Results

| Metric | Value |
|--------|-------|
| Starting value | {currency}{start_val:,.2f} |
| Mean final value | {currency}{mc_mean:,.2f} ({(mc_mean-start_val)/start_val*100:+.1f}%) |
| Median final value | {currency}{mc_median:,.2f} ({(mc_median-start_val)/start_val*100:+.1f}%) |
| 5th pct (MC VaR 95%) | {currency}{mc_p5:,.2f} ({mc_var_mc:+.1f}%) |
| 95th pct (Upside) | {currency}{mc_p95:,.2f} ({(mc_p95-start_val)/start_val*100:+.1f}%) |
| Probability of Profit | {prob_profit:.1f}% |
| Prob. of >10% Loss | {prob_loss_10:.1f}% |

---

#### Monte Carlo VaR vs Historical VaR

| Method | Approach | Strength |
|--------|----------|----------|
| **Historical VaR** (Tab 2) | Uses actual past return percentiles | Captures real fat-tails |
| **Monte Carlo VaR** (this tab) | Simulates future paths via GBM | Forward-looking, scenario-flexible |

Both are used in practice. Banks use Monte Carlo for **complex derivatives** where
closed-form solutions don't exist. For equities, they complement each other:
historical VaR anchors you in reality; Monte Carlo shows where you could go.

---

        """)

    st.markdown("""
    <div style='background:#0d1b2a; border:1px solid #a78bfa33; border-radius:8px;
                padding:16px; margin-top:12px; font-size:11px; color:#556677; line-height:1.8'>
      <span style='color:#a78bfa; font-weight:700'>⚠️ DISCLAIMER:</span>
      Monte Carlo simulations are probabilistic models, not predictions.
      Past volatility and drift do not guarantee future performance.
      GBM assumes constant volatility and log-normal returns — real markets exhibit fat tails, jumps, and regime changes.
      This tool is for <span style='color:#e2e8f0'>educational and interview preparation purposes only.</span>
    </div>
    """, unsafe_allow_html=True)



#  TAB 5 — AI ANALYSIS

with tab5:

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0a1525,#0d1b2a);
                border:1px solid #00d4ff33; border-radius:10px;
                padding:20px; margin-bottom:20px'>
      <div class='section-label'>▸ CLAUDE AI — QUANTITATIVE ANALYST</div>
      <div style='font-size:13px; color:#556677'>
        Deep analysis for
        <span style='color:#00d4ff'>{selected}</span>
        — {info["name"]} · {info["sector"]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Get your free key at console.anthropic.com — never stored or logged",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("⚡ RUN AI ANALYSIS", use_container_width=True)
    with col_info:
        st.markdown("""
        <div style='font-size:10px; color:#334455; padding:10px'>
          MODEL: claude-sonnet-4 &nbsp;│&nbsp;
          INPUT: 60d OHLCV · VaR · RSI · Sharpe · Momentum &nbsp;│&nbsp;
          METHOD: Quant + Sentiment Fusion
        </div>
        """, unsafe_allow_html=True)

    if run_analysis:
        if not api_key:
            st.error("Please enter your Anthropic API key above.")
        else:
            with st.spinner("🤖 Proscio AI is analyzing market data..."):
                try:
                    client = Anthropic(api_key=api_key)
                    prompt = (
                        f"Analyze {info['name']} ({selected}) — Sector: {info['sector']}\n\n"
                        f"Current Price:         ${d['price']:.2f}\n"
                        f"Daily Change:          {d['change_pct']:.2f}%\n"
                        f"Weekly Change:         {d['week_change']:.2f}%\n"
                        f"Annualized Volatility: {d['volatility']:.1f}%\n"
                        f"VaR (95%):             {d['var95']:.2f}%\n"
                        f"VaR (99%):             {d['var99']:.2f}%\n"
                        f"RSI (14-day):          {d['rsi']:.1f}\n"
                        f"Sharpe Ratio:          {d['sharpe']:.2f}\n"
                        f"AI Signal:             {d['signal']} ({d['confidence']}% confidence)\n\n"
                        "Provide: risk assessment, trend analysis, key risk factors, "
                        "and a clear recommendation for a portfolio manager."
                    )
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1024,
                        system=(
                            "You are a senior quantitative analyst and risk manager at a "
                            "top-tier investment bank. Provide sharp, data-driven market "
                            "analysis. Be professional, cite the specific numbers from the "
                            "data provided, and give actionable insights. "
                            "Use emoji section headers (📊 📉 ⚠️ ✅ 🎯). Keep to 280-320 words."
                        ),
                        messages=[{"role": "user", "content": prompt}],
                    )
                    st.session_state.ai_analysis[selected] = message.content[0].text
                except Exception as e:
                    st.error(f"API Error: {e}")

    if selected in st.session_state.ai_analysis:
        st.markdown(
            "<div style='background:#060e18; border:1px solid #1e2d40; border-radius:8px;"
            "padding:20px; font-size:13px; line-height:1.8; color:#b0c4d8;"
            f"white-space:pre-wrap; margin-top:16px'>{st.session_state.ai_analysis[selected]}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#334455'>
          <div style='font-size:48px; margin-bottom:16px'>◈</div>
          <div style='font-size:12px; letter-spacing:2px'>ENTER API KEY AND CLICK "RUN AI ANALYSIS"</div>
          <div style='font-size:10px; margin-top:8px; color:#223344'>
            Powered by Claude Sonnet · Professional quantitative analysis
          </div>
        </div>
        """, unsafe_allow_html=True)

  
    st.divider()
    st.markdown("<div class='section-label'>▸ METRICS EXPLAINED (FOR INTERVIEWS)</div>",
                unsafe_allow_html=True)

    with st.expander("What is Value at Risk (VaR)?"):
        st.markdown("""
**VaR 95%** means: *"In the worst 5% of trading days, I expect to lose at least X% of the position."*

This dashboard uses **Historical Simulation** — 60 days of returns sorted ascending, 5th percentile taken.
Standard under **Basel III** at major banks.

> If VaR 95% = 2.1% on a ₹10,00,000 position → you could lose ₹21,000 on a bad day (1-in-20 chance).
        """)

    with st.expander("How does the AI Trend Signal work?"):
        st.markdown("""
A **two-factor momentum + RSI fusion model**:

1. **Momentum** = (10-day avg price − 20-day avg price) / 20-day avg price
2. **RSI** filters false signals — don't BUY into already-overbought momentum

| Condition | Signal |
|-----------|--------|
| Momentum > 2% AND RSI < 70 | **BUY** |
| Momentum < −2% AND RSI > 30 | **SELL** |
| RSI > 75 | **OVERBOUGHT** |
| RSI < 25 | **OVERSOLD** |
| Everything else | **NEUTRAL** |

The **14-day forecast** is a momentum-biased stochastic drift model (simplified Monte Carlo).
        """)

    with st.expander("What is the Sharpe Ratio?"):
        st.markdown("""
**Sharpe = (Annual Return − Risk-Free Rate) / Annual Volatility**

Risk-free rate = 5% (US/India T-bill proxy). Annualised from daily returns.

| Sharpe    | Interpretation |
|-----------|---------------|
| > 2.0     | Exceptional   |
| 1.0–2.0   | Good          |
| 0–1.0     | Acceptable    |
| < 0       | Below risk-free assets |
        """)

    with st.expander("What is MACD?"):
        st.markdown("""
**MACD = EMA(12) − EMA(26)**  |  **Signal = EMA(MACD, 9)**

- When MACD crosses **above** the signal line → bullish momentum
- When MACD crosses **below** the signal line → bearish momentum
- Histogram = MACD − Signal (positive = bullish, negative = bearish)
        """)

    with st.expander("What is Monte Carlo Simulation?"):
        st.markdown("""
Monte Carlo uses **Geometric Brownian Motion (GBM)** to simulate thousands of random future price paths.

```
S(t+dt) = S(t) · exp[ (μ - ½σ²)·dt + σ·√dt·Z ],   Z ~ N(0,1)
```

- **μ** = estimated annual drift from historical data
- **σ** = estimated annual volatility from historical data
- Each run = one possible future; run 500 times → a probability distribution

**Why it matters:** Used for options pricing (Black-Scholes), portfolio stress testing,
and regulatory capital calculation (FRTB). The 5th percentile of simulated outcomes = Monte Carlo VaR.
        """)


# ── Auto-refresh to simulate live data 
time.sleep(0.5)
st.rerun()