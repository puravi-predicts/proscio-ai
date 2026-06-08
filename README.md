# ⬡ PROSCIO AI — Market Risk & Equity Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude-Sonnet%204-blueviolet?logo=anthropic&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> A professional-grade, AI-integrated equity risk intelligence dashboard built with Python, Streamlit, Plotly, and the Anthropic Claude API. Simulates live market data, computes quant finance metrics, and runs Monte Carlo portfolio simulations — all in a dark-themed terminal UI.

---

## 📸 Features at a Glance
[PROSCIO AI Dashboard Interface]
<img width="1882" height="799" alt="dashboard,png" src="https://github.com/user-attachments/assets/c59c6404-176e-49f5-a52c-b8cab6303e6a" />





| Tab | What It Does |
|-----|-------------|
| 📊 **Overview** | Live price ticker, 60-day price history with Bollinger Bands, 14-day AI forecast, MACD indicator, peer sparklines |
| ⚠️ **Risk Analysis** | VaR (95%/99%), RSI gauge, return distribution histogram, full risk matrix for all 10 stocks, volatility comparison |
| 💼 **Portfolio** | Interactive weight sliders, portfolio VaR + Sharpe, pie chart, sector concentration bar, VaR contribution per position |
| 🎲 **Monte Carlo** | GBM-based Monte Carlo simulation (single stock or full portfolio), confidence cone, final value distribution, in-depth explanation |
| 🤖 **AI Analysis** | Claude Sonnet 4 powered quantitative analysis — paste your API key and get a professional risk report for any stock |

---

## 🏗️ Tech Stack

```
Python 3.10+
├── streamlit        — web UI framework
├── plotly           — interactive charts
├── pandas           — data manipulation
├── numpy            — quantitative math
└── anthropic        — Claude AI API (Sonnet 4)
```

---

## 📈 Equity Universe (10 Stocks)

| Ticker | Company | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| GOOGL | Alphabet Inc. | Technology |
| **AMZN** | **Amazon.com Inc.** | **E-Commerce/Cloud** |
| **META** | **Meta Platforms** | **Social Media** |
| JPM | JPMorgan Chase | Finance |
| TSLA | Tesla Inc. | EV/Auto |
| NVDA | NVIDIA Corp. | Semiconductors |
| NFLX | Netflix Inc. | Streaming |
| AMD | Advanced Micro Devices | Semiconductors |

> All prices are **simulated** using Geometric Brownian Motion seeded from realistic base prices. No live data API required.

---

## 📐 Quant Metrics Computed

| Metric | Formula / Method |
|--------|-----------------|
| **Annualized Volatility** | `σ_daily × √252` from log-returns |
| **VaR (95% & 99%)** | Historical simulation — percentile of 60-day return distribution |
| **RSI (14-day)** | Wilder's Relative Strength Index |
| **Sharpe Ratio** | `(μ_annual − r_f) / σ_annual`, r_f = 5% |
| **MACD** | EMA(12) − EMA(26), Signal = EMA(9), Histogram |
| **Bollinger Bands** | SMA(20) ± 2σ |
| **Momentum Signal** | 10-day vs 20-day avg price with RSI filter |
| **Monte Carlo VaR** | 5th percentile of GBM-simulated final values |

---

## 🎲 Monte Carlo Simulation

The Monte Carlo engine uses **Geometric Brownian Motion (GBM)**, the same model underlying Black-Scholes options pricing:

```
S(t+dt) = S(t) · exp[ (μ - ½σ²)·dt + σ·√dt·Z ]
```

Where:
- `μ` = annualised drift estimated from historical log-returns
- `σ` = annualised volatility from historical data
- `Z ~ N(0,1)` = standard normal random draw
- `dt = 1/252` = one trading day

### Two Simulation Modes

1. **Single Stock** — simulate any individual equity from the universe
2. **Portfolio** — simulate all 10 stocks weighted by your portfolio sliders, aggregated into total portfolio value

### Output Statistics
- Mean & Median final value
- 5th / 95th percentile outcomes
- Probability of profit
- Probability of >10% loss
- Spaghetti path chart (up to 200 paths rendered)
- Confidence cone (±1σ, ±2σ bands)
- Final value distribution histogram with VaR marked

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/proscio-ai.git
cd proscio-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

### 5. (Optional) Enable AI Analysis

Head to the **🤖 AI Analysis** tab, paste your Anthropic API key (`sk-ant-...`), and click **⚡ RUN AI ANALYSIS**. Get your key at [console.anthropic.com](https://console.anthropic.com).

---

## 📁 Project Structure

```
proscio-ai/
├── app.py              # Main Streamlit application (single-file)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧠 Interview Prep — Key Concepts

This project is designed to demonstrate and explain core quant finance concepts. 

- **VaR** — Historical simulation, Basel III context, real-world interpretation
- **Monte Carlo** — GBM derivation, parameter estimation, limitations (fat tails, GBM assumptions), comparison with Historical VaR
- **Sharpe Ratio** — Risk-adjusted return, interpretation scale
- **MACD** — Crossover signals, histogram divergence
- **AI Signal** — Momentum + RSI fusion logic explained

> The Monte Carlo tab alone covers topics from **FRM Part 2**, **CFA Level 2**, and standard **quant finance interviews** at banks and hedge funds.

---

## ⚙️ Configuration

Key constants at the top of `app.py`:

```python
# Add or remove stocks here
STOCKS = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "base": 189.5},
    # ... add more as needed
}

# Adjust simulation defaults
mc_sims  = 500   # number of Monte Carlo paths
mc_days  = 252   # forecast horizon (trading days)
```

---

## ⚠️ Disclaimer

All price data in this application is **synthetically generated** using Geometric Brownian Motion and does not represent real market data. This project is for **educational and portfolio demonstration purposes only** and should not be used for actual investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## Built With

- [Streamlit](https://streamlit.io) — rapid Python web apps
- [Plotly](https://plotly.com/python/) — interactive data visualization
- [Anthropic Claude](https://www.anthropic.com) — AI analysis engine
- [NumPy](https://numpy.org) / [Pandas](https://pandas.pydata.org) — quantitative computing

---

<div align="center">
  <b>⬡ PROSCIO AI</b> · 💻 <code>Python + Streamlit</code> · 📊 <code>Quant Analytics</code> · 🤖 <code>Claude Sonnet Fusion</code>
</div>
