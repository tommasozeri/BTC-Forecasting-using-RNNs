(Sources and basic code was taken from the UNIBO Intensive Program in Artifical Intelligence by professors Maurizio Morini, Umberto Cherubini and Giovanni Dalla Lunga)

# (Trying to) Forecast Bitcoin using Deep Neural Networks

> *An ablation study comparing on-chain-only vs. multi-source deep learning models for Bitcoin return forecasting, with a simulated trading strategy evaluation.*

**Author:** Tommaso Zeri — March 2026

---

## Overview

This project investigates whether incorporating **off-chain data** (macroeconomic indicators, sentiment, derivatives) meaningfully improves the predictive accuracy of Bitcoin forecasting models built on deep neural networks.

The methodology is structured as a rigorous **ablation study** with two experimental configurations:

| Run | Feature Set | Features |
|-----|------------|----------|
| **RUN B** (baseline) | On-chain only | 16 blockchain features |
| **RUN A** (enhanced) | All sources | 148 features across 8 categories |

Seven recurrent and attention-based architectures are evaluated, and model quality is assessed through both **statistical metrics** and a **simulated directional trading strategy**.

---

## Research Question

> Does adding macroeconomic, sentiment, and derivatives data to a Bitcoin forecasting model improve predictive accuracy beyond what on-chain blockchain data alone can provide?

The baseline (RUN B) is grounded in the **Efficient Market Hypothesis** — if markets are efficient, on-chain data should already reflect all relevant external information. The enhanced model (RUN A) challenges this by incorporating broader global risk factors.

---

## Target Variable

The model forecasts the **3-day-ahead smoothed log-return** of BTC/USD:

$$y_{t,\text{smooth}} = \frac{\ell r_t + \ell r_{t+1} + \ell r_{t+2}}{3}, \quad \ell r_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$

The smoothing reduces high-frequency noise, improving the signal-to-noise ratio for downstream trading.

---

## Dataset & Features

- **Source:** `ip26_btconchain.xlsx` — 3,000+ daily observations (2010–2026)
- **Working sample:** January 2017 – January 2026
- **Data leakage prevention:** all rolling windows, lags, and normalisation are strictly backward-looking

### Feature Categories (RUN A)

| Category | Features | Count |
|----------|----------|-------|
| On-Chain (Blockchain.com API) | Transactions, fees, hash rate, difficulty, addresses | 16 |
| Market (yfinance) | S&P 500, Gold, DXY, ETH, VIX + rolling stats | 19 |
| Macroeconomic (FRED API) | Fed Rate, CPI, M2, Yield Curve, Yield Spread | 10 |
| Sentiment | Fear & Greed Index, Google Trends | 8 |
| Derivatives | Open Interest, Funding Rate (Bybit/CoinGlass) | 3–6 |
| Technical | RSI, MACD, Bollinger Bands, ATR, Stochastic, moving averages | 24 |
| Regime | ATH distance, volatility regime, trend, MA200 signal | 5 |
| Seasonal & Halving | Month/week dummies, days since halving | ~65 |

---

## Architectures

Seven deep learning architectures are evaluated:

| Model | Description |
|-------|-------------|
| **LSTM** | Long Short-Term Memory — standard recurrent baseline |
| **GRU** | Gated Recurrent Unit — lightweight LSTM variant |
| **BI-LSTM** | Bidirectional LSTM — processes sequence in both directions |
| **LSTM-ATT** | LSTM + Bahdanau additive attention |
| **GRU-ATT** | GRU + Bahdanau additive attention |
| **BI-LSTM-ATT** | Bidirectional LSTM + attention |
| **SELF-ATT** | Transformer-style scaled dot-product self-attention |

---

## Methodology

### Data Split

| Partition | Period | Samples | Role |
|-----------|--------|---------|------|
| Training | 2017-01-01 → 2025-06-01 | ~1,700 | Weight optimisation |
| Validation | Last 15% of training | ~255 | Early stopping |
| Gap (excluded) | 2025-06-01 → 2025-09-01 | — | Anti-leakage buffer |
| **Test OOS** | **2025-09-01 → 2026-01-30** | **~109** | **Held-out evaluation** |

The 3-month gap prevents rolling-window features from bleeding information forward into the test set.

### Hyperparameter Tuning

- **Framework:** [Optuna](https://optuna.org/) — Bayesian hyperparameter optimisation
- **CV scheme:** 5-fold expanding walk-forward with 7-day anti-leakage gap per fold
- **Search space:** hidden units, dropout, learning rate, weight decay, lookback window (`n_lags`), directional loss weight

### Regularisation

- Dropout after recurrent/attention blocks
- AdamW weight decay (L2)
- Directional auxiliary loss (MSE + sign penalty)
- EMA-smoothed early stopping (patience = 20 epochs)
- Cosine Annealing + ReduceLROnPlateau scheduling

---

## Results Summary

### Statistical Performance (OOS Test)

> **Note:** The test window (Sep 2025 – Jan 2026) features a rare **double regime shift** — a ~20% drawdown in October followed by a ~40% rally through January. Negative R² and Theil's U > 1 are the expected outcome under such structural breaks and are not the primary evaluation criterion.

**Best performers by Directional Accuracy (DA):**

| Model | Run | DA (%) | Theil's U |
|-------|-----|--------|-----------|
| LSTM | B | 54.55% | 1.293 |
| SELF-ATT | A | 54.55% | 1.010 |
| GRU | A | 53.64% | 1.062 |
| BI-LSTM | B/A | 52.73% | ~1.08 |

### Trading Strategy Performance (OOS)

Two strategies simulated on 110-day OOS window:
- **Strategy A (Long/Flat):** Long on positive signal, cash on negative signal
- **Strategy B (Long/Short):** Long on positive signal, short on negative signal
- **Benchmark:** Buy & Hold → **−24.88%** (Sharpe −1.658)

**Top performers:**

| Model | Run | Strategy | Return | Sharpe | MaxDD | Trades |
|-------|-----|----------|--------|--------|-------|--------|
| **LSTM** | B | B | **+43.31%** | **2.515** | −13.54% | 21 |
| GRU | B | B | +26.12% | 1.674 | −12.64% | 25 |
| BI-LSTM | B | B | +16.47% | 1.156 | −17.17% | 7 |
| BI-LSTM | A | B | +7.50% | 0.643 | −20.48% | 16 |

---

## Key Findings

1. **Feature richness is not a free lunch.** Adding 148 variables to a 16-variable baseline improves in-sample fit for some architectures, but introduces fragility that manifests as directional bias under the OOS regime shift.

2. **RUN B outperforms RUN A in trading.** On-chain features alone support profitable strategies for 5 out of 7 architectures under Strategy B. In RUN A, only BI-LSTM generates a credibly active profitable result.

3. **Architecture mediates the effect of feature expansion.** GRU and SELF-ATT absorb the additional information constructively. LSTM-ATT and GRU-ATT degrade sharply — explicit cross-temporal attention over 164 heterogeneous features amplifies noise rather than signal under distribution shift.

4. **Statistical and economic performance diverge.** LSTM achieves negative R² but generates a +43.31% return with Sharpe 2.515 through 21 active trades. Pointwise accuracy metrics are unreliable proxies for trading utility.

5. **On-chain data carries a more stable signal.** Blockchain metrics, being endogenous to the Bitcoin network, appear more regime-robust than heterogeneous macroeconomic and sentiment variables subject to their own structural breaks.

---

## Hardware

All experiments were run on Google Colab:

- **GPU:** NVIDIA A100 Tensor Core (80 GB VRAM)
- **CPU:** Intel® Xeon® @ 2.20 GHz (12 vCPUs)
- **RAM:** 83 GB

---

## Project Structure

```
├── data/
│   └── ip26_btconchain.xlsx       # Primary on-chain dataset
├── features/
│   ├── onchain.py                 # On-chain feature engineering
│   ├── market.py                  # Market features (yfinance)
│   ├── macro.py                   # Macroeconomic features (FRED)
│   └── sentiment.py               # Sentiment & derivatives features
├── models/
│   ├── lstm.py
│   ├── gru.py
│   ├── bilstm.py
│   ├── attention.py               # Bahdanau attention mechanism
│   └── self_attention.py          # Transformer-style self-attention
├── training/
│   ├── preprocessing.py           # Leakage-free preprocessing pipeline
│   ├── walk_forward.py            # Walk-forward cross-validation
│   └── optuna_tuning.py           # Hyperparameter search
├── evaluation/
│   ├── metrics.py                 # RMSE, MAE, Theil's U, R², DA
│   └── trading_strategy.py        # Long/Flat and Long/Short backtests
└── README.md
```

---

## Citation

If you use this work, please cite:

```
Zeri, T. (2026). (Trying to) Forecast Bitcoin using Deep Neural Networks.
```

---

## License

This project is for academic and research purposes.
