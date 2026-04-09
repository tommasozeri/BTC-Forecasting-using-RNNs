(Sources and basic code was taken from the UNIBO Intensive Program in Artifical Intelligence by professors Maurizio Morini, Umberto Cherubini and Giovanni Dalla Lunga)

Overview
This project investigates whether incorporating off-chain data (macroeconomic indicators, sentiment, derivatives) meaningfully improves the predictive accuracy of Bitcoin forecasting models built on deep neural networks.
The methodology is structured as a rigorous ablation study with two experimental configurations:
RunFeature SetFeaturesRUN B (baseline)On-chain only16 blockchain featuresRUN A (enhanced)All sources148 features across 8 categories
Seven recurrent and attention-based architectures are evaluated, and model quality is assessed through both statistical metrics and a simulated directional trading strategy.

Research Question

Does adding macroeconomic, sentiment, and derivatives data to a Bitcoin forecasting model improve predictive accuracy beyond what on-chain blockchain data alone can provide?


Target Variable
The model forecasts the 3-day-ahead smoothed log-return of BTC/USD:
yt,smooth=ℓrt+ℓrt+1+ℓrt+23,ℓrt=log⁡(PtPt−1)y_{t,\text{smooth}} = \frac{\ell r_t + \ell r_{t+1} + \ell r_{t+2}}{3}, \quad \ell r_t = \log\left(\frac{P_t}{P_{t-1}}\right)yt,smooth​=3ℓrt​+ℓrt+1​+ℓrt+2​​,ℓrt​=log(Pt−1​Pt​​)

Dataset & Features

Source: ip26_btconchain.xlsx — 3,000+ daily observations (2010–2026)
Working sample: January 2017 – January 2026

CategoryCountOn-Chain (Blockchain.com API)16Market (yfinance)19Macroeconomic (FRED API)10Sentiment8Derivatives3–6Technical24Regime5Seasonal & Halving~65

Architectures
ModelDescriptionLSTMLong Short-Term MemoryGRUGated Recurrent UnitBI-LSTMBidirectional LSTMLSTM-ATTLSTM + Bahdanau attentionGRU-ATTGRU + Bahdanau attentionBI-LSTM-ATTBidirectional LSTM + attentionSELF-ATTTransformer-style self-attention

Methodology
PartitionPeriodRoleTraining2017-01-01 → 2025-06-01Weight optimisationValidationLast 15% of trainingEarly stoppingGap2025-06-01 → 2025-09-01Anti-leakage bufferTest OOS2025-09-01 → 2026-01-30Held-out evaluation
Hyperparameter tuning via Optuna with 5-fold expanding walk-forward CV.

Key Results
Benchmark (Buy & Hold): −24.88% | Sharpe −1.658
ModelRunStrategyReturnSharpeTradesLSTMBLong/Short+43.31%2.51521GRUBLong/Short+26.12%1.67425BI-LSTMBLong/Short+16.47%1.1567BI-LSTMALong/Short+7.50%0.64316

Key Findings

Feature richness is not a free lunch — adding 148 variables to a 16-variable baseline introduces fragility under regime shifts.
RUN B (on-chain only) outperforms RUN A in trading — on-chain features alone support profitable strategies for 5/7 architectures.
Architecture mediates feature expansion — GRU and SELF-ATT benefit; LSTM-ATT and GRU-ATT degrade sharply.
Statistical and economic performance diverge — LSTM achieves negative R² but generates +43.31% return with Sharpe 2.515.
On-chain data is more regime-robust than heterogeneous macroeconomic and sentiment variables.


Hardware
Google Colab — NVIDIA A100 (80 GB VRAM), Intel Xeon @ 2.20 GHz, 83 GB RAM

Citation
Zeri, T. (2026). (Trying to) Forecast Bitcoin using Deep Neural Networks.
