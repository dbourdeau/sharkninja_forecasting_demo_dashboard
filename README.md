# SharkNinja Customer Support Forecasting Dashboard

**Author:** Daniel Bourdeau  
**Purpose:** SharkNinja CS Forecasting Analyst Interview  
**Note:** For demonstration purposes only - uses synthetic data

## Overview

Interactive forecasting dashboard for call center volume prediction using multiple time series models:

- **SARIMAX** - Seasonal ARIMA with trend
- **SARIMAX + Axiom Ray** - With AI early-warning signals as exogenous variable
- **Holt-Winters** - Triple Exponential Smoothing
- **Ensemble** - Weighted combination of models

## Features

- Multi-model forecast comparison
- Business impact analysis (staffing, costs, ROI)
- Scenario planning (what-if analysis)
- Product category breakdown (Shark vs Ninja)
- Axiom Ray AI leading indicator analysis
- Executive summary dashboard

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Launch dashboard
python -m streamlit run dashboard.py
```

Or simply double-click `run_demo.bat` on Windows.

### Deploy to Streamlit Cloud

1. Push to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Deploy from `main` branch with `dashboard.py` as entry point

## Project Structure

```
├── dashboard.py          # Main Streamlit application
├── forecast_model.py     # SARIMAX, Holt-Winters, Ensemble models
├── generate_data.py      # Synthetic data generation
├── business_metrics.py   # Staffing, costs, ROI calculations
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version for deployment
├── data/
│   └── combined_data.csv # Generated synthetic data
└── .streamlit/
    └── config.toml      # Streamlit configuration
```

## Models

| Model | Description | Use Case |
|-------|-------------|----------|
| SARIMAX Baseline | Trend + seasonality | Standard forecasting |
| SARIMAX + Axiom Ray | With 2-week leading indicator | Early warning detection |
| Holt-Winters | Exponential smoothing | Adaptive forecasting |
| Ensemble | Weighted average | Best overall accuracy |

## Axiom Ray Leading Indicator

Simulates an AI system that detects support volume changes 2 weeks ahead by monitoring:
- Social media sentiment
- Product review trends
- Warranty claim patterns
- Search trend anomalies

The leading indicator has ~0.7 correlation with future volume (realistic, not perfect).

## Technologies

- **Streamlit** - Interactive dashboard framework
- **Statsmodels** - SARIMAX and Holt-Winters models
- **Plotly** - Interactive visualizations
- **Pandas/NumPy** - Data processing

---

*Built for SharkNinja CS Forecasting Analyst Interview - Demonstration purposes only*
