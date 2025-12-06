# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: Prophet may require additional system dependencies. If you encounter errors:
   - **Windows**: Usually works out of the box with pip
   - **Linux/Mac**: May need to install `pystan` dependencies first

2. **Generate synthetic data:**
   ```bash
   python generate_data.py
   ```

   This creates:
   - `data/historical_volume.csv` - Historical call volume
   - `data/axiom_ray_predictions.csv` - Axiom Ray AI predictions
   - `data/combined_data.csv` - Combined dataset

## Running the Dashboard

**Option 1: Windows (Double-click)**
- Double-click `run_demo.bat`

**Option 2: Command Line**
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## Dashboard Features

### 📊 Forecast Tab
- Interactive forecast visualization
- Forecast summary table
- Forecast statistics

### 📈 Components Tab
- Breakdown of forecast components:
  - Trend
  - Weekly seasonality
  - Yearly seasonality
  - Axiom Ray AI contribution

### 🔗 Axiom Ray Analysis Tab
- Correlation analysis between Axiom Ray predictions and call volume
- Time series comparison
- Scatter plot analysis

### 📉 Model Performance Tab
- Evaluation metrics (MAE, RMSE, MAPE)
- Predictions vs Actuals plot
- Residuals analysis

## Customization

### Adjust Model Parameters (Sidebar)
- **Test Set Size**: Percentage of data to use for testing (10-40%)
- **Forecast Weeks**: Number of weeks to forecast ahead (4-24)
- **Changepoint Prior Scale**: Controls trend flexibility (0.01-0.5)
- **Seasonality Prior Scale**: Controls seasonality strength (1.0-20.0)

### Modify Data Generation

Edit `generate_data.py` to customize:
- Number of weeks of historical data
- Seasonality patterns
- Correlation strength between Axiom Ray and volume
- Start date

## Troubleshooting

### Import Errors
If you get import errors, ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

### Prophet Installation Issues
Prophet can be tricky to install. Try:
```bash
pip install prophet --upgrade
```

If that fails, try:
```bash
conda install -c conda-forge prophet
```

### Data Not Found
Make sure to run `python generate_data.py` before starting the dashboard.

### Port Already in Use
If port 8501 is already in use:
```bash
streamlit run dashboard.py --server.port 8502
```

## Next Steps

1. Replace synthetic data with real call center data
2. Integrate actual Axiom Ray AI predictions API
3. Add more external regressors (marketing campaigns, product launches, etc.)
4. Implement model retraining pipeline
5. Add alerting for forecasted high-volume periods

## For Your Interview

Key points to highlight:

1. **External Regressors**: Demonstrated use of Axiom Ray AI as predictive input
2. **Seasonality Handling**: Automatic detection of weekly and yearly patterns
3. **Uncertainty Quantification**: Confidence intervals for decision-making
4. **Model Evaluation**: Comprehensive metrics and validation approach
5. **Interactive Dashboard**: Easy-to-use interface for stakeholders

## Questions to Prepare For

- How would you validate the model on real data?
- What additional features would improve forecast accuracy?
- How would you handle sudden spikes (product recalls, viral issues)?
- What's the business impact of forecast errors?
- How would you scale this to multiple products?

