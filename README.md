# Call Center Volume Forecasting Demo

A demonstration forecasting dashboard for call center volume prediction using Facebook's Prophet with exogenous variables (Axiom Ray AI predictions).

## Features

- **Weekly Call Volume Forecasting**: Predicts future call volumes for a major product line
- **Exogenous Variables**: Incorporates Axiom Ray AI tool predictions as external regressors
- **Interactive Dashboard**: Streamlit-based dashboard for visualization and analysis
- **Synthetic Data**: Realistic synthetic data generation for demonstration purposes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Generate synthetic data and run the dashboard:

```bash
python generate_data.py
streamlit run dashboard.py
```

## Model Details

- **Base Model**: Facebook Prophet with weekly seasonality
- **External Regressors**: Axiom Ray AI early issue predictions
- **Forecast Horizon**: Configurable (default: 12 weeks)
- **Evaluation Metrics**: MAE, RMSE, MAPE

## Files

- `generate_data.py`: Synthetic data generation for historical volumes and Axiom Ray predictions
- `forecast_model.py`: Prophet forecasting model implementation
- `dashboard.py`: Streamlit dashboard application
- `data/`: Directory containing generated datasets

## Deployment

The dashboard can be deployed to various platforms. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Quick Deploy to Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file: `dashboard.py`
5. Click "Deploy"

The dashboard will auto-generate data on first run.

### Other Deployment Options

- **Docker**: Use the included `Dockerfile` for containerized deployment
- **Heroku**: Use the included `Procfile`
- **Cloud Platforms**: Deploy Docker containers to AWS, GCP, or Azure

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

