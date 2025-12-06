"""
SARIMAX Forecasting Model with Axiom Ray as Leading Indicator.

Two models available:
1. Baseline: SARIMAX without exogenous variables (trend + seasonality only)
2. Enhanced: SARIMAX with Axiom Ray score as exogenous variable

Axiom Ray is a 2-week LEADING indicator, so:
- axiom_score[t] predicts volume[t+2]
- When forecasting, we use axiom_score from 2 weeks prior
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class CallVolumeForecaster:
    """
    SARIMAX-based forecaster with optional Axiom Ray exogenous variable.
    
    Supports forecasting with and without the leading indicator to
    demonstrate the value of the Axiom Ray AI system.
    """
    
    LEAD_WEEKS = 2  # Axiom Ray leads volume by 2 weeks
    
    def __init__(self, use_exogenous=True):
        """
        Initialize the forecaster.
        
        Args:
            use_exogenous: Whether to use Axiom Ray as exogenous variable
        """
        self.use_exogenous = use_exogenous
        self.model = None
        self.model_fitted = None
        self.training_data = None
        self.last_train_date = None
        self.freq = 'W-MON'
        
    def _prepare_exog(self, df, for_forecast=False):
        """
        Prepare exogenous variable with proper lag.
        
        Since Axiom Ray is a 2-week leading indicator:
        - axiom_score[t] predicts volume[t+2]
        - So for volume[t], we use axiom_score[t-2]
        """
        if not self.use_exogenous:
            return None
        
        # Check if df is a DataFrame with the required column
        if not isinstance(df, pd.DataFrame):
            return None
        if 'axiom_ray_score' not in df.columns:
            return None
        
        # Shift axiom score forward by LEAD_WEEKS to align with volume
        # axiom_score[t-2] → volume[t]
        exog = df['axiom_ray_score'].shift(self.LEAD_WEEKS).ffill().bfill()
        return exog.values.reshape(-1, 1)
    
    def fit(self, df, changepoint_prior_scale=0.05, seasonality_prior_scale=10, verbose=False):
        """
        Fit the SARIMAX model.
        
        Args:
            df: Training DataFrame with 'ds', 'y', and optionally 'axiom_ray_score'
        """
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        # Prepare endogenous variable
        endog = df['y'].values
        
        # Prepare exogenous variable (lagged Axiom Ray score)
        exog = self._prepare_exog(df)
        
        # SARIMAX order: (p,d,q) x (P,D,Q,s)
        # Using modest seasonal component for speed
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, 13)  # Quarterly seasonality for speed
        
        try:
            self.model = SARIMAX(
                endog=endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fitted = self.model.fit(disp=False, maxiter=100)
        except Exception as e:
            if verbose:
                print(f"SARIMAX with seasonality failed: {e}, trying simpler model")
            # Fallback to non-seasonal
            self.model = SARIMAX(
                endog=endog,
                exog=exog,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fitted = self.model.fit(disp=False, maxiter=50)
        
        return self
    
    def forecast_future(self, periods, last_date=None, future_exog=None):
        """
        Forecast future periods.
        
        Args:
            periods: Number of periods to forecast
            future_exog: DataFrame with future axiom_ray_score values (if using exogenous)
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if last_date is None:
            last_date = self.last_train_date
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=7),
            periods=periods,
            freq=self.freq
        )
        
        # Prepare exogenous for forecast
        exog_forecast = None
        has_training_axiom = isinstance(self.training_data, pd.DataFrame) and 'axiom_ray_score' in self.training_data.columns
        
        if self.use_exogenous and has_training_axiom:
            has_future_axiom = isinstance(future_exog, pd.DataFrame) and 'axiom_ray_score' in future_exog.columns
            
            if has_future_axiom:
                # Use provided future axiom scores (shifted by lead time)
                exog_forecast = future_exog['axiom_ray_score'].shift(self.LEAD_WEEKS).ffill().bfill().values[:periods]
            else:
                # Use last known axiom scores (they predict 2 weeks ahead!)
                # So the last LEAD_WEEKS axiom scores predict the first LEAD_WEEKS future volumes
                last_axiom_scores = self.training_data['axiom_ray_score'].tail(self.LEAD_WEEKS + periods).values
                exog_forecast = last_axiom_scores[:periods]
            
            # Ensure we have enough values
            if len(exog_forecast) < periods:
                # Pad with last value
                exog_forecast = np.pad(exog_forecast, (0, periods - len(exog_forecast)), mode='edge')
            
            exog_forecast = exog_forecast.reshape(-1, 1)
        
        # Get forecast
        forecast_result = self.model_fitted.get_forecast(steps=periods, exog=exog_forecast)
        
        predicted_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # Handle column names
        if 'lower y' in conf_int.columns:
            yhat_lower = conf_int['lower y'].values
            yhat_upper = conf_int['upper y'].values
        else:
            yhat_lower = conf_int.iloc[:, 0].values
            yhat_upper = conf_int.iloc[:, 1].values
        
        # Handle NaN/inf
        mean_val = self.training_data['y'].mean()
        predicted_mean = np.nan_to_num(predicted_mean, nan=mean_val)
        yhat_lower = np.nan_to_num(yhat_lower, nan=predicted_mean * 0.8)
        yhat_upper = np.nan_to_num(yhat_upper, nan=predicted_mean * 1.2)
        
        # Ensure reasonable bounds
        predicted_mean = np.maximum(predicted_mean, 100)
        yhat_lower = np.maximum(yhat_lower, 50)
        yhat_upper = np.maximum(yhat_upper, predicted_mean)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predicted_mean,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })
    
    def predict(self, df_future, include_history=True):
        """Make predictions for given dates."""
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before making predictions")
        
        df = df_future.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        fitted = self.model_fitted.fittedvalues
        resid_std = np.std(self.model_fitted.resid) if len(self.model_fitted.resid) > 0 else 50
        
        results = []
        for i, date in enumerate(df['ds']):
            if i < len(fitted):
                yhat = fitted[i]
            else:
                yhat = fitted[-1] if len(fitted) > 0 else self.training_data['y'].mean()
            
            results.append({
                'ds': date,
                'yhat': yhat,
                'yhat_lower': yhat - 1.96 * resid_std,
                'yhat_upper': yhat + 1.96 * resid_std
            })
        
        result_df = pd.DataFrame(results)
        result_df['yhat'] = np.maximum(result_df['yhat'], 100)
        result_df['yhat_lower'] = np.maximum(result_df['yhat_lower'], 50)
        result_df['yhat_upper'] = np.maximum(result_df['yhat_upper'], result_df['yhat'])
        
        return result_df
    
    def evaluate(self, df_test):
        """Evaluate model on test data."""
        df_test = df_test.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_test['ds']):
            df_test['ds'] = pd.to_datetime(df_test['ds'])
        df_test = df_test.sort_values('ds').reset_index(drop=True)
        
        n_test = len(df_test)
        
        # For exogenous forecasting, we need axiom scores
        future_exog = df_test if isinstance(df_test, pd.DataFrame) and 'axiom_ray_score' in df_test.columns else None
        forecast = self.forecast_future(periods=n_test, future_exog=future_exog)
        
        actual_values = df_test['y'].values
        predicted_values = forecast['yhat'].values
        
        min_len = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_len]
        predicted_values = predicted_values[:min_len]
        
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / np.maximum(actual_values, 1))) * 100
        
        yhat_lower = forecast['yhat_lower'].values[:min_len]
        yhat_upper = forecast['yhat_upper'].values[:min_len]
        within_ci = np.mean((actual_values >= yhat_lower) & (actual_values <= yhat_upper)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Within_CI_%': within_ci,
            'n_samples': min_len
        }
        
        eval_df = forecast.iloc[:min_len].copy()
        eval_df['y'] = actual_values
        
        return metrics, eval_df
    
    def get_components(self, forecast_df):
        """Extract forecast components (approximate)."""
        components = pd.DataFrame({
            'ds': forecast_df['ds'],
            'trend': forecast_df['yhat'],
            'yearly': np.sin(2 * np.pi * pd.to_datetime(forecast_df['ds']).dt.dayofyear / 365.25) * forecast_df['yhat'] * 0.1
        })
        return components


def compare_forecasts(df_train, df_test, forecast_periods):
    """
    Compare forecasts with and without Axiom Ray exogenous variable.
    
    Returns metrics and forecasts for both models.
    """
    results = {}
    
    # Model WITHOUT Axiom Ray (baseline)
    model_baseline = CallVolumeForecaster(use_exogenous=False)
    model_baseline.fit(df_train)
    metrics_baseline, eval_baseline = model_baseline.evaluate(df_test)
    forecast_baseline = model_baseline.forecast_future(periods=forecast_periods)
    
    results['baseline'] = {
        'model': model_baseline,
        'metrics': metrics_baseline,
        'eval_df': eval_baseline,
        'forecast': forecast_baseline,
        'name': 'Baseline (No Axiom Ray)'
    }
    
    # Model WITH Axiom Ray (enhanced)
    if isinstance(df_train, pd.DataFrame) and 'axiom_ray_score' in df_train.columns:
        model_enhanced = CallVolumeForecaster(use_exogenous=True)
        model_enhanced.fit(df_train)
        metrics_enhanced, eval_enhanced = model_enhanced.evaluate(df_test)
        forecast_enhanced = model_enhanced.forecast_future(periods=forecast_periods, future_exog=df_test)
        
        results['enhanced'] = {
            'model': model_enhanced,
            'metrics': metrics_enhanced,
            'eval_df': eval_enhanced,
            'forecast': forecast_enhanced,
            'name': 'Enhanced (With Axiom Ray)'
        }
        
        # Calculate improvement
        mape_improvement = metrics_baseline['MAPE'] - metrics_enhanced['MAPE']
        mae_improvement = metrics_baseline['MAE'] - metrics_enhanced['MAE']
        
        results['improvement'] = {
            'MAPE_reduction': mape_improvement,
            'MAE_reduction': mae_improvement,
            'MAPE_pct_improvement': (mape_improvement / metrics_baseline['MAPE']) * 100 if metrics_baseline['MAPE'] > 0 else 0,
            'MAE_pct_improvement': (mae_improvement / metrics_baseline['MAE']) * 100 if metrics_baseline['MAE'] > 0 else 0
        }
    
    return results


def train_test_split(df, test_size=0.2):
    """Split time series data into train and test sets."""
    df = df.sort_values('ds').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df
