"""
Forecasting model for call center volume using Holt-Winters Triple Exponential Smoothing.

This method captures:
- Level (base)
- Trend (growth/decline)
- Seasonality (repeating patterns)

Fast and robust for time series with clear seasonal patterns.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class CallVolumeForecaster:
    """
    Forecast call center volume using Holt-Winters Triple Exponential Smoothing.
    """
    
    def __init__(self, weekly_seasonality=True, yearly_seasonality=True):
        """
        Initialize the forecaster.
        """
        self.model = None
        self.model_fitted = None
        self.training_data = None
        self.last_train_date = None
        self.freq = 'W-MON'
        self.seasonal_period = 52  # Weekly data, yearly seasonality
        
    def fit(self, df, changepoint_prior_scale=0.05, seasonality_prior_scale=10, verbose=False):
        """
        Fit the Holt-Winters model.
        
        Args:
            df: Training DataFrame with 'ds' and 'y' columns
        """
        df = df.copy()
        
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort and store training data
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        # Extract the time series
        y = df['y'].values
        
        # Determine if we have enough data for yearly seasonality
        # Need at least 2 full seasonal periods
        if len(y) >= self.seasonal_period * 2:
            seasonal_periods = self.seasonal_period
            seasonal = 'add'  # Additive seasonality
        elif len(y) >= 13 * 2:
            # Try quarterly seasonality (13 weeks)
            seasonal_periods = 13
            seasonal = 'add'
        else:
            # Not enough data for seasonality
            seasonal_periods = None
            seasonal = None
        
        try:
            # Holt-Winters Triple Exponential Smoothing
            if seasonal_periods:
                self.model = ExponentialSmoothing(
                    y,
                    trend='add',           # Additive trend
                    seasonal=seasonal,      # Additive seasonality
                    seasonal_periods=seasonal_periods,
                    damped_trend=True,      # Dampen trend to prevent explosion
                    initialization_method='estimated'
                )
            else:
                # Double Exponential Smoothing (no seasonality)
                self.model = ExponentialSmoothing(
                    y,
                    trend='add',
                    seasonal=None,
                    damped_trend=True,
                    initialization_method='estimated'
                )
            
            self.model_fitted = self.model.fit(optimized=True)
            
        except Exception as e:
            # Fallback to simple exponential smoothing
            print(f"Holt-Winters failed: {e}, falling back to simple model")
            self.model = ExponentialSmoothing(
                y,
                trend='add',
                seasonal=None,
                damped_trend=True
            )
            self.model_fitted = self.model.fit(optimized=True)
        
        return self
    
    def forecast_future(self, periods, last_date=None, future_exog=None):
        """
        Forecast future periods.
        
        Args:
            periods: Number of periods to forecast
        
        Returns:
            DataFrame with forecast
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
        
        # Get forecast
        forecast = self.model_fitted.forecast(periods)
        
        # Get prediction intervals using simulation or residual-based approach
        # Holt-Winters doesn't provide native confidence intervals easily,
        # so we'll estimate based on training residuals
        residuals = self.model_fitted.resid
        residual_std = np.std(residuals)
        
        # Wider intervals for further forecasts
        horizon_factor = np.sqrt(np.arange(1, periods + 1))
        interval_width = 1.96 * residual_std * horizon_factor
        
        yhat = np.array(forecast)
        yhat_lower = yhat - interval_width
        yhat_upper = yhat + interval_width
        
        # Ensure reasonable bounds
        yhat = np.maximum(yhat, 100)
        yhat_lower = np.maximum(yhat_lower, 50)
        yhat_upper = np.maximum(yhat_upper, yhat)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': yhat,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })
    
    def predict(self, df_future, include_history=True):
        """
        Make predictions for given dates (historical fitted values).
        """
        if self.model_fitted is None:
            raise ValueError("Model must be fitted before making predictions")
        
        df = df_future.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        # Get fitted values for historical data
        fitted = self.model_fitted.fittedvalues
        
        # Residual std for confidence intervals
        residual_std = np.std(self.model_fitted.resid)
        
        results = []
        for i, date in enumerate(df['ds']):
            if i < len(fitted):
                yhat = fitted[i]
            else:
                # Out of sample - use last fitted value as approximation
                yhat = fitted[-1] if len(fitted) > 0 else self.training_data['y'].mean()
            
            results.append({
                'ds': date,
                'yhat': yhat,
                'yhat_lower': yhat - 1.96 * residual_std,
                'yhat_upper': yhat + 1.96 * residual_std
            })
        
        result_df = pd.DataFrame(results)
        
        # Clean up
        result_df['yhat'] = np.maximum(result_df['yhat'], 100)
        result_df['yhat_lower'] = np.maximum(result_df['yhat_lower'], 50)
        result_df['yhat_upper'] = np.maximum(result_df['yhat_upper'], result_df['yhat'])
        
        return result_df
    
    def evaluate(self, df_test):
        """
        Evaluate model on test data using TRUE out-of-sample forecasting.
        """
        df_test = df_test.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_test['ds']):
            df_test['ds'] = pd.to_datetime(df_test['ds'])
        df_test = df_test.sort_values('ds').reset_index(drop=True)
        
        # Number of test periods
        n_test = len(df_test)
        
        # Get TRUE out-of-sample forecast
        forecast = self.forecast_future(periods=n_test)
        
        # Get actual values
        actual_values = df_test['y'].values
        predicted_values = forecast['yhat'].values
        
        # Ensure same length
        min_len = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_len]
        predicted_values = predicted_values[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        
        # MAPE
        mape = np.mean(np.abs((actual_values - predicted_values) / np.maximum(actual_values, 1))) * 100
        
        # Within confidence interval
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
        
        # Evaluation dataframe
        eval_df = forecast.iloc[:min_len].copy()
        eval_df['y'] = actual_values
        
        return metrics, eval_df
    
    def get_components(self, forecast_df):
        """
        Extract approximate forecast components.
        """
        if self.model_fitted is None:
            return pd.DataFrame({'ds': forecast_df['ds'], 'trend': forecast_df['yhat']})
        
        # Extract level, trend, season from fitted model if available
        try:
            level = self.model_fitted.level
            trend = self.model_fitted.trend if hasattr(self.model_fitted, 'trend') else np.zeros(len(level))
            season = self.model_fitted.season if hasattr(self.model_fitted, 'season') else np.zeros(len(level))
            
            # Extend to forecast length if needed
            n_forecast = len(forecast_df)
            if len(level) < n_forecast:
                # Pad with last values
                level = np.concatenate([level, np.full(n_forecast - len(level), level[-1])])
                trend = np.concatenate([trend, np.full(n_forecast - len(trend), trend[-1] if len(trend) > 0 else 0)])
                if len(season) > 0:
                    # Repeat seasonal pattern
                    season_period = self.seasonal_period if self.seasonal_period else 13
                    reps = (n_forecast - len(season)) // season_period + 1
                    season_extended = np.tile(season[-season_period:], reps)
                    season = np.concatenate([season, season_extended])[:n_forecast]
                else:
                    season = np.zeros(n_forecast)
            
            components = pd.DataFrame({
                'ds': forecast_df['ds'],
                'trend': (level + trend)[:n_forecast],
                'yearly': season[:n_forecast] if len(season) >= n_forecast else np.zeros(n_forecast)
            })
        except Exception as e:
            # Fallback
            components = pd.DataFrame({
                'ds': forecast_df['ds'],
                'trend': forecast_df['yhat'],
                'yearly': np.sin(2 * np.pi * pd.to_datetime(forecast_df['ds']).dt.dayofyear / 365.25) * forecast_df['yhat'] * 0.1
            })
        
        return components


def train_test_split(df, test_size=0.2):
    """
    Split time series data into train and test sets.
    """
    df = df.sort_values('ds').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df
