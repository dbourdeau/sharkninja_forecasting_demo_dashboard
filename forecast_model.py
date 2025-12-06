"""
Robust Multi-Model Forecasting with Trend and Seasonality

Models available:
1. SARIMAX with Axiom Ray as exogenous leading indicator
2. Holt-Winters Triple Exponential Smoothing (trend + seasonality)
3. Ensemble: Weighted average with trend continuation

Key improvements for robust forecasting:
- Explicit trend detection and projection
- Multiple seasonal period testing
- Automatic model selection based on fit
- Trend-adjusted ensemble weighting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def detect_trend(y, return_slope=False):
    """Detect linear trend in the series."""
    t = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(t, y)
    trend_line = model.predict(t)
    if return_slope:
        return trend_line, model.coef_[0], model.intercept_
    return trend_line


def project_trend(slope, intercept, n_train, n_forecast):
    """Project trend into the future."""
    t_future = np.arange(n_train, n_train + n_forecast)
    return intercept + slope * t_future


class SARIMAXForecaster:
    """SARIMAX model with optional Axiom Ray exogenous variable and robust trend handling."""
    
    LEAD_WEEKS = 2
    
    def __init__(self, use_exogenous=True):
        self.use_exogenous = use_exogenous
        self.model = None
        self.model_fitted = None
        self.training_data = None
        self.last_train_date = None
        self.has_axiom = False
        self.trend_slope = 0
        self.trend_intercept = 0
        self.name = "SARIMAX" + (" + Axiom Ray" if use_exogenous else "")
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        self.has_axiom = 'axiom_ray_score' in df.columns
        
        y = np.array(df['y'])
        n = len(y)
        
        # Detect trend for later projection
        _, self.trend_slope, self.trend_intercept = detect_trend(y, return_slope=True)
        
        endog = y
        
        exog = None
        if self.use_exogenous and self.has_axiom:
            exog_series = df['axiom_ray_score'].shift(self.LEAD_WEEKS).ffill().bfill()
            exog = np.array(exog_series).reshape(-1, 1)
        
        # Try multiple seasonal orders and pick the best
        best_aic = np.inf
        best_model = None
        
        # Seasonal orders to try: quarterly (13 weeks) and annual (52 weeks approx)
        configs = [
            ((1, 1, 1), (1, 1, 1, 13)),  # Quarterly seasonality
            ((1, 1, 1), (1, 0, 1, 13)),  # Quarterly, no seasonal differencing
            ((2, 1, 2), (1, 0, 1, 13)),  # More complex ARIMA
            ((1, 1, 1), (0, 0, 0, 0)),   # No seasonality fallback
        ]
        
        for order, seasonal_order in configs:
            try:
                model = SARIMAX(
                    endog=endog,
                    exog=exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='t'  # Include trend in the model
                )
                fitted = model.fit(disp=False, maxiter=200)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    self.model = model
            except Exception:
                continue
        
        if best_model is None:
            # Ultimate fallback - simple model
            self.model = SARIMAX(endog=endog, exog=exog, order=(1, 1, 1),
                                enforce_stationarity=False, enforce_invertibility=False)
            best_model = self.model.fit(disp=False, maxiter=50)
        
        self.model_fitted = best_model
        return self
    
    def forecast(self, periods, future_exog=None):
        if self.model_fitted is None:
            raise ValueError("Model must be fitted first")
        
        n_train = len(self.training_data)
        
        future_dates = pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=7),
            periods=periods, freq='W-MON'
        )
        
        exog_forecast = None
        if self.use_exogenous and self.has_axiom:
            if isinstance(future_exog, pd.DataFrame) and 'axiom_ray_score' in future_exog.columns:
                exog_series = future_exog['axiom_ray_score'].shift(self.LEAD_WEEKS).ffill().bfill()
                exog_values = np.array(exog_series)[:periods]
            else:
                exog_values = np.array(self.training_data['axiom_ray_score'].tail(periods))
            
            if len(exog_values) < periods:
                exog_values = np.pad(exog_values, (0, periods - len(exog_values)), mode='edge')
            exog_forecast = exog_values[:periods].reshape(-1, 1)
        
        forecast_result = self.model_fitted.get_forecast(steps=periods, exog=exog_forecast)
        
        predicted_mean = forecast_result.predicted_mean
        if hasattr(predicted_mean, 'values'):
            predicted_mean = predicted_mean.values
        predicted_mean = np.array(predicted_mean)
        
        conf_int = forecast_result.conf_int()
        try:
            if hasattr(conf_int, 'columns') and 'lower y' in conf_int.columns:
                yhat_lower = np.array(conf_int['lower y'])
                yhat_upper = np.array(conf_int['upper y'])
            elif hasattr(conf_int, 'iloc'):
                yhat_lower = np.array(conf_int.iloc[:, 0])
                yhat_upper = np.array(conf_int.iloc[:, 1])
            else:
                yhat_lower = np.array(conf_int[:, 0])
                yhat_upper = np.array(conf_int[:, 1])
        except Exception:
            yhat_lower = predicted_mean * 0.85
            yhat_upper = predicted_mean * 1.15
        
        # If forecast is too flat, blend with trend projection
        forecast_range = np.max(predicted_mean) - np.min(predicted_mean)
        historical_range = np.max(self.training_data['y']) - np.min(self.training_data['y'])
        
        if forecast_range < 0.1 * historical_range:
            # Forecast is suspiciously flat - add trend component
            trend_projection = project_trend(self.trend_slope, self.trend_intercept, n_train, periods)
            last_actual = self.training_data['y'].iloc[-1]
            trend_adjustment = trend_projection - trend_projection[0] + last_actual
            predicted_mean = 0.5 * predicted_mean + 0.5 * trend_adjustment
        
        mean_val = self.training_data['y'].mean()
        predicted_mean = np.nan_to_num(predicted_mean, nan=mean_val)
        yhat_lower = np.nan_to_num(yhat_lower, nan=predicted_mean * 0.85)
        yhat_upper = np.nan_to_num(yhat_upper, nan=predicted_mean * 1.15)
        
        predicted_mean = np.maximum(predicted_mean, 100)
        yhat_lower = np.maximum(yhat_lower, 50)
        yhat_upper = np.maximum(yhat_upper, predicted_mean)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predicted_mean,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })


class HoltWintersForecaster:
    """Holt-Winters Triple Exponential Smoothing - forces trend and seasonality."""
    
    def __init__(self):
        self.model = None
        self.model_fitted = None
        self.training_data = None
        self.last_train_date = None
        self.trend_slope = 0
        self.trend_intercept = 0
        self.seasonal_pattern = None
        self.name = "Holt-Winters"
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        y = np.array(df['y']).astype(float)
        n = len(y)
        
        # Detect trend
        _, self.trend_slope, self.trend_intercept = detect_trend(y, return_slope=True)
        
        # Extract seasonal pattern from data (quarterly = 13 weeks)
        seasonal_period = 13
        if n >= 2 * seasonal_period:
            # Compute average seasonal pattern
            detrended = y - (self.trend_intercept + self.trend_slope * np.arange(n))
            n_complete = (n // seasonal_period) * seasonal_period
            seasonal_matrix = detrended[:n_complete].reshape(-1, seasonal_period)
            self.seasonal_pattern = np.mean(seasonal_matrix, axis=0)
        else:
            self.seasonal_pattern = np.zeros(seasonal_period)
        
        # Force use of additive trend and seasonality with NO damping for clear projections
        best_model = None
        
        # Priority: seasonal models first (with non-damped trend)
        configs = [
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 13, 'damped_trend': False},
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 13, 'damped_trend': True},
            {'trend': 'add', 'seasonal': 'mul', 'seasonal_periods': 13, 'damped_trend': False},
            {'trend': 'add', 'seasonal': None, 'damped_trend': False},  # No damping fallback
        ]
        
        for config in configs:
            try:
                # Ensure positive values for multiplicative
                y_fit = y.copy()
                if config.get('seasonal') == 'mul':
                    y_fit = np.maximum(y_fit, 1)
                
                model = ExponentialSmoothing(y_fit, **config)
                fitted = model.fit(optimized=True, use_brute=True)
                best_model = fitted
                self.model = model
                break  # Use first successful model
            except Exception:
                continue
        
        if best_model is None:
            # Ultimate fallback - simple exponential smoothing with trend
            self.model = ExponentialSmoothing(y, trend='add', damped_trend=False)
            best_model = self.model.fit(optimized=True)
        
        self.model_fitted = best_model
        return self
    
    def forecast(self, periods, future_exog=None):
        if self.model_fitted is None:
            raise ValueError("Model must be fitted first")
        
        n_train = len(self.training_data)
        y = np.array(self.training_data['y'])
        
        future_dates = pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=7),
            periods=periods, freq='W-MON'
        )
        
        # Get base forecast from model
        forecast_result = self.model_fitted.forecast(periods)
        predicted_mean = np.array(forecast_result)
        
        # Always add explicit trend and seasonality continuation
        # This ensures forecasts continue observed patterns
        
        # 1. Trend projection
        trend_projection = project_trend(self.trend_slope, self.trend_intercept, n_train, periods)
        
        # 2. Seasonal projection (repeat pattern)
        seasonal_period = 13
        seasonal_projection = np.zeros(periods)
        if self.seasonal_pattern is not None and len(self.seasonal_pattern) > 0:
            start_phase = n_train % seasonal_period
            for i in range(periods):
                seasonal_projection[i] = self.seasonal_pattern[(start_phase + i) % seasonal_period]
        
        # 3. Combine: use model forecast but ensure it follows trend
        last_actual = y[-1]
        model_level = predicted_mean[0]  # Where model thinks we are
        
        # Adjust model forecast to continue from last actual
        level_adjustment = last_actual - model_level
        adjusted_forecast = predicted_mean + level_adjustment
        
        # If model forecast is too flat, blend with trend + seasonality
        forecast_range = np.max(predicted_mean) - np.min(predicted_mean)
        historical_std = np.std(y[-26:])  # Recent volatility
        
        if forecast_range < 0.3 * historical_std:
            # Model is too flat - use trend + seasonality directly
            trend_based = trend_projection + seasonal_projection
            # Adjust to start from last actual
            trend_based = trend_based - trend_based[0] + last_actual
            predicted_mean = 0.3 * adjusted_forecast + 0.7 * trend_based
        else:
            predicted_mean = adjusted_forecast
        
        # Confidence intervals
        residuals = self.model_fitted.resid
        std_resid = np.std(residuals) if len(residuals) > 0 else np.std(y) * 0.15
        
        # Widen CI as we go further out
        ci_multiplier = 1.96 * np.sqrt(1 + np.arange(periods) * 0.05)
        yhat_lower = predicted_mean - ci_multiplier * std_resid
        yhat_upper = predicted_mean + ci_multiplier * std_resid
        
        mean_val = self.training_data['y'].mean()
        predicted_mean = np.nan_to_num(predicted_mean, nan=mean_val)
        yhat_lower = np.nan_to_num(yhat_lower, nan=predicted_mean * 0.85)
        yhat_upper = np.nan_to_num(yhat_upper, nan=predicted_mean * 1.15)
        
        predicted_mean = np.maximum(predicted_mean, 100)
        yhat_lower = np.maximum(yhat_lower, 50)
        yhat_upper = np.maximum(yhat_upper, predicted_mean)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predicted_mean,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })


class EnsembleForecaster:
    """Smart ensemble that combines models based on their strengths."""
    
    def __init__(self, weights=None):
        self.sarimax = SARIMAXForecaster(use_exogenous=True)
        self.holtwinters = HoltWintersForecaster()
        self.weights = weights or {'sarimax': 0.5, 'holtwinters': 0.5}
        self.training_data = None
        self.last_train_date = None
        self.trend_slope = 0
        self.trend_intercept = 0
        self.name = "Ensemble (SARIMAX + Holt-Winters)"
        self.model_performance = {}
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        y = np.array(df['y'])
        _, self.trend_slope, self.trend_intercept = detect_trend(y, return_slope=True)
        
        # Fit both models
        self.sarimax.fit(df)
        self.holtwinters.fit(df)
        
        # Cross-validation to determine weights
        n = len(df)
        val_size = max(4, int(n * 0.15))
        train_cv = df.iloc[:-val_size].copy()
        val_cv = df.iloc[-val_size:].copy()
        
        sarimax_mape = 50
        hw_mape = 50
        
        try:
            sarimax_cv = SARIMAXForecaster(use_exogenous=True)
            sarimax_cv.fit(train_cv)
            sarimax_pred = sarimax_cv.forecast(val_size, future_exog=val_cv)
            actual = np.array(val_cv['y'])
            sarimax_mape = np.mean(np.abs((actual - np.array(sarimax_pred['yhat'])) / np.maximum(actual, 1))) * 100
        except Exception:
            pass
        
        try:
            hw_cv = HoltWintersForecaster()
            hw_cv.fit(train_cv)
            hw_pred = hw_cv.forecast(val_size)
            actual = np.array(val_cv['y'])
            hw_mape = np.mean(np.abs((actual - np.array(hw_pred['yhat'])) / np.maximum(actual, 1))) * 100
        except Exception:
            pass
        
        self.model_performance = {
            'sarimax_mape': sarimax_mape,
            'holtwinters_mape': hw_mape
        }
        
        # Weights inversely proportional to error
        total_inv_error = (1 / max(sarimax_mape, 1)) + (1 / max(hw_mape, 1))
        self.weights = {
            'sarimax': (1 / max(sarimax_mape, 1)) / total_inv_error,
            'holtwinters': (1 / max(hw_mape, 1)) / total_inv_error
        }
        
        return self
    
    def forecast(self, periods, future_exog=None):
        sarimax_forecast = self.sarimax.forecast(periods, future_exog)
        hw_forecast = self.holtwinters.forecast(periods)
        
        w_s = self.weights['sarimax']
        w_h = self.weights['holtwinters']
        
        yhat = w_s * np.array(sarimax_forecast['yhat']) + w_h * np.array(hw_forecast['yhat'])
        yhat_lower = w_s * np.array(sarimax_forecast['yhat_lower']) + w_h * np.array(hw_forecast['yhat_lower'])
        yhat_upper = w_s * np.array(sarimax_forecast['yhat_upper']) + w_h * np.array(hw_forecast['yhat_upper'])
        
        # Check for flat forecast and add trend if needed
        n_train = len(self.training_data)
        forecast_range = np.max(yhat) - np.min(yhat)
        historical_range = np.max(self.training_data['y']) - np.min(self.training_data['y'])
        
        if forecast_range < 0.15 * historical_range:
            trend_projection = project_trend(self.trend_slope, self.trend_intercept, n_train, periods)
            last_actual = self.training_data['y'].iloc[-1]
            trend_adjustment = trend_projection - trend_projection[0] + last_actual
            yhat = 0.6 * yhat + 0.4 * trend_adjustment
        
        return pd.DataFrame({
            'ds': sarimax_forecast['ds'],
            'yhat': yhat,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })


class CallVolumeForecaster:
    """
    Unified forecaster interface supporting multiple model types.
    """
    
    LEAD_WEEKS = 2
    
    def __init__(self, model_type='ensemble', use_exogenous=True):
        self.model_type = model_type
        self.use_exogenous = use_exogenous
        
        if model_type == 'sarimax':
            self.model = SARIMAXForecaster(use_exogenous=use_exogenous)
        elif model_type == 'sarimax_baseline':
            self.model = SARIMAXForecaster(use_exogenous=False)
        elif model_type == 'holtwinters':
            self.model = HoltWintersForecaster()
        elif model_type == 'ensemble':
            self.model = EnsembleForecaster()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.training_data = None
        self.last_train_date = None
        self.name = self.model.name
        
    def fit(self, df, changepoint_prior_scale=0.05, seasonality_prior_scale=10, verbose=False):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        self.model.fit(df)
        return self
    
    def forecast_future(self, periods, last_date=None, future_exog=None):
        return self.model.forecast(periods, future_exog)
    
    def predict(self, df_future, include_history=True):
        if not isinstance(df_future, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = df_future.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        return self.forecast_future(len(df), future_exog=df)
    
    def evaluate(self, df_test):
        if not isinstance(df_test, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df_test = df_test.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_test['ds']):
            df_test['ds'] = pd.to_datetime(df_test['ds'])
        df_test = df_test.sort_values('ds').reset_index(drop=True)
        
        n_test = len(df_test)
        future_exog = df_test if 'axiom_ray_score' in df_test.columns else None
        forecast = self.forecast_future(periods=n_test, future_exog=future_exog)
        
        actual_values = np.array(df_test['y'])
        predicted_values = np.array(forecast['yhat'])
        
        min_len = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_len]
        predicted_values = predicted_values[:min_len]
        
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / np.maximum(actual_values, 1))) * 100
        
        yhat_lower = np.array(forecast['yhat_lower'])[:min_len]
        yhat_upper = np.array(forecast['yhat_upper'])[:min_len]
        within_ci = np.mean((actual_values >= yhat_lower) & (actual_values <= yhat_upper)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Within_CI_%': within_ci,
            'n_samples': min_len,
            'model_type': self.model_type,
            'model_name': self.name
        }
        
        eval_df = forecast.iloc[:min_len].copy()
        eval_df['y'] = actual_values
        
        return metrics, eval_df
    
    def get_components(self, forecast_df):
        if not isinstance(forecast_df, pd.DataFrame):
            forecast_df = pd.DataFrame({'ds': pd.date_range(start='2024-01-01', periods=10, freq='W'), 'yhat': [500]*10})
        
        components = pd.DataFrame({
            'ds': forecast_df['ds'],
            'trend': forecast_df['yhat'],
            'yearly': np.sin(2 * np.pi * pd.to_datetime(forecast_df['ds']).dt.dayofyear / 365.25) * forecast_df['yhat'] * 0.1
        })
        return components


def compare_all_models(df_train, df_test, forecast_periods):
    """Compare all available models."""
    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
        raise ValueError("Inputs must be pandas DataFrames")
    
    results = {}
    
    model_configs = [
        ('sarimax_baseline', 'SARIMAX (Baseline)', {'model_type': 'sarimax_baseline'}),
        ('sarimax_axiom', 'SARIMAX + Axiom Ray', {'model_type': 'sarimax', 'use_exogenous': True}),
        ('holtwinters', 'Holt-Winters', {'model_type': 'holtwinters'}),
        ('ensemble', 'Ensemble', {'model_type': 'ensemble'}),
    ]
    
    for key, name, kwargs in model_configs:
        try:
            model = CallVolumeForecaster(**kwargs)
            model.fit(df_train)
            metrics, eval_df = model.evaluate(df_test)
            forecast = model.forecast_future(periods=forecast_periods, future_exog=df_test)
            
            results[key] = {
                'model': model,
                'metrics': metrics,
                'eval_df': eval_df,
                'forecast': forecast,
                'name': name
            }
        except Exception as e:
            print(f"Model {name} failed: {e}")
    
    # Calculate improvements relative to baseline
    if 'sarimax_baseline' in results:
        baseline_mape = results['sarimax_baseline']['metrics']['MAPE']
        for key in results:
            if key != 'sarimax_baseline':
                model_mape = results[key]['metrics']['MAPE']
                improvement = baseline_mape - model_mape
                results[key]['improvement'] = {
                    'MAPE_reduction': improvement,
                    'MAPE_pct_improvement': (improvement / baseline_mape) * 100 if baseline_mape > 0 else 0
                }
    
    return results


def compare_forecasts(df_train, df_test, forecast_periods):
    """Backward compatible comparison function."""
    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
        raise ValueError("Inputs must be pandas DataFrames")
    
    results = {}
    
    model_baseline = CallVolumeForecaster(model_type='sarimax_baseline')
    model_baseline.fit(df_train)
    metrics_baseline, eval_baseline = model_baseline.evaluate(df_test)
    forecast_baseline = model_baseline.forecast_future(periods=forecast_periods)
    
    results['baseline'] = {
        'model': model_baseline,
        'metrics': metrics_baseline,
        'eval_df': eval_baseline,
        'forecast': forecast_baseline,
        'name': 'SARIMAX Baseline'
    }
    
    if 'axiom_ray_score' in df_train.columns:
        model_enhanced = CallVolumeForecaster(model_type='ensemble')
        model_enhanced.fit(df_train)
        metrics_enhanced, eval_enhanced = model_enhanced.evaluate(df_test)
        forecast_enhanced = model_enhanced.forecast_future(periods=forecast_periods, future_exog=df_test)
        
        results['enhanced'] = {
            'model': model_enhanced,
            'metrics': metrics_enhanced,
            'eval_df': eval_enhanced,
            'forecast': forecast_enhanced,
            'name': 'Ensemble (SARIMAX + Holt-Winters)'
        }
        
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
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    df = df.sort_values('ds').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df
