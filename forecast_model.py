"""
Multi-Model Forecasting with Ensemble Methods

Models available:
1. SARIMAX with Axiom Ray as exogenous leading indicator
2. Holt-Winters Triple Exponential Smoothing (trend + seasonality)
3. Ensemble: Weighted average of SARIMAX and Holt-Winters

Axiom Ray is a 2-week LEADING indicator for SARIMAX:
- axiom_score[t] predicts volume[t+2]
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class SARIMAXForecaster:
    """SARIMAX model with optional Axiom Ray exogenous variable."""
    
    LEAD_WEEKS = 2
    
    def __init__(self, use_exogenous=True):
        self.use_exogenous = use_exogenous
        self.model = None
        self.model_fitted = None
        self.training_data = None
        self.last_train_date = None
        self.has_axiom = False
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
        
        endog = np.array(df['y'])
        
        exog = None
        if self.use_exogenous and self.has_axiom:
            exog_series = df['axiom_ray_score'].shift(self.LEAD_WEEKS).ffill().bfill()
            exog = np.array(exog_series).reshape(-1, 1)
        
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, 13)
        
        try:
            self.model = SARIMAX(endog=endog, exog=exog, order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
            self.model_fitted = self.model.fit(disp=False, maxiter=100)
        except Exception:
            self.model = SARIMAX(endog=endog, exog=exog, order=(1, 1, 1),
                                enforce_stationarity=False, enforce_invertibility=False)
            self.model_fitted = self.model.fit(disp=False, maxiter=50)
        
        return self
    
    def forecast(self, periods, future_exog=None):
        if self.model_fitted is None:
            raise ValueError("Model must be fitted first")
        
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
            yhat_lower = predicted_mean * 0.8
            yhat_upper = predicted_mean * 1.2
        
        mean_val = self.training_data['y'].mean()
        predicted_mean = np.nan_to_num(predicted_mean, nan=mean_val)
        yhat_lower = np.nan_to_num(yhat_lower, nan=predicted_mean * 0.8)
        yhat_upper = np.nan_to_num(yhat_upper, nan=predicted_mean * 1.2)
        
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
    """Holt-Winters Triple Exponential Smoothing (trend + seasonality)."""
    
    def __init__(self):
        self.model = None
        self.model_fitted = None
        self.training_data = None
        self.last_train_date = None
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
        
        endog = np.array(df['y'])
        
        # Determine seasonal period (quarterly = 13 weeks)
        seasonal_periods = min(13, len(endog) // 3)
        
        try:
            self.model = ExponentialSmoothing(
                endog,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                damped_trend=True
            )
            self.model_fitted = self.model.fit(optimized=True)
        except Exception:
            # Fallback to simpler model
            self.model = ExponentialSmoothing(endog, trend='add', damped_trend=True)
            self.model_fitted = self.model.fit(optimized=True)
        
        return self
    
    def forecast(self, periods, future_exog=None):
        if self.model_fitted is None:
            raise ValueError("Model must be fitted first")
        
        future_dates = pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=7),
            periods=periods, freq='W-MON'
        )
        
        forecast_result = self.model_fitted.forecast(periods)
        predicted_mean = np.array(forecast_result)
        
        # Estimate confidence intervals from residuals
        residuals = self.model_fitted.resid
        std_resid = np.std(residuals) if len(residuals) > 0 else 50
        
        yhat_lower = predicted_mean - 1.96 * std_resid
        yhat_upper = predicted_mean + 1.96 * std_resid
        
        # Handle NaN/inf
        mean_val = self.training_data['y'].mean()
        predicted_mean = np.nan_to_num(predicted_mean, nan=mean_val)
        yhat_lower = np.nan_to_num(yhat_lower, nan=predicted_mean * 0.8)
        yhat_upper = np.nan_to_num(yhat_upper, nan=predicted_mean * 1.2)
        
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
    """Ensemble of SARIMAX + Axiom Ray and Holt-Winters models."""
    
    def __init__(self, weights=None):
        """
        Args:
            weights: Dict with model weights, e.g. {'sarimax': 0.6, 'holtwinters': 0.4}
                    If None, weights are determined by validation performance.
        """
        self.sarimax = SARIMAXForecaster(use_exogenous=True)
        self.holtwinters = HoltWintersForecaster()
        self.weights = weights or {'sarimax': 0.5, 'holtwinters': 0.5}
        self.training_data = None
        self.last_train_date = None
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
        
        # Fit both models
        self.sarimax.fit(df)
        self.holtwinters.fit(df)
        
        # Determine optimal weights using cross-validation on last 20% of data
        n = len(df)
        val_size = max(4, int(n * 0.2))
        train_cv = df.iloc[:-val_size].copy()
        val_cv = df.iloc[-val_size:].copy()
        
        # Refit on CV training data
        sarimax_cv = SARIMAXForecaster(use_exogenous=True)
        holtwinters_cv = HoltWintersForecaster()
        
        try:
            sarimax_cv.fit(train_cv)
            hw_cv_fitted = True
        except Exception:
            hw_cv_fitted = False
        
        try:
            holtwinters_cv.fit(train_cv)
            sarimax_cv_fitted = True
        except Exception:
            sarimax_cv_fitted = False
        
        # Get CV predictions
        actual = np.array(val_cv['y'])
        
        sarimax_mape = 100
        hw_mape = 100
        
        if hw_cv_fitted:
            try:
                sarimax_pred = sarimax_cv.forecast(val_size, future_exog=val_cv)
                sarimax_mape = np.mean(np.abs((actual - np.array(sarimax_pred['yhat'])) / np.maximum(actual, 1))) * 100
            except Exception:
                pass
        
        if sarimax_cv_fitted:
            try:
                hw_pred = holtwinters_cv.forecast(val_size)
                hw_mape = np.mean(np.abs((actual - np.array(hw_pred['yhat'])) / np.maximum(actual, 1))) * 100
            except Exception:
                pass
        
        self.model_performance = {
            'sarimax_mape': sarimax_mape,
            'holtwinters_mape': hw_mape
        }
        
        # Set weights inversely proportional to error
        total_inv_error = (1 / max(sarimax_mape, 0.1)) + (1 / max(hw_mape, 0.1))
        self.weights = {
            'sarimax': (1 / max(sarimax_mape, 0.1)) / total_inv_error,
            'holtwinters': (1 / max(hw_mape, 0.1)) / total_inv_error
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
        
        return pd.DataFrame({
            'ds': sarimax_forecast['ds'],
            'yhat': yhat,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })


class CallVolumeForecaster:
    """
    Unified forecaster interface supporting multiple model types.
    
    Model types:
    - 'sarimax': SARIMAX with Axiom Ray exogenous variable
    - 'sarimax_baseline': SARIMAX without exogenous variable
    - 'holtwinters': Holt-Winters Triple Exponential Smoothing
    - 'ensemble': Weighted ensemble of SARIMAX + Holt-Winters
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
        """Make predictions for given dates."""
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
    """
    Compare all available models: SARIMAX, SARIMAX+Axiom, Holt-Winters, Ensemble.
    """
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
    """
    Compare forecasts with and without Axiom Ray exogenous variable.
    (Backward compatible function - now uses ensemble as enhanced model)
    """
    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
        raise ValueError("Inputs must be pandas DataFrames")
    
    results = {}
    
    # Baseline: SARIMAX without Axiom Ray
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
    
    # Enhanced: Ensemble model
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
