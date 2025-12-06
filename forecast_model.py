"""
Robust Multi-Model Forecasting with Trend and Seasonality

Models available:
1. SARIMAX with Axiom Ray as exogenous leading indicator
2. Holt-Winters Triple Exponential Smoothing (trend + seasonality)
3. LSTM (Long Short-Term Memory) - Deep learning for sequential patterns
4. Neural Network (MLP) - Feedforward network with features
5. Ensemble: Weighted average with trend continuation

Key improvements for robust forecasting:
- Explicit trend detection and projection
- Multiple seasonal period testing
- Automatic model selection based on fit
- Trend-adjusted ensemble weighting
- Deep learning models for complex patterns
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for neural networks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


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


class LSTMForecaster:
    """LSTM neural network for weekly time series forecasting."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.training_data = None
        self.last_train_date = None
        self.sequence_length = 13  # Use 13 weeks (quarter) to predict next week
        self.name = "LSTM (Deep Learning)"
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        y = np.array(df['y']).astype(float)
        n = len(y)
        
        if n < self.sequence_length + 5:
            raise ValueError(f"Need at least {self.sequence_length + 5} weeks of data")
        
        # Normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y_seq = [], []
        for i in range(self.sequence_length, n):
            X.append(y_scaled[i-self.sequence_length:i])
            y_seq.append(y_scaled[i])
        
        X = np.array(X)
        y_seq = np.array(y_seq)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(X, y_seq, epochs=50, batch_size=min(16, len(X)//2),
                      verbose=0, callbacks=[early_stop], validation_split=0.2)
        
        return self
    
    def forecast(self, periods, future_exog=None):
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted first")
        
        n_train = len(self.training_data)
        y = np.array(self.training_data['y']).astype(float)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()
        
        future_dates = pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=7),
            periods=periods, freq='W-MON'
        )
        
        # Start with last sequence
        last_sequence = y_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(periods):
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            forecasts.append(next_pred)
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts).flatten()
        
        # Confidence intervals
        residuals = y - self.scaler.inverse_transform(
            self.model.predict(self.scaler.transform(y.reshape(-1, 1))[self.sequence_length-1:], verbose=0)
        ).flatten()[:len(y) - self.sequence_length + 1]
        std_resid = np.std(residuals) if len(residuals) > 0 else np.std(y) * 0.15
        
        ci_width = 1.96 * std_resid * np.sqrt(1 + np.arange(periods) * 0.05)
        yhat_lower = forecasts - ci_width
        yhat_upper = forecasts + ci_width
        
        forecasts = np.maximum(forecasts, 100)
        yhat_lower = np.maximum(yhat_lower, 50)
        yhat_upper = np.maximum(yhat_upper, forecasts)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecasts,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })


class NeuralNetworkForecaster:
    """Feedforward neural network for weekly time series forecasting."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.training_data = None
        self.last_train_date = None
        self.feature_length = 13  # Use 13 weeks of lagged values
        self.name = "Neural Network (MLP)"
        
    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not available")
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        self.last_train_date = df['ds'].max()
        
        y = np.array(df['y']).astype(float)
        n = len(y)
        
        if n < self.feature_length + 5:
            raise ValueError(f"Need at least {self.feature_length + 5} weeks of data")
        
        # Create features
        X_list = []
        y_list = []
        
        for i in range(self.feature_length, n):
            # Lagged values
            lag_features = y[i-self.feature_length:i]
            
            # Week of year (seasonal)
            week_of_year = pd.to_datetime(df['ds'].iloc[i]).isocalendar()[1] / 52.0
            seasonal_feature = np.sin(2 * np.pi * week_of_year)
            seasonal_feature2 = np.cos(2 * np.pi * week_of_year)
            
            # Trend
            trend = (i - self.feature_length) / n
            
            # Combine features
            features = np.concatenate([
                lag_features,
                [seasonal_feature, seasonal_feature2, trend]
            ])
            
            X_list.append(features)
            y_list.append(y[i])
        
        X = np.array(X_list)
        y_seq = np.array(y_list)
        
        # Normalize
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.target_scaler = {'mean': y_seq.mean(), 'std': y_seq.std()}
        y_scaled = (y_seq - self.target_scaler['mean']) / (self.target_scaler['std'] + 1e-8)
        
        # Build model
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(X_scaled, y_scaled, epochs=50, batch_size=min(16, len(X)//2),
                      verbose=0, callbacks=[early_stop], validation_split=0.2)
        
        return self
    
    def forecast(self, periods, future_exog=None):
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted first")
        
        n_train = len(self.training_data)
        y = np.array(self.training_data['y']).astype(float)
        
        future_dates = pd.date_range(
            start=self.last_train_date + pd.Timedelta(days=7),
            periods=periods, freq='W-MON'
        )
        
        forecasts = []
        last_y = y[-self.feature_length:].tolist()
        
        for i in range(periods):
            # Features
            lag_features = np.array(last_y[-self.feature_length:])
            
            forecast_date = future_dates[i]
            week_of_year = forecast_date.isocalendar()[1] / 52.0
            seasonal_feature = np.sin(2 * np.pi * week_of_year)
            seasonal_feature2 = np.cos(2 * np.pi * week_of_year)
            trend = (n_train + i) / (n_train + periods)
            
            features = np.concatenate([lag_features, [seasonal_feature, seasonal_feature2, trend]]).reshape(1, -1)
            
            # Scale and predict
            features_scaled = self.scaler.transform(features)
            pred_scaled = self.model.predict(features_scaled, verbose=0)[0, 0]
            
            # Inverse transform
            pred = pred_scaled * self.target_scaler['std'] + self.target_scaler['mean']
            forecasts.append(max(pred, 100))
            
            # Update
            last_y.append(pred)
            if len(last_y) > self.feature_length * 2:
                last_y = last_y[-self.feature_length * 2:]
        
        forecasts = np.array(forecasts)
        
        # Confidence intervals
        std_resid = self.target_scaler['std'] * 0.15
        ci_width = 1.96 * std_resid * np.sqrt(1 + np.arange(periods) * 0.05)
        yhat_lower = forecasts - ci_width
        yhat_upper = forecasts + ci_width
        
        forecasts = np.maximum(forecasts, 100)
        yhat_lower = np.maximum(yhat_lower, 50)
        yhat_upper = np.maximum(yhat_upper, forecasts)
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecasts,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper
        })


class EnsembleForecaster:
    """Smart ensemble that combines models based on their strengths."""
    
    def __init__(self, weights=None):
        self.sarimax = SARIMAXForecaster(use_exogenous=True)
        self.holtwinters = HoltWintersForecaster()
        self.lstm = None
        self.nn = None
        self.weights = weights or {'sarimax': 0.5, 'holtwinters': 0.5}
        self.training_data = None
        self.last_train_date = None
        self.trend_slope = 0
        self.trend_intercept = 0
        self.name = "Ensemble"
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
        
        # Fit base models
        self.sarimax.fit(df)
        self.holtwinters.fit(df)
        
        # Try to fit neural network models if TensorFlow available
        # Temporarily disabled for faster training
        # if TENSORFLOW_AVAILABLE and len(df) >= 30:
        #     try:
        #         self.lstm = LSTMForecaster()
        #         self.lstm.fit(df)
        #     except Exception:
        #         self.lstm = None
        #     
        #     try:
        #         self.nn = NeuralNetworkForecaster()
        #         self.nn.fit(df)
        #     except Exception:
        #         self.nn = None
        
        # Cross-validation to determine weights
        n = len(df)
        val_size = max(4, int(n * 0.15))
        train_cv = df.iloc[:-val_size].copy()
        val_cv = df.iloc[-val_size:].copy()
        
        sarimax_mape = 50
        hw_mape = 50
        lstm_mape = 50
        nn_mape = 50
        
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
        
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_cv = LSTMForecaster()
                lstm_cv.fit(train_cv)
                lstm_pred = lstm_cv.forecast(val_size)
                actual = np.array(val_cv['y'])
                lstm_mape = np.mean(np.abs((actual - np.array(lstm_pred['yhat'])) / np.maximum(actual, 1))) * 100
            except Exception:
                pass
            
            try:
                nn_cv = NeuralNetworkForecaster()
                nn_cv.fit(train_cv)
                nn_pred = nn_cv.forecast(val_size)
                actual = np.array(val_cv['y'])
                nn_mape = np.mean(np.abs((actual - np.array(nn_pred['yhat'])) / np.maximum(actual, 1))) * 100
            except Exception:
                pass
        
        self.model_performance = {
            'sarimax_mape': sarimax_mape,
            'holtwinters_mape': hw_mape,
            'lstm_mape': lstm_mape if self.lstm else None,
            'nn_mape': nn_mape if self.nn else None
        }
        
        # Calculate weights (inversely proportional to error)
        weights_dict = {
            'sarimax': 1 / max(sarimax_mape, 1),
            'holtwinters': 1 / max(hw_mape, 1)
        }
        
        if self.lstm:
            weights_dict['lstm'] = 1 / max(lstm_mape, 1)
        if self.nn:
            weights_dict['nn'] = 1 / max(nn_mape, 1)
        
        total_inv_error = sum(weights_dict.values())
        self.weights = {k: v / total_inv_error for k, v in weights_dict.items()}
        
        # Update name
        model_names = []
        if 'sarimax' in self.weights:
            model_names.append('SARIMAX')
        if 'holtwinters' in self.weights:
            model_names.append('Holt-Winters')
        if 'lstm' in self.weights:
            model_names.append('LSTM')
        if 'nn' in self.weights:
            model_names.append('NN')
        self.name = f"Ensemble ({'+'.join(model_names)})"
        
        return self
    
    def forecast(self, periods, future_exog=None):
        sarimax_forecast = self.sarimax.forecast(periods, future_exog)
        hw_forecast = self.holtwinters.forecast(periods)
        
        yhat = (self.weights.get('sarimax', 0) * np.array(sarimax_forecast['yhat']) +
                self.weights.get('holtwinters', 0) * np.array(hw_forecast['yhat']))
        yhat_lower = (self.weights.get('sarimax', 0) * np.array(sarimax_forecast['yhat_lower']) +
                      self.weights.get('holtwinters', 0) * np.array(hw_forecast['yhat_lower']))
        yhat_upper = (self.weights.get('sarimax', 0) * np.array(sarimax_forecast['yhat_upper']) +
                      self.weights.get('holtwinters', 0) * np.array(hw_forecast['yhat_upper']))
        
        # Add LSTM if available
        if self.lstm and 'lstm' in self.weights:
            try:
                lstm_forecast = self.lstm.forecast(periods)
                w_lstm = self.weights['lstm']
                yhat += w_lstm * np.array(lstm_forecast['yhat'])
                yhat_lower += w_lstm * np.array(lstm_forecast['yhat_lower'])
                yhat_upper += w_lstm * np.array(lstm_forecast['yhat_upper'])
            except Exception:
                pass
        
        # Add Neural Network if available
        if self.nn and 'nn' in self.weights:
            try:
                nn_forecast = self.nn.forecast(periods)
                w_nn = self.weights['nn']
                yhat += w_nn * np.array(nn_forecast['yhat'])
                yhat_lower += w_nn * np.array(nn_forecast['yhat_lower'])
                yhat_upper += w_nn * np.array(nn_forecast['yhat_upper'])
            except Exception:
                pass
        
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
        elif model_type == 'lstm':
            if not TENSORFLOW_AVAILABLE:
                raise ValueError("TensorFlow not available for LSTM model")
            self.model = LSTMForecaster()
        elif model_type == 'nn':
            if not TENSORFLOW_AVAILABLE:
                raise ValueError("TensorFlow not available for Neural Network model")
            self.model = NeuralNetworkForecaster()
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
    
    # Add neural network models if TensorFlow available
    # Temporarily disabled for faster training - can be enabled if needed
    # if TENSORFLOW_AVAILABLE and len(df_train) >= 30:
    #     model_configs.extend([
    #         ('lstm', 'LSTM (Deep Learning)', {'model_type': 'lstm'}),
    #         ('nn', 'Neural Network (MLP)', {'model_type': 'nn'}),
    #     ])
    
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
