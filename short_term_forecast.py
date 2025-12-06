"""
Short-Term (Intraday/Daily) Forecasting Module

Provides 5-day ahead forecasts for immediate staffing decisions.
Uses different techniques optimized for short horizons:
- Simple Exponential Smoothing
- Moving Average with Day-of-Week patterns
- ARIMA for short-term dynamics
- LSTM (Long Short-Term Memory) - Deep learning approach
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def generate_daily_data(weekly_df, days_back=90):
    """
    Convert weekly data to daily data with realistic intraday patterns.
    
    Daily patterns:
    - Monday: High volume (weekend backlog)
    - Tuesday-Wednesday: Peak days
    - Thursday: Moderate
    - Friday: Lower (weekend approaching)
    - Saturday-Sunday: Low volume
    """
    np.random.seed(42)
    
    # Get the last date from weekly data
    last_weekly_date = pd.to_datetime(weekly_df['ds'].max())
    
    # Generate daily dates for the past N days
    end_date = last_weekly_date
    start_date = end_date - timedelta(days=days_back)
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Day-of-week multipliers (0=Monday, 6=Sunday)
    dow_multipliers = {
        0: 1.15,  # Monday - high (backlog)
        1: 1.20,  # Tuesday - peak
        2: 1.18,  # Wednesday - peak
        3: 1.05,  # Thursday - moderate
        4: 0.90,  # Friday - lower
        5: 0.35,  # Saturday - low
        6: 0.25,  # Sunday - lowest
    }
    
    # Get average weekly volume from recent weeks
    recent_weekly_avg = weekly_df['y'].tail(13).mean()
    daily_base = recent_weekly_avg / 5  # Approximate daily from weekly (5 working days equiv)
    
    # Generate daily volumes
    daily_volumes = []
    for date in daily_dates:
        dow = date.dayofweek
        multiplier = dow_multipliers[dow]
        
        # Add some trend (slight growth)
        days_from_start = (date - start_date).days
        trend_factor = 1 + (days_from_start / days_back) * 0.05
        
        # Add noise
        noise = np.random.normal(0, daily_base * 0.12)
        
        volume = daily_base * multiplier * trend_factor + noise
        volume = max(volume, 20)  # Minimum floor
        daily_volumes.append(int(round(volume)))
    
    # Create daily dataframe
    daily_df = pd.DataFrame({
        'ds': daily_dates,
        'y': daily_volumes,
        'day_of_week': [d.dayofweek for d in daily_dates],
        'day_name': [d.strftime('%A') for d in daily_dates],
        'is_weekend': [d.dayofweek >= 5 for d in daily_dates]
    })
    
    return daily_df


class ShortTermForecaster:
    """
    Short-term forecaster for 1-5 day ahead predictions.
    """
    
    def __init__(self, method='ensemble'):
        """
        Args:
            method: 'ses', 'arima', 'dow_avg', 'lstm', or 'ensemble'
        """
        self.method = method
        self.training_data = None
        self.dow_patterns = None
        self.models = {}
        self.scaler = None
        self.lstm_sequence_length = 7  # Use 7 days to predict next day
        
    def fit(self, daily_df):
        """Fit the short-term forecasting models."""
        if not isinstance(daily_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = daily_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        self.training_data = df.copy()
        
        y = np.array(df['y']).astype(float)
        
        # Learn day-of-week patterns
        self.dow_patterns = df.groupby('day_of_week')['y'].mean().to_dict()
        
        # Fit Simple Exponential Smoothing
        try:
            ses_model = SimpleExpSmoothing(y).fit(smoothing_level=0.3, optimized=False)
            self.models['ses'] = ses_model
        except Exception:
            self.models['ses'] = None
        
        # Fit ARIMA (short-term dynamics)
        try:
            arima_model = ARIMA(y, order=(2, 0, 1)).fit()
            self.models['arima'] = arima_model
        except Exception:
            self.models['arima'] = None
        
        # Fit Holt-Winters with weekly seasonality
        try:
            if len(y) >= 14:  # Need at least 2 weeks
                hw_model = ExponentialSmoothing(
                    y, trend='add', seasonal='add', 
                    seasonal_periods=7, damped_trend=True
                ).fit()
                self.models['hw'] = hw_model
            else:
                self.models['hw'] = None
        except Exception:
            self.models['hw'] = None
        
        # Fit LSTM (if TensorFlow available and enough data)
        if TENSORFLOW_AVAILABLE and len(y) >= 21:  # Need at least 3 weeks
            try:
                lstm_model, scaler = self._fit_lstm(y)
                self.models['lstm'] = lstm_model
                self.scaler = scaler
            except Exception as e:
                self.models['lstm'] = None
                self.scaler = None
        else:
            self.models['lstm'] = None
            self.scaler = None
        
        # Fit Feedforward Neural Network (MLP)
        if TENSORFLOW_AVAILABLE and len(y) >= 14:
            try:
                nn_model, nn_scaler = self._fit_neural_network(df)
                self.models['nn'] = nn_model
                self.nn_scaler = nn_scaler
            except Exception as e:
                self.models['nn'] = None
                self.nn_scaler = None
        else:
            self.models['nn'] = None
            self.nn_scaler = None
        
        return self
    
    def _fit_lstm(self, y):
        """Fit LSTM model on time series data."""
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences: use last 7 days to predict next day
        sequence_length = self.lstm_sequence_length
        X, y_seq = [], []
        
        for i in range(sequence_length, len(y_scaled)):
            X.append(y_scaled[i-sequence_length:i])
            y_seq.append(y_scaled[i])
        
        X = np.array(X)
        y_seq = np.array(y_seq)
        
        # Reshape for LSTM: [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train with minimal epochs for small dataset
        model.fit(X, y_seq, epochs=30, batch_size=min(16, len(X)//2), 
                 verbose=0, callbacks=[early_stop], validation_split=0.2)
        
        return model, scaler
    
    def _forecast_lstm(self, days=5):
        """Generate LSTM forecast."""
        if self.models.get('lstm') is None or self.scaler is None:
            return None
        
        model = self.models['lstm']
        scaler = self.scaler
        
        # Get last sequence_length days
        y = np.array(self.training_data['y']).astype(float)
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Start with last sequence
        last_sequence = y_scaled[-self.lstm_sequence_length:].reshape(1, self.lstm_sequence_length, 1)
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # Update sequence: remove first, add prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts).flatten()
        
        return forecasts
    
    def _fit_neural_network(self, df):
        """Fit feedforward neural network on time series with features."""
        # Create features: lagged values, day of week, trend
        feature_length = 7  # Use last 7 days
        
        X_list = []
        y_list = []
        
        y = np.array(df['y']).astype(float)
        dow = np.array(df['day_of_week'])
        
        for i in range(feature_length, len(y)):
            # Lagged values (last 7 days)
            lag_features = y[i-feature_length:i]
            
            # Day of week (one-hot encoded)
            dow_onehot = np.zeros(7)
            dow_onehot[dow[i]] = 1
            
            # Recent trend (3-day moving average vs 7-day)
            ma3 = np.mean(y[i-3:i]) if i >= 3 else y[i-1]
            ma7 = np.mean(y[i-7:i])
            trend_feature = ma3 - ma7
            
            # Combine all features
            features = np.concatenate([
                lag_features,
                dow_onehot,
                [trend_feature]
            ])
            
            X_list.append(features)
            y_list.append(y[i])
        
        X = np.array(X_list)
        y_seq = np.array(y_list)
        
        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Normalize target
        y_scaled = (y_seq - y_seq.mean()) / (y_seq.std() + 1e-8)
        
        # Build feedforward neural network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=min(16, len(X)//2),
                 verbose=0, callbacks=[early_stop], validation_split=0.2)
        
        # Store scaler and normalization params for inverse transform
        nn_scaler = {
            'feature_scaler': scaler,
            'y_mean': y_seq.mean(),
            'y_std': y_seq.std()
        }
        
        return model, nn_scaler
    
    def _forecast_neural_network(self, days=5):
        """Generate neural network forecast."""
        if self.models.get('nn') is None or self.nn_scaler is None:
            return None
        
        model = self.models['nn']
        scaler = self.nn_scaler
        df = self.training_data
        
        y = np.array(df['y']).astype(float)
        dow = np.array(df['day_of_week'])
        
        forecasts = []
        
        # Get last values for initial features
        last_y = y[-7:].tolist()
        last_dow = dow[-7:].tolist()
        
        for i in range(days):
            # Determine day of week for forecast day
            last_date = df['ds'].max()
            forecast_date = last_date + timedelta(days=i+1)
            forecast_dow = forecast_date.dayofweek
            
            # Prepare features
            lag_features = np.array(last_y[-7:])
            
            dow_onehot = np.zeros(7)
            dow_onehot[forecast_dow] = 1
            
            # Trend feature
            if len(last_y) >= 7:
                ma3 = np.mean(last_y[-3:])
                ma7 = np.mean(last_y[-7:])
            else:
                ma3 = np.mean(last_y)
                ma7 = np.mean(last_y)
            trend_feature = ma3 - ma7
            
            features = np.concatenate([lag_features, dow_onehot, [trend_feature]]).reshape(1, -1)
            
            # Scale and predict
            features_scaled = scaler['feature_scaler'].transform(features)
            pred_scaled = model.predict(features_scaled, verbose=0)[0, 0]
            
            # Inverse transform
            pred = pred_scaled * scaler['y_std'] + scaler['y_mean']
            forecasts.append(max(pred, 10))  # Minimum floor
            
            # Update last_y for next iteration
            last_y.append(pred)
            if len(last_y) > 14:
                last_y = last_y[-14:]
        
        return np.array(forecasts)
    
    def forecast(self, days=5):
        """Generate forecast for next N days."""
        if self.training_data is None:
            raise ValueError("Model must be fitted first")
        
        last_date = self.training_data['ds'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
        
        forecasts = {}
        
        # Day-of-week average forecast
        dow_forecast = []
        for date in future_dates:
            dow = date.dayofweek
            dow_forecast.append(self.dow_patterns.get(dow, self.training_data['y'].mean()))
        forecasts['dow_avg'] = np.array(dow_forecast)
        
        # SES forecast
        if self.models.get('ses'):
            ses_forecast = self.models['ses'].forecast(days)
            forecasts['ses'] = np.array(ses_forecast)
        
        # ARIMA forecast
        if self.models.get('arima'):
            arima_forecast = self.models['arima'].forecast(days)
            forecasts['arima'] = np.array(arima_forecast)
        
        # Holt-Winters forecast
        if self.models.get('hw'):
            hw_forecast = self.models['hw'].forecast(days)
            forecasts['hw'] = np.array(hw_forecast)
        
        # LSTM forecast
        if self.models.get('lstm'):
            lstm_forecast = self._forecast_lstm(days)
            if lstm_forecast is not None:
                forecasts['lstm'] = lstm_forecast
        
        # Neural Network forecast
        if self.models.get('nn'):
            nn_forecast = self._forecast_neural_network(days)
            if nn_forecast is not None:
                forecasts['nn'] = nn_forecast
        
        # Ensemble: weighted average (includes all models)
        available_forecasts = [f for f in [forecasts.get('ses'), forecasts.get('arima'), 
                                           forecasts.get('hw'), forecasts.get('lstm'),
                                           forecasts.get('nn'), forecasts.get('dow_avg')] 
                             if f is not None]
        
        if available_forecasts:
            ensemble_forecast = np.mean(available_forecasts, axis=0)
        else:
            ensemble_forecast = forecasts['dow_avg']
        
        forecasts['ensemble'] = ensemble_forecast
        
        # Select method
        if self.method in forecasts:
            predicted = forecasts[self.method]
        else:
            predicted = forecasts['ensemble']
        
        # Confidence intervals (based on recent volatility)
        recent_std = self.training_data['y'].tail(14).std()
        ci_width = 1.96 * recent_std * np.sqrt(1 + np.arange(days) * 0.1)
        
        result_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': np.maximum(predicted, 10).round(0).astype(int),
            'yhat_lower': np.maximum(predicted - ci_width, 5).round(0).astype(int),
            'yhat_upper': np.maximum(predicted + ci_width, predicted).round(0).astype(int),
            'day_name': [d.strftime('%A') for d in future_dates],
            'day_of_week': [d.dayofweek for d in future_dates]
        })
        
        # Store all model forecasts for comparison
        result_df['ses_forecast'] = forecasts.get('ses', forecasts['dow_avg']).round(0).astype(int) if 'ses' in forecasts else None
        result_df['arima_forecast'] = forecasts.get('arima', forecasts['dow_avg']).round(0).astype(int) if 'arima' in forecasts else None
        result_df['hw_forecast'] = forecasts.get('hw', forecasts['dow_avg']).round(0).astype(int) if 'hw' in forecasts else None
        result_df['lstm_forecast'] = forecasts.get('lstm', forecasts['dow_avg']).round(0).astype(int) if 'lstm' in forecasts else None
        result_df['nn_forecast'] = forecasts.get('nn', forecasts['dow_avg']).round(0).astype(int) if 'nn' in forecasts else None
        result_df['dow_forecast'] = forecasts['dow_avg'].round(0).astype(int)
        
        return result_df
    
    def evaluate(self, test_df):
        """Evaluate on test data."""
        if not isinstance(test_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        test_df = test_df.copy()
        n_test = len(test_df)
        
        # Get forecasts
        forecast = self.forecast(days=n_test)
        
        actual = np.array(test_df['y'])
        predicted = np.array(forecast['yhat'])
        
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
        
        # Within CI
        yhat_lower = np.array(forecast['yhat_lower'])[:min_len]
        yhat_upper = np.array(forecast['yhat_upper'])[:min_len]
        within_ci = np.mean((actual >= yhat_lower) & (actual <= yhat_upper)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Within_CI_%': within_ci,
            'n_days': min_len
        }
        
        eval_df = forecast.iloc[:min_len].copy()
        eval_df['y'] = actual
        
        return metrics, eval_df


def compare_short_term_models(daily_df, test_days=5):
    """Compare all short-term forecasting methods."""
    
    if daily_df is None or len(daily_df) < 14:
        raise ValueError(f"Insufficient data: need at least 14 days, got {len(daily_df) if daily_df is not None else 0}")
    
    if len(daily_df) < test_days + 7:
        raise ValueError(f"Insufficient data: need at least {test_days + 7} days for train/test split")
    
    # Split into train/test
    train_df = daily_df.iloc[:-test_days].copy()
    test_df = daily_df.iloc[-test_days:].copy()
    
    results = {}
    methods = ['ses', 'arima', 'dow_avg', 'lstm', 'nn', 'ensemble']
    
    name_map = {
        'ses': 'Simple Exp. Smoothing',
        'arima': 'ARIMA(2,0,1)',
        'dow_avg': 'Day-of-Week Average',
        'lstm': 'LSTM (Deep Learning)',
        'nn': 'Neural Network (MLP)',
        'ensemble': 'Ensemble'
    }
    
    for method in methods:
        try:
            model = ShortTermForecaster(method=method)
            model.fit(train_df)
            metrics, eval_df = model.evaluate(test_df)
            forecast = model.forecast(days=5)
            
            results[method] = {
                'model': model,
                'metrics': metrics,
                'eval_df': eval_df,
                'forecast': forecast,
                'name': name_map.get(method, method)
            }
        except Exception as e:
            print(f"Method {method} failed: {e}")
    
    return results, train_df, test_df


def get_staffing_recommendation(forecast_df, calls_per_agent_hour=8, hours_per_shift=8):
    """Convert daily forecast to staffing recommendations."""
    
    recommendations = []
    for _, row in forecast_df.iterrows():
        daily_calls = row['yhat']
        
        # Assume 10-hour operating day
        operating_hours = 10
        calls_per_hour = daily_calls / operating_hours
        
        # Agents needed per hour
        agents_per_hour = calls_per_hour / calls_per_agent_hour
        
        # Peak factor (20% buffer)
        peak_agents = int(np.ceil(agents_per_hour * 1.2))
        
        # FTE calculation
        fte_needed = (daily_calls / calls_per_agent_hour) / hours_per_shift
        
        recommendations.append({
            'date': row['ds'],
            'day': row['day_name'],
            'forecast_calls': int(row['yhat']),
            'agents_needed': peak_agents,
            'fte_needed': round(fte_needed, 1),
            'confidence': f"{int(row['yhat_lower'])} - {int(row['yhat_upper'])}"
        })
    
    return pd.DataFrame(recommendations)

