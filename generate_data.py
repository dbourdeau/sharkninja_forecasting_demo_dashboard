"""
Generate synthetic data for SharkNinja Customer Support Volume Forecasting.

Axiom Ray is a LEADING indicator - it predicts volume 2 weeks ahead.
This simulates an AI system monitoring social media, reviews, and warranty
claims that detects issues BEFORE they result in support calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_call_volume_data(start_date='2021-01-04', weeks=156):
    """
    Generate realistic call center volume data for SharkNinja.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=weeks, freq='W-MON')
    t = np.arange(weeks)
    
    # BASE TREND - Linear growth
    base_level = 450
    growth_rate = 200 / weeks
    trend = base_level + growth_rate * t
    
    # ANNUAL SEASONALITY
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    annual_season = 100 * np.cos(2 * np.pi * (day_of_year - 340) / 365.25)
    annual_season += 30 * np.cos(4 * np.pi * (day_of_year - 340) / 365.25)
    
    # MONTHLY EFFECTS
    months = np.array([d.month for d in dates])
    monthly_effect = np.zeros(weeks)
    monthly_effect[months == 1] = 60   # January returns
    monthly_effect[months == 3] = 30   # Spring cleaning
    monthly_effect[months == 4] = 20
    monthly_effect[months == 8] = 25   # Back to school
    
    # SPECIAL EVENTS
    events = np.zeros(weeks)
    launch_weeks = [28, 80, 132]
    for launch in launch_weeks:
        if launch < weeks:
            events[launch] = 100
            if launch + 1 < weeks:
                events[launch + 1] = 50
    
    for year_start in [0, 52, 104]:
        bf_week = year_start + 47
        if bf_week < weeks:
            events[bf_week] = 80
    
    # Random spikes
    np.random.seed(42)
    random_spike_weeks = np.random.choice(weeks, size=8, replace=False)
    for spike in random_spike_weeks:
        events[spike] += np.random.uniform(40, 120)
    
    # NOISE
    white_noise = np.random.normal(0, 70, weeks)
    ar_noise = np.zeros(weeks)
    ar_noise[0] = np.random.normal(0, 30)
    for i in range(1, weeks):
        ar_noise[i] = 0.5 * ar_noise[i-1] + np.random.normal(0, 25)
    
    random_walk = np.cumsum(np.random.normal(0, 12, weeks))
    random_walk = random_walk - np.linspace(random_walk[0], random_walk[-1], weeks)
    
    # COMBINE
    volume = trend + annual_season + monthly_effect + events + white_noise + ar_noise + random_walk
    volume = np.maximum(volume, 250)
    volume = np.round(volume).astype(int)
    
    # PRODUCT BREAKDOWN
    shark_pct = np.zeros(weeks)
    for i, month in enumerate(months):
        if month in [3, 4]:
            shark_pct[i] = 0.60
        elif month in [5, 6, 7]:
            shark_pct[i] = 0.45
        elif month in [11, 12]:
            shark_pct[i] = 0.50
        else:
            shark_pct[i] = 0.52
    
    shark_pct = shark_pct + np.random.normal(0, 0.05, weeks)
    shark_pct = np.clip(shark_pct, 0.35, 0.65)
    
    df = pd.DataFrame({
        'ds': dates,
        'y': volume,
        'shark_volume': (volume * shark_pct).astype(int),
        'ninja_volume': (volume * (1 - shark_pct)).astype(int)
    })
    
    # Subcategories
    df['shark_vacuums'] = (df['shark_volume'] * 0.70).astype(int)
    df['shark_haircare'] = (df['shark_volume'] * 0.15).astype(int)
    df['shark_airpurifiers'] = df['shark_volume'] - df['shark_vacuums'] - df['shark_haircare']
    
    df['ninja_airfryers'] = (df['ninja_volume'] * 0.35).astype(int)
    df['ninja_blenders'] = (df['ninja_volume'] * 0.25).astype(int)
    df['ninja_coffee'] = (df['ninja_volume'] * 0.20).astype(int)
    df['ninja_grills'] = df['ninja_volume'] - df['ninja_airfryers'] - df['ninja_blenders'] - df['ninja_coffee']
    
    return df


def generate_axiom_ray_signals(volume_df, lead_weeks=2, correlation_strength=0.70):
    """
    Generate Axiom Ray AI early warning signals as a LEADING indicator.
    
    The Axiom Ray score at week T predicts call volume at week T+lead_weeks.
    This simulates an AI system that detects issues (via social media, reviews,
    warranty claims) BEFORE they result in support calls.
    
    Args:
        volume_df: DataFrame with volume data
        lead_weeks: How many weeks ahead Axiom Ray predicts (default 2)
        correlation_strength: Target correlation with future volume (0.6-0.8 realistic)
    
    Returns:
        DataFrame with Axiom Ray signals
    """
    np.random.seed(45)
    weeks = len(volume_df)
    volume = volume_df['y'].values
    
    # Shift volume BACKWARD to create leading indicator
    # axiom_score[t] should correlate with volume[t + lead_weeks]
    # So we base axiom_score[t] on volume[t + lead_weeks]
    future_volume = np.roll(volume, -lead_weeks)
    # Fill the last lead_weeks with extrapolated values
    for i in range(lead_weeks):
        future_volume[-(i+1)] = volume[-1] + (i+1) * (volume[-1] - volume[-2])
    
    # Normalize future volume to 0-100 scale
    vol_min, vol_max = future_volume.min(), future_volume.max()
    normalized_future = (future_volume - vol_min) / (vol_max - vol_min) * 100
    
    # Create base Axiom score with controlled correlation
    # Mix of future volume signal and noise
    signal_component = normalized_future * correlation_strength
    noise_component = np.random.normal(50, 15, weeks) * (1 - correlation_strength)
    
    axiom_base = signal_component + noise_component
    
    # Add autocorrelation to make it more realistic (smooth week-to-week)
    axiom_smooth = np.zeros(weeks)
    axiom_smooth[0] = axiom_base[0]
    for i in range(1, weeks):
        axiom_smooth[i] = 0.3 * axiom_smooth[i-1] + 0.7 * axiom_base[i]
    
    # Clip to 0-100 and round
    axiom_score = np.clip(axiom_smooth, 0, 100).astype(int)
    
    # Generate component signals (also leading indicators)
    # Each component adds noise independently
    social_sentiment = axiom_score + np.random.normal(0, 10, weeks)
    review_trend = axiom_score + np.random.normal(0, 12, weeks)
    warranty_claims = axiom_score + np.random.normal(0, 8, weeks)
    search_trends = axiom_score + np.random.normal(0, 15, weeks)
    retail_alerts = axiom_score + np.random.normal(0, 9, weeks)
    
    return pd.DataFrame({
        'ds': volume_df['ds'],
        'axiom_ray_score': axiom_score,
        'social_sentiment': np.clip(social_sentiment, 0, 100).astype(int),
        'review_trend': np.clip(review_trend, 0, 100).astype(int),
        'warranty_claims': np.clip(warranty_claims, 0, 100).astype(int),
        'search_trends': np.clip(search_trends, 0, 100).astype(int),
        'retail_alerts': np.clip(retail_alerts, 0, 100).astype(int)
    })


def generate_business_metrics(volume_df):
    """Generate operational metrics that correlate with volume."""
    np.random.seed(44)
    weeks = len(volume_df)
    vol_norm = (volume_df['y'] - volume_df['y'].mean()) / volume_df['y'].std()
    
    base_sl = 0.82
    service_level = base_sl - vol_norm * 0.04 + np.random.normal(0, 0.02, weeks)
    service_level = np.clip(service_level, 0.68, 0.95)
    
    base_aht = 4.5
    aht = base_aht + vol_norm * 0.3 + np.random.normal(0, 0.2, weeks)
    aht = np.clip(aht, 3.5, 6.5)
    
    base_fcr = 0.76
    fcr = base_fcr - vol_norm * 0.02 + np.random.normal(0, 0.03, weeks)
    fcr = np.clip(fcr, 0.65, 0.88)
    
    base_csat = 82
    csat = base_csat + (service_level - base_sl) * 50 + np.random.normal(0, 2, weeks)
    csat = np.clip(csat, 70, 95).astype(int)
    
    return pd.DataFrame({
        'ds': volume_df['ds'],
        'service_level': service_level,
        'avg_handle_time': aht,
        'first_call_resolution': fcr,
        'customer_satisfaction': csat
    })


def main():
    """Generate and save all datasets."""
    print("=" * 60)
    print("Generating SharkNinja Customer Support Data")
    print("=" * 60)
    
    # Generate main volume data
    print("\n1. Generating call volume with trend & seasonality...")
    volume_df = generate_call_volume_data(weeks=156)
    
    # Generate Axiom Ray AI signals (2-week leading indicator)
    print("2. Generating Axiom Ray AI signals (2-week lead time)...")
    axiom_df = generate_axiom_ray_signals(volume_df, lead_weeks=2, correlation_strength=0.70)
    
    # Generate business metrics
    print("3. Generating operational metrics...")
    metrics_df = generate_business_metrics(volume_df)
    
    # Combine all data
    combined_df = volume_df.merge(axiom_df, on='ds').merge(metrics_df, on='ds')
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save files
    volume_df[['ds', 'y']].to_csv('data/historical_volume.csv', index=False)
    axiom_df.to_csv('data/axiom_ray_predictions.csv', index=False)
    metrics_df.to_csv('data/business_metrics.csv', index=False)
    combined_df.to_csv('data/combined_data.csv', index=False)
    
    # Validate leading indicator relationship
    print("\n" + "=" * 60)
    print("AXIOM RAY LEADING INDICATOR VALIDATION")
    print("=" * 60)
    
    # Check correlation with future volume (should be ~0.70)
    axiom = combined_df['axiom_ray_score'].values
    volume = combined_df['y'].values
    
    # Concurrent correlation (should be lower)
    concurrent_corr = np.corrcoef(axiom[:-2], volume[:-2])[0,1]
    
    # Leading correlation (axiom[t] vs volume[t+2]) - should be higher
    leading_corr = np.corrcoef(axiom[:-2], volume[2:])[0,1]
    
    print(f"\nCorrelation Analysis:")
    print(f"  Axiom[t] vs Volume[t] (concurrent): {concurrent_corr:.3f}")
    print(f"  Axiom[t] vs Volume[t+2] (2-week lead): {leading_corr:.3f}")
    print(f"  Lead improvement: {(leading_corr - concurrent_corr):.3f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Period: {volume_df['ds'].min().strftime('%Y-%m-%d')} to {volume_df['ds'].max().strftime('%Y-%m-%d')}")
    print(f"Total weeks: {len(volume_df)}")
    print(f"\nVolume Statistics:")
    print(f"  Mean: {volume_df['y'].mean():.0f} calls/week")
    print(f"  Std:  {volume_df['y'].std():.0f} calls/week")
    print(f"\nAxiom Ray Score Statistics:")
    print(f"  Mean: {axiom_df['axiom_ray_score'].mean():.0f}")
    print(f"  Std:  {axiom_df['axiom_ray_score'].std():.0f}")
    print(f"  Range: {axiom_df['axiom_ray_score'].min()}-{axiom_df['axiom_ray_score'].max()}")
    print("\n" + "=" * 60)
    print("Files saved to data/ directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
