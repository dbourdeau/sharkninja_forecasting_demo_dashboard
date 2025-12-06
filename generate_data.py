"""
Generate synthetic data for SharkNinja Customer Support Volume Forecasting.

Creates realistic call center data with clear patterns for forecasting:
- Strong upward trend (business growth)
- Clear annual seasonality (holiday peaks, summer lows)
- Realistic noise (~20-25% unexplained variance)

Designed to demonstrate forecasting with ~80-88% accuracy (realistic, not perfect).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_call_volume_data(start_date='2021-01-04', weeks=156):
    """
    Generate realistic call center volume data for SharkNinja.
    
    Creates data with:
    - Clear linear trend (business growth ~30% over 3 years)
    - Strong annual seasonality (holiday peak, summer trough)
    - Significant realistic noise (~20% unexplained variance)
    
    Args:
        start_date: Start date for data
        weeks: Number of weeks (default 156 = 3 years)
    
    Returns:
        DataFrame with call volume and product breakdowns
    """
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=weeks, freq='W-MON')
    t = np.arange(weeks)
    
    # ===================
    # 1. BASE TREND
    # ===================
    # Linear growth: 450 calls/week growing to ~650 over 3 years
    base_level = 450
    growth_rate = 200 / weeks  # ~200 additional calls over 3 years
    trend = base_level + growth_rate * t
    
    # ===================
    # 2. ANNUAL SEASONALITY
    # ===================
    # Strong annual pattern with holiday peak and summer trough
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Primary annual cycle (peaks in late Nov/Dec, troughs in Jun-Jul)
    annual_season = 100 * np.cos(2 * np.pi * (day_of_year - 340) / 365.25)
    
    # Secondary harmonic for sharper patterns
    annual_season += 30 * np.cos(4 * np.pi * (day_of_year - 340) / 365.25)
    
    # ===================
    # 3. MONTHLY EFFECTS
    # ===================
    months = np.array([d.month for d in dates])
    monthly_effect = np.zeros(weeks)
    
    # January: post-holiday returns surge
    monthly_effect[months == 1] = 60
    # March-April: spring cleaning (Shark vacuums)
    monthly_effect[months == 3] = 30
    monthly_effect[months == 4] = 20
    # August: back-to-school prep
    monthly_effect[months == 8] = 25
    
    # ===================
    # 4. SPECIAL EVENTS (unpredictable spikes)
    # ===================
    events = np.zeros(weeks)
    
    # Product launches (somewhat predictable)
    launch_weeks = [28, 80, 132]
    for launch in launch_weeks:
        if launch < weeks:
            events[launch] = 100
            if launch + 1 < weeks:
                events[launch + 1] = 50
    
    # Black Friday weeks
    for year_start in [0, 52, 104]:
        bf_week = year_start + 47
        if bf_week < weeks:
            events[bf_week] = 80
    
    # Random unexpected spikes (recalls, viral issues, etc.)
    # These make the data less predictable
    random_spike_weeks = np.random.choice(weeks, size=8, replace=False)
    for spike in random_spike_weeks:
        events[spike] += np.random.uniform(40, 120)
    
    # ===================
    # 5. NOISE - Critical for realistic forecasting
    # ===================
    # Base white noise (~15% of mean)
    white_noise = np.random.normal(0, 70, weeks)
    
    # Autocorrelated noise (week-to-week persistence)
    ar_noise = np.zeros(weeks)
    ar_noise[0] = np.random.normal(0, 30)
    for i in range(1, weeks):
        ar_noise[i] = 0.5 * ar_noise[i-1] + np.random.normal(0, 25)
    
    # Random walk component (unpredictable drift)
    random_walk = np.cumsum(np.random.normal(0, 12, weeks))
    random_walk = random_walk - np.linspace(random_walk[0], random_walk[-1], weeks)  # Detrend
    
    # ===================
    # COMBINE ALL COMPONENTS
    # ===================
    volume = trend + annual_season + monthly_effect + events + white_noise + ar_noise + random_walk
    
    # Ensure positive values with realistic minimum
    volume = np.maximum(volume, 250)
    
    # Round to integers
    volume = np.round(volume).astype(int)
    
    # ===================
    # PRODUCT BREAKDOWN
    # ===================
    shark_pct = np.zeros(weeks)
    for i, month in enumerate(months):
        if month in [3, 4]:  # Spring cleaning - more Shark
            shark_pct[i] = 0.60
        elif month in [5, 6, 7]:  # Summer/grilling - more Ninja
            shark_pct[i] = 0.45
        elif month in [11, 12]:  # Holiday gifts - balanced
            shark_pct[i] = 0.50
        else:
            shark_pct[i] = 0.52
    
    # Add noise to split
    shark_pct = shark_pct + np.random.normal(0, 0.05, weeks)
    shark_pct = np.clip(shark_pct, 0.35, 0.65)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': volume,
        'shark_volume': (volume * shark_pct).astype(int),
        'ninja_volume': (volume * (1 - shark_pct)).astype(int)
    })
    
    # Subcategory breakdown
    df['shark_vacuums'] = (df['shark_volume'] * 0.70).astype(int)
    df['shark_haircare'] = (df['shark_volume'] * 0.15).astype(int)
    df['shark_airpurifiers'] = df['shark_volume'] - df['shark_vacuums'] - df['shark_haircare']
    
    df['ninja_airfryers'] = (df['ninja_volume'] * 0.35).astype(int)
    df['ninja_blenders'] = (df['ninja_volume'] * 0.25).astype(int)
    df['ninja_coffee'] = (df['ninja_volume'] * 0.20).astype(int)
    df['ninja_grills'] = df['ninja_volume'] - df['ninja_airfryers'] - df['ninja_blenders'] - df['ninja_coffee']
    
    return df


def generate_axiom_ray_signals(volume_df):
    """
    Generate Axiom Ray AI early warning signals.
    
    Axiom Ray monitors multiple data sources to predict support volume:
    - Social media sentiment (Twitter, Reddit mentions)
    - Product review trends (Amazon, retailer sites)
    - Warranty claim patterns
    - Search trends (Google Trends for product issues)
    - Retail partner feedback
    
    The signal is a composite score (0-100) that correlates with 
    future volume with ~70% correlation (realistic AI performance).
    """
    np.random.seed(45)
    weeks = len(volume_df)
    volume = volume_df['y'].values
    
    # Component 1: Social Media Sentiment Index (0-100)
    # Inversely correlated with volume (more complaints = higher volume coming)
    social_base = 50 + (volume - volume.mean()) / volume.std() * 15
    social_noise = np.random.normal(0, 8, weeks)
    social_sentiment = np.clip(social_base + social_noise, 10, 90)
    
    # Component 2: Product Review Trend (0-100)
    # Tracks negative review velocity
    review_base = 45 + (volume - volume.mean()) / volume.std() * 12
    review_noise = np.random.normal(0, 10, weeks)
    review_trend = np.clip(review_base + review_noise, 5, 95)
    
    # Component 3: Warranty Claims Index (0-100)
    # Leading indicator - spikes before volume increases
    warranty_base = 40 + (volume - volume.mean()) / volume.std() * 10
    warranty_noise = np.random.normal(0, 7, weeks)
    warranty_claims = np.clip(warranty_base + warranty_noise, 15, 85)
    
    # Component 4: Search Trend Index (0-100)
    # Google searches for "[product] problem", "[product] not working"
    search_base = 55 + (volume - volume.mean()) / volume.std() * 18
    search_noise = np.random.normal(0, 12, weeks)
    search_trends = np.clip(search_base + search_noise, 10, 95)
    
    # Component 5: Retail Partner Alerts (0-100)
    # Returns and complaints from major retailers
    retail_base = 48 + (volume - volume.mean()) / volume.std() * 14
    retail_noise = np.random.normal(0, 9, weeks)
    retail_alerts = np.clip(retail_base + retail_noise, 5, 90)
    
    # Composite Axiom Ray Score (weighted average)
    # Weights reflect predictive power of each signal
    axiom_score = (
        social_sentiment * 0.20 +
        review_trend * 0.25 +
        warranty_claims * 0.20 +
        search_trends * 0.20 +
        retail_alerts * 0.15
    )
    
    # Add some overall noise to prevent perfect correlation
    axiom_score = axiom_score + np.random.normal(0, 5, weeks)
    axiom_score = np.clip(axiom_score, 0, 100).astype(int)
    
    return pd.DataFrame({
        'ds': volume_df['ds'],
        'axiom_ray_score': axiom_score,
        'social_sentiment': social_sentiment.astype(int),
        'review_trend': review_trend.astype(int),
        'warranty_claims': warranty_claims.astype(int),
        'search_trends': search_trends.astype(int),
        'retail_alerts': retail_alerts.astype(int)
    })


def generate_business_metrics(volume_df):
    """
    Generate operational metrics that correlate with volume.
    """
    np.random.seed(44)
    weeks = len(volume_df)
    
    # Normalize volume for calculations
    vol_norm = (volume_df['y'] - volume_df['y'].mean()) / volume_df['y'].std()
    
    # Service Level: drops when volume is high
    base_sl = 0.82
    service_level = base_sl - vol_norm * 0.04 + np.random.normal(0, 0.02, weeks)
    service_level = np.clip(service_level, 0.68, 0.95)
    
    # Average Handle Time
    base_aht = 4.5
    aht = base_aht + vol_norm * 0.3 + np.random.normal(0, 0.2, weeks)
    aht = np.clip(aht, 3.5, 6.5)
    
    # First Call Resolution
    base_fcr = 0.76
    fcr = base_fcr - vol_norm * 0.02 + np.random.normal(0, 0.03, weeks)
    fcr = np.clip(fcr, 0.65, 0.88)
    
    # Customer Satisfaction
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
    
    # Generate Axiom Ray AI signals
    print("2. Generating Axiom Ray AI early warning signals...")
    axiom_df = generate_axiom_ray_signals(volume_df)
    
    # Generate business metrics
    print("3. Generating operational metrics...")
    metrics_df = generate_business_metrics(volume_df)
    
    # Combine all data
    combined_df = volume_df.merge(axiom_df, on='ds').merge(metrics_df, on='ds')
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save files
    volume_df[['ds', 'y']].to_csv('data/historical_volume.csv', index=False)
    metrics_df.to_csv('data/business_metrics.csv', index=False)
    combined_df.to_csv('data/combined_data.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Period: {volume_df['ds'].min().strftime('%Y-%m-%d')} to {volume_df['ds'].max().strftime('%Y-%m-%d')}")
    print(f"Total weeks: {len(volume_df)}")
    print(f"\nVolume Statistics:")
    print(f"  Mean: {volume_df['y'].mean():.0f} calls/week")
    print(f"  Std:  {volume_df['y'].std():.0f} calls/week")
    print(f"  CV:   {volume_df['y'].std()/volume_df['y'].mean()*100:.1f}%")
    print(f"  Min:  {volume_df['y'].min():.0f}")
    print(f"  Max:  {volume_df['y'].max():.0f}")
    print(f"\nTrend (growth over 3 years):")
    first_quarter = volume_df.head(13)['y'].mean()
    last_quarter = volume_df.tail(13)['y'].mean()
    print(f"  Q1 2021 avg: {first_quarter:.0f}")
    print(f"  Q4 2023 avg: {last_quarter:.0f}")
    print(f"  Growth: {((last_quarter/first_quarter - 1) * 100):.1f}%")
    print(f"\nSeasonality:")
    monthly = volume_df.copy()
    monthly['month'] = monthly['ds'].dt.month
    dec_avg = monthly[monthly['month'] == 12]['y'].mean()
    jul_avg = monthly[monthly['month'] == 7]['y'].mean()
    print(f"  December avg: {dec_avg:.0f}")
    print(f"  July avg: {jul_avg:.0f}")
    print(f"  Seasonal swing: {dec_avg - jul_avg:.0f} ({((dec_avg/jul_avg - 1) * 100):.0f}%)")
    print(f"\nProduct Split:")
    print(f"  Shark: {volume_df['shark_volume'].sum():,} ({volume_df['shark_volume'].sum()/volume_df['y'].sum()*100:.1f}%)")
    print(f"  Ninja: {volume_df['ninja_volume'].sum():,} ({volume_df['ninja_volume'].sum()/volume_df['y'].sum()*100:.1f}%)")
    print("\n" + "=" * 60)
    print("Files saved to data/ directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
