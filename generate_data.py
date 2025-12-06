"""
Generate synthetic data for SharkNinja Customer Support Volume Forecasting.

Key features:
- Clear upward TREND (growing customer base)
- Strong SEASONALITY (holiday patterns, product cycles)
- Axiom Ray as a LEADING indicator (2 weeks ahead)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_call_volume_data(start_date='2021-01-04', weeks=156):
    """
    Generate realistic call center volume data with clear trend and seasonality.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=weeks, freq='W-MON')
    t = np.arange(weeks)
    
    # ========== STRONG TREND ==========
    # Clear upward trend - company growing ~15% per year
    base_level = 400
    annual_growth = 0.15  # 15% annual growth
    weekly_growth = (1 + annual_growth) ** (1/52) - 1
    trend = base_level * ((1 + weekly_growth) ** t)
    
    # ========== STRONG SEASONALITY ==========
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Primary annual cycle - peaks in winter (holidays), lowest in summer
    annual_season = 120 * np.cos(2 * np.pi * (day_of_year - 355) / 365.25)  # Peak near New Year
    
    # Secondary cycle - spring cleaning bump
    spring_bump = 40 * np.exp(-((day_of_year - 90) ** 2) / (2 * 30 ** 2))  # March-April
    
    # Quarterly pattern (product launches, Q4 retail)
    quarterly = 30 * np.cos(4 * np.pi * day_of_year / 365.25)
    
    seasonality = annual_season + spring_bump + quarterly
    
    # ========== MONTHLY EFFECTS ==========
    months = np.array([d.month for d in dates])
    monthly_effect = np.zeros(weeks)
    monthly_effect[months == 1] = 80   # Post-holiday returns
    monthly_effect[months == 12] = 60  # Holiday rush
    monthly_effect[months == 11] = 50  # Black Friday / Cyber Monday
    monthly_effect[months == 7] = -30  # Summer lull
    monthly_effect[months == 8] = -20  # Late summer
    
    # ========== SPECIAL EVENTS (Product Launches) ==========
    events = np.zeros(weeks)
    
    # Major product launches - consistent timing each year
    for year_offset in [0, 52, 104]:
        # Spring launch (week 12-13 each year)
        launch_week = year_offset + 12
        if launch_week < weeks:
            events[launch_week] = 150
            if launch_week + 1 < weeks:
                events[launch_week + 1] = 80
            if launch_week + 2 < weeks:
                events[launch_week + 2] = 40
        
        # Fall launch (week 38-39)
        launch_week = year_offset + 38
        if launch_week < weeks:
            events[launch_week] = 130
            if launch_week + 1 < weeks:
                events[launch_week + 1] = 70
    
    # Black Friday spikes
    for year_start in [0, 52, 104]:
        bf_week = year_start + 47
        if bf_week < weeks:
            events[bf_week] = 100
            if bf_week + 1 < weeks:
                events[bf_week + 1] = 60
    
    # ========== NOISE (Realistic but not overwhelming) ==========
    # Moderate white noise
    white_noise = np.random.normal(0, 40, weeks)
    
    # Autocorrelated component (week-to-week persistence)
    ar_noise = np.zeros(weeks)
    ar_noise[0] = np.random.normal(0, 20)
    for i in range(1, weeks):
        ar_noise[i] = 0.4 * ar_noise[i-1] + np.random.normal(0, 20)
    
    total_noise = white_noise + ar_noise
    
    # ========== COMBINE ALL COMPONENTS ==========
    volume = trend + seasonality + monthly_effect + events + total_noise
    volume = np.maximum(volume, 200)  # Minimum floor
    volume = np.round(volume).astype(int)
    
    # ========== PRODUCT BREAKDOWN ==========
    shark_pct = np.zeros(weeks)
    for i, month in enumerate(months):
        if month in [3, 4]:  # Spring cleaning - vacuums
            shark_pct[i] = 0.62
        elif month in [5, 6, 7]:  # Summer - kitchen (Ninja)
            shark_pct[i] = 0.42
        elif month in [11, 12]:  # Holiday - balanced
            shark_pct[i] = 0.50
        else:
            shark_pct[i] = 0.52
    
    shark_pct = shark_pct + np.random.normal(0, 0.03, weeks)
    shark_pct = np.clip(shark_pct, 0.35, 0.65)
    
    df = pd.DataFrame({
        'ds': dates,
        'y': volume,
        'shark_volume': (volume * shark_pct).astype(int),
        'ninja_volume': (volume * (1 - shark_pct)).astype(int)
    })
    
    # Subcategories
    df['shark_vacuums'] = (df['shark_volume'] * 0.70).astype(int)
    df['shark_hair_care'] = (df['shark_volume'] * 0.18).astype(int)
    df['shark_air_purifiers'] = (df['shark_volume'] * 0.12).astype(int)
    
    df['ninja_blenders'] = (df['ninja_volume'] * 0.25).astype(int)
    df['ninja_air_fryers'] = (df['ninja_volume'] * 0.35).astype(int)
    df['ninja_coffee'] = (df['ninja_volume'] * 0.25).astype(int)
    df['ninja_grills'] = (df['ninja_volume'] * 0.15).astype(int)
    
    return df


def generate_axiom_ray_score(df, lead_weeks=2):
    """
    Generate Axiom Ray AI score as a LEADING indicator.
    
    The score predicts volume 2 weeks ahead by detecting:
    - Social media complaints trending
    - Review sentiment changes
    - Warranty claim patterns
    - Search trend anomalies
    
    Key: axiom_score[t] correlates with volume[t + lead_weeks]
    """
    np.random.seed(123)
    
    n = len(df)
    volume = np.array(df['y'])
    
    # Create base signal that leads volume
    # We want axiom[t] to predict volume[t + lead_weeks]
    # So axiom[t] should be based on volume[t + lead_weeks]
    
    # Shift volume backward to create leading relationship
    future_volume = np.zeros(n)
    future_volume[:-lead_weeks] = volume[lead_weeks:]
    future_volume[-lead_weeks:] = volume[-lead_weeks:]  # Extrapolate last values
    
    # Normalize future volume to 0-100 scale
    vol_min, vol_max = np.percentile(future_volume, [5, 95])
    axiom_base = 30 + 60 * (future_volume - vol_min) / (vol_max - vol_min)
    axiom_base = np.clip(axiom_base, 20, 95)
    
    # Add realistic noise (not perfectly correlated)
    signal_noise = np.random.normal(0, 8, n)
    
    # Add some autocorrelation to the noise (smooth transitions)
    smooth_noise = np.zeros(n)
    smooth_noise[0] = signal_noise[0]
    for i in range(1, n):
        smooth_noise[i] = 0.6 * smooth_noise[i-1] + 0.4 * signal_noise[i]
    
    axiom_score = axiom_base + smooth_noise
    
    # Occasional false positives/negatives (real-world imperfection)
    anomaly_weeks = np.random.choice(n, size=int(n * 0.08), replace=False)
    for week in anomaly_weeks:
        axiom_score[week] += np.random.uniform(-15, 15)
    
    axiom_score = np.clip(axiom_score, 15, 98)
    axiom_score = np.round(axiom_score, 1)
    
    return axiom_score


def generate_signal_components(df):
    """Generate individual signal components that make up Axiom Ray score."""
    np.random.seed(456)
    
    n = len(df)
    volume = np.array(df['y'])
    axiom = np.array(df['axiom_ray_score'])
    
    # Social sentiment (most reactive, 1 week lead)
    social = 0.3 * axiom + np.random.normal(50, 10, n)
    social = np.clip(social, 10, 100).round(1)
    
    # Review trend (2 week lead, matches axiom closely)
    review = 0.4 * axiom + np.random.normal(30, 8, n)
    review = np.clip(review, 10, 100).round(1)
    
    # Warranty claims (3 week lead, lagged)
    warranty = np.roll(axiom, 1) * 0.35 + np.random.normal(40, 12, n)
    warranty = np.clip(warranty, 10, 100).round(1)
    
    # Search trends
    search = 0.25 * axiom + np.random.normal(45, 10, n)
    search = np.clip(search, 10, 100).round(1)
    
    # Retail alerts
    retail = 0.2 * axiom + np.random.normal(50, 15, n)
    retail = np.clip(retail, 10, 100).round(1)
    
    return social, review, warranty, search, retail


def create_combined_dataset():
    """Create the full dataset with all features."""
    
    # Generate base volume data
    df = generate_call_volume_data(start_date='2021-01-04', weeks=156)
    
    # Generate Axiom Ray score (leading indicator)
    df['axiom_ray_score'] = generate_axiom_ray_score(df, lead_weeks=2)
    
    # Generate signal components
    social, review, warranty, search, retail = generate_signal_components(df)
    df['social_sentiment'] = social
    df['review_trend'] = review
    df['warranty_claims'] = warranty
    df['search_trends'] = search
    df['retail_alerts'] = retail
    
    return df


def validate_leading_indicator(df, lead_weeks=2):
    """Validate that axiom_ray_score is indeed a leading indicator."""
    
    volume = np.array(df['y'])
    axiom = np.array(df['axiom_ray_score'])
    
    # Concurrent correlation
    concurrent_corr = np.corrcoef(volume, axiom)[0, 1]
    
    # Leading correlation (axiom predicts future volume)
    leading_corr = np.corrcoef(volume[lead_weeks:], axiom[:-lead_weeks])[0, 1]
    
    # Lagging correlation (for comparison)
    lagging_corr = np.corrcoef(volume[:-lead_weeks], axiom[lead_weeks:])[0, 1]
    
    print("=" * 50)
    print("AXIOM RAY LEADING INDICATOR VALIDATION")
    print("=" * 50)
    print(f"Concurrent correlation:  {concurrent_corr:.3f}")
    print(f"Leading correlation:     {leading_corr:.3f} (axiom[t] → volume[t+{lead_weeks}])")
    print(f"Lagging correlation:     {lagging_corr:.3f}")
    print()
    
    if leading_corr > concurrent_corr and leading_corr > lagging_corr:
        print("SUCCESS: Axiom Ray IS a leading indicator!")
    else:
        print("WARNING: Leading correlation should be highest")
    
    print()
    print("Data Summary:")
    print(f"  Total weeks: {len(df)}")
    print(f"  Volume range: {df['y'].min()} - {df['y'].max()}")
    print(f"  Axiom score range: {df['axiom_ray_score'].min():.1f} - {df['axiom_ray_score'].max():.1f}")
    print(f"  Volume trend: {(df['y'].iloc[-13:].mean() / df['y'].iloc[:13].mean() - 1) * 100:.1f}% growth over period")
    
    return leading_corr, concurrent_corr


if __name__ == '__main__':
    # Generate data
    df = create_combined_dataset()
    
    # Validate leading indicator
    validate_leading_indicator(df)
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/combined_data.csv', index=False)
    print(f"\nSaved {len(df)} rows to data/combined_data.csv")
    
    # Print sample
    print("\nFirst few rows:")
    print(df[['ds', 'y', 'axiom_ray_score', 'shark_volume', 'ninja_volume']].head(10))
    
    print("\nLast few rows:")
    print(df[['ds', 'y', 'axiom_ray_score', 'shark_volume', 'ninja_volume']].tail(10))
