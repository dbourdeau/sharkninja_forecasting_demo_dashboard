"""
Generate synthetic data for SharkNinja Customer Support Volume Forecasting.

Key features:
- Clear upward TREND (growing customer base)
- Strong SEASONALITY (holiday patterns, product cycles)
- Axiom Ray as a LEADING indicator (independent signal, not derived from volume)

IMPORTANT: Axiom Ray is generated from the SAME underlying factors that drive
volume (events, seasonality) but NOT from volume itself. This avoids data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_underlying_factors(weeks, dates):
    """
    Generate the underlying factors that drive BOTH volume and axiom scores.
    These are the "ground truth" signals that exist in the world.
    """
    np.random.seed(42)
    
    t = np.arange(weeks)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    months = np.array([d.month for d in dates])
    
    # ========== UNDERLYING DRIVERS ==========
    
    # 1. Market Growth (underlying trend)
    growth_factor = 1.0 + 0.003 * t  # ~15% annual growth
    
    # 2. Seasonal Pattern (holidays, weather, etc.)
    seasonal_factor = (
        np.cos(2 * np.pi * (day_of_year - 355) / 365.25) +  # Winter peak
        0.3 * np.exp(-((day_of_year - 90) ** 2) / (2 * 30 ** 2))  # Spring bump
    )
    
    # 3. Monthly Effects
    monthly_factor = np.zeros(weeks)
    monthly_factor[months == 1] = 0.15   # Post-holiday
    monthly_factor[months == 12] = 0.12  # Holiday rush
    monthly_factor[months == 11] = 0.10  # Black Friday
    monthly_factor[months == 7] = -0.08  # Summer lull
    
    # 4. Product Launch Events (these create spikes)
    event_factor = np.zeros(weeks)
    for year_offset in [0, 52, 104]:
        # Spring launch
        launch_week = year_offset + 12
        if launch_week < weeks:
            event_factor[launch_week] = 0.35
            if launch_week + 1 < weeks:
                event_factor[launch_week + 1] = 0.15
        
        # Fall launch
        launch_week = year_offset + 38
        if launch_week < weeks:
            event_factor[launch_week] = 0.30
            if launch_week + 1 < weeks:
                event_factor[launch_week + 1] = 0.12
        
        # Black Friday
        bf_week = year_offset + 47
        if bf_week < weeks:
            event_factor[bf_week] = 0.20
    
    # 5. Random quality issues (surprise spikes)
    np.random.seed(789)
    issue_weeks = np.random.choice(weeks, size=6, replace=False)
    quality_factor = np.zeros(weeks)
    for week in issue_weeks:
        quality_factor[week] = np.random.uniform(0.1, 0.25)
    
    return {
        'growth': growth_factor,
        'seasonal': seasonal_factor,
        'monthly': monthly_factor,
        'events': event_factor,
        'quality_issues': quality_factor
    }


def generate_call_volume_data(start_date='2021-01-04', weeks=156):
    """
    Generate realistic call center volume data.
    Volume is driven by underlying factors + noise.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=weeks, freq='W-MON')
    factors = generate_underlying_factors(weeks, dates)
    
    # BASE VOLUME
    base_level = 450
    
    # Combine factors to create volume
    volume = base_level * factors['growth'] * (
        1 + 
        0.25 * factors['seasonal'] +    # Seasonality has ~25% impact
        factors['monthly'] +             # Monthly effects
        factors['events'] +              # Event spikes
        factors['quality_issues']        # Quality issues
    )
    
    # Add observation noise (things we can't predict)
    np.random.seed(42)
    observation_noise = np.random.normal(0, 35, weeks)
    
    # Add autocorrelated noise (week-to-week persistence)
    ar_noise = np.zeros(weeks)
    ar_noise[0] = np.random.normal(0, 20)
    for i in range(1, weeks):
        ar_noise[i] = 0.3 * ar_noise[i-1] + np.random.normal(0, 18)
    
    volume = volume + observation_noise + ar_noise
    volume = np.maximum(volume, 200)
    volume = np.round(volume).astype(int)
    
    # Product breakdown
    months = np.array([d.month for d in dates])
    shark_pct = np.zeros(weeks)
    for i, month in enumerate(months):
        if month in [3, 4]:
            shark_pct[i] = 0.62
        elif month in [5, 6, 7]:
            shark_pct[i] = 0.42
        elif month in [11, 12]:
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
    
    return df, factors


def generate_axiom_ray_score(weeks, dates, factors, lead_weeks=2):
    """
    Generate Axiom Ray AI score as a LEADING indicator.
    
    KEY: This is generated from the SAME underlying factors that drive volume,
    but with a time LEAD and DIFFERENT noise. This simulates an AI system
    detecting patterns BEFORE they result in support calls.
    
    NO DATA LEAKAGE: We don't use volume values at all!
    """
    np.random.seed(999)  # Different seed than volume
    
    # Axiom Ray detects the underlying factors EARLY
    # It sees events/issues ~2 weeks before they hit support lines
    
    # Shift factors BACKWARD to create leading effect
    # (axiom sees events at t, volume spike happens at t+lead_weeks)
    
    def shift_forward(arr, shift):
        """Shift array forward (axiom sees it first)"""
        result = np.zeros_like(arr)
        result[:-shift] = arr[shift:]
        result[-shift:] = arr[-shift:]  # Extrapolate end
        return result
    
    # Axiom detects these factors BEFORE volume is affected
    detected_events = shift_forward(factors['events'], lead_weeks)
    detected_quality = shift_forward(factors['quality_issues'], lead_weeks)
    detected_seasonal = shift_forward(factors['seasonal'], lead_weeks)
    
    # Combine detected signals (Axiom doesn't see everything perfectly)
    base_signal = (
        50 +  # Base level
        detected_events * 80 +       # Events are strongly detected
        detected_quality * 100 +     # Quality issues are very visible
        detected_seasonal * 15 +     # Seasonal patterns somewhat visible
        factors['growth'] * 5        # Slight trend awareness
    )
    
    # Add significant noise - Axiom is helpful but not perfect
    detection_noise = np.random.normal(0, 12, weeks)
    
    # Autocorrelated noise (smooth transitions)
    smooth_noise = np.zeros(weeks)
    smooth_noise[0] = detection_noise[0]
    for i in range(1, weeks):
        smooth_noise[i] = 0.5 * smooth_noise[i-1] + 0.5 * detection_noise[i]
    
    axiom_score = base_signal + smooth_noise
    
    # False positives/negatives (Axiom makes mistakes)
    np.random.seed(888)
    false_positive_weeks = np.random.choice(weeks, size=int(weeks * 0.06), replace=False)
    for week in false_positive_weeks:
        axiom_score[week] += np.random.uniform(10, 25)  # False alarm
    
    false_negative_weeks = np.random.choice(weeks, size=int(weeks * 0.04), replace=False)
    for week in false_negative_weeks:
        axiom_score[week] -= np.random.uniform(10, 20)  # Missed signal
    
    axiom_score = np.clip(axiom_score, 15, 98)
    axiom_score = np.round(axiom_score, 1)
    
    return axiom_score


def generate_signal_components(df, factors):
    """Generate individual signal components that make up Axiom Ray score."""
    np.random.seed(456)
    
    n = len(df)
    axiom = np.array(df['axiom_ray_score'])
    
    # These are the sub-signals that Axiom aggregates
    # They correlate with axiom but have their own noise
    
    # Social sentiment (most reactive)
    social = 0.4 * axiom + 0.6 * np.random.uniform(30, 70, n)
    social = np.clip(social, 10, 100).round(1)
    
    # Review trend
    review = 0.35 * axiom + 0.65 * np.random.uniform(25, 75, n)
    review = np.clip(review, 10, 100).round(1)
    
    # Warranty claims (lagged slightly)
    warranty = 0.3 * np.roll(axiom, 1) + 0.7 * np.random.uniform(30, 70, n)
    warranty = np.clip(warranty, 10, 100).round(1)
    
    # Search trends
    search = 0.3 * axiom + 0.7 * np.random.uniform(35, 65, n)
    search = np.clip(search, 10, 100).round(1)
    
    # Retail alerts
    retail = 0.25 * axiom + 0.75 * np.random.uniform(40, 60, n)
    retail = np.clip(retail, 10, 100).round(1)
    
    return social, review, warranty, search, retail


def create_combined_dataset():
    """Create the full dataset with all features."""
    
    # Generate volume data and underlying factors
    df, factors = generate_call_volume_data(start_date='2021-01-04', weeks=156)
    
    # Generate Axiom Ray score (from factors, NOT from volume!)
    dates = df['ds']
    df['axiom_ray_score'] = generate_axiom_ray_score(len(df), dates, factors, lead_weeks=2)
    
    # Generate signal components
    social, review, warranty, search, retail = generate_signal_components(df, factors)
    df['social_sentiment'] = social
    df['review_trend'] = review
    df['warranty_claims'] = warranty
    df['search_trends'] = search
    df['retail_alerts'] = retail
    
    return df


def validate_leading_indicator(df, lead_weeks=2):
    """Validate that axiom_ray_score is a useful leading indicator without leakage."""
    
    volume = np.array(df['y'])
    axiom = np.array(df['axiom_ray_score'])
    
    # Concurrent correlation
    concurrent_corr = np.corrcoef(volume, axiom)[0, 1]
    
    # Leading correlation (axiom predicts future volume)
    leading_corr = np.corrcoef(volume[lead_weeks:], axiom[:-lead_weeks])[0, 1]
    
    # Lagging correlation (for comparison)
    lagging_corr = np.corrcoef(volume[:-lead_weeks], axiom[lead_weeks:])[0, 1]
    
    print("=" * 60)
    print("AXIOM RAY LEADING INDICATOR VALIDATION (NO DATA LEAKAGE)")
    print("=" * 60)
    print(f"Concurrent correlation:  {concurrent_corr:.3f}")
    print(f"Leading correlation:     {leading_corr:.3f} (axiom[t] -> volume[t+{lead_weeks}])")
    print(f"Lagging correlation:     {lagging_corr:.3f}")
    print()
    
    # Check for data leakage
    if leading_corr > 0.9:
        print("WARNING: Leading correlation > 0.9 suggests possible data leakage!")
    elif leading_corr > concurrent_corr:
        print("GOOD: Axiom Ray IS a leading indicator (leading > concurrent)")
    else:
        print("NOTE: Leading correlation not stronger than concurrent")
    
    print()
    print("Expected realistic values:")
    print("  - Concurrent: 0.4-0.6 (moderate same-time correlation)")
    print("  - Leading: 0.5-0.7 (useful but imperfect prediction)")
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
