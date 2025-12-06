"""
Business metrics calculations for call center forecasting.
Converts forecasts into actionable business insights.
"""

import pandas as pd
import numpy as np


def calculate_staffing_needs(forecast_df, avg_handle_time_minutes=4.5, service_level_target=0.80):
    """
    Calculate staffing needs based on forecast using Erlang C approximation.
    
    Uses realistic call center staffing formulas accounting for:
    - Non-uniform call arrival patterns
    - Queueing theory (Erlang C)
    - Service level requirements
    - Agent occupancy rates
    
    Args:
        forecast_df: Forecast DataFrame with 'yhat' (predicted volume)
        avg_handle_time_minutes: Average handle time per call in minutes
        service_level_target: Target service level (e.g., 0.80 for 80%)
    
    Returns:
        DataFrame with staffing calculations
    """
    # Convert weekly calls to staffing needs
    # Assume 5-day work week, 8 hours/day = 40 hours/week
    hours_per_week = 40
    
    # Calculate total productive hours needed (time actually handling calls)
    productive_hours = (forecast_df['yhat'] * avg_handle_time_minutes) / 60
    
    # Account for call arrival patterns (not uniform)
    # Calls cluster during business hours - busy hours need more agents
    # Typical distribution: 60% of calls in 40% of hours (busy period)
    busy_period_factor = 1.5  # Busy periods have 1.5x average volume
    
    # Effective hours needed accounting for clustering
    effective_hours = productive_hours * busy_period_factor
    
    # Target occupancy rate (agents not 100% busy due to service level needs)
    # For 80% service level: target 75-85% occupancy is realistic
    # Lower occupancy = better service level (agents available for spikes)
    base_occupancy = 0.75 + (service_level_target * 0.15)  # 75-90% range
    target_occupancy = np.clip(base_occupancy, 0.70, 0.90)
    
    # Base FTE calculation accounting for occupancy
    base_ftes = effective_hours / (hours_per_week * target_occupancy)
    
    # Add service level safety buffer
    # Erlang C theory: need extra agents for queue management
    # 80% service level typically needs 15-20% buffer over base
    sl_buffer_multiplier = 1.0 + ((1.0 - service_level_target) * 0.25)
    fte_with_buffer = base_ftes * sl_buffer_multiplier
    
    # Minimum staffing requirement (always need coverage)
    # For professional call center: minimum 2 agents for redundancy
    min_agents = 2
    
    # Final FTE needed (use max of calculated or minimum)
    fte_needed = np.maximum(fte_with_buffer, min_agents)
    
    # Round up to whole agents (can't have fractional people)
    agents_needed = np.ceil(fte_needed).astype(int)
    
    # Calculate weekly hours worked
    weekly_hours = (agents_needed * hours_per_week).astype(float)
    
    # For display: also show productive hours (time actually handling calls)
    productive_hours = (forecast_df['yhat'] * avg_handle_time_minutes) / 60
    
    result = forecast_df[['ds', 'yhat']].copy()
    result['weekly_hours'] = weekly_hours.round(1)
    result['productive_hours'] = productive_hours.round(1)
    result['agents_needed'] = agents_needed
    result['fte_needed'] = fte_needed.round(1)
    result['occupancy_pct'] = ((productive_hours / weekly_hours) * 100).round(1)
    
    return result


def calculate_costs(staffing_df, hourly_rate=25.0, overhead_rate=1.35):
    """
    Calculate staffing costs with realistic overhead.
    
    Overhead typically includes:
    - Benefits (health, dental, 401k): ~25-35% of base
    - Facilities (space, equipment): ~10-15%
    - Training & management: ~5-10%
    Total overhead: ~40-50%, so 1.35-1.50 multiplier is realistic
    
    Args:
        staffing_df: DataFrame from calculate_staffing_needs
        hourly_rate: Base hourly rate per agent (before overhead)
        overhead_rate: Overhead multiplier (benefits, facilities, etc.)
    
    Returns:
        DataFrame with cost calculations
    """
    result = staffing_df.copy()
    
    # Weekly hours worked per agent (typically 40 hours)
    hours_per_week = 40
    
    # Weekly cost = agents * hours_per_week * hourly_rate * overhead
    # This gives fully-loaded cost (base salary + benefits + facilities)
    result['weekly_cost'] = (result['agents_needed'] * hours_per_week * hourly_rate * overhead_rate).round(2)
    
    # Monthly cost (average 4.33 weeks per month)
    result['monthly_cost'] = (result['weekly_cost'] * 4.33).round(2)
    
    # Annualized cost (for reference)
    result['annual_cost'] = (result['weekly_cost'] * 52).round(2)
    
    # Cost per call (efficiency metric)
    # Avoid division by zero
    result['cost_per_call'] = (result['weekly_cost'] / result['yhat'].replace(0, 1)).round(2)
    
    return result


def calculate_roi(forecast_df, historical_df, avg_handle_time=4.5, hourly_rate=25.0, overhead_rate=1.35):
    """
    Calculate ROI of using forecasting vs. reactive staffing.
    
    Compares:
    - Forecast-based staffing: Staff for predicted volume each week
    - Reactive staffing: Always staff for historical peak (overstaffing)
    
    Args:
        forecast_df: Future forecast
        historical_df: Historical data for comparison
        avg_handle_time: Average handle time in minutes
        hourly_rate: Hourly rate per agent
        overhead_rate: Overhead multiplier
    
    Returns:
        Dictionary with ROI metrics
    """
    # Calculate forecasted costs with optimal staffing
    staffing_forecast = calculate_staffing_needs(forecast_df, avg_handle_time)
    costs_forecast = calculate_costs(staffing_forecast, hourly_rate, overhead_rate)
    
    # Calculate what costs would be with reactive staffing (always staff for peak)
    # This represents the "old way" - staff for worst case every week
    peak_volume = historical_df['y'].max()
    
    # Create DataFrame with same structure as forecast_df, but with peak volume
    reactive_forecast = pd.DataFrame({
        'ds': forecast_df['ds'].copy(),
        'yhat': [peak_volume] * len(forecast_df)
    })
    reactive_staffing = calculate_staffing_needs(
        reactive_forecast,
        avg_handle_time
    )
    reactive_costs = calculate_costs(reactive_staffing, hourly_rate, overhead_rate)
    
    # Calculate savings from forecasting
    total_forecast_cost = costs_forecast['weekly_cost'].sum()
    total_reactive_cost = reactive_costs['weekly_cost'].sum()
    savings = total_reactive_cost - total_forecast_cost
    savings_pct = (savings / total_reactive_cost * 100) if total_reactive_cost > 0 else 0
    
    # Calculate average metrics
    avg_agents_forecast = staffing_forecast['agents_needed'].mean()
    avg_agents_reactive = reactive_staffing['agents_needed'].mean()
    avg_weekly_cost_forecast = costs_forecast['weekly_cost'].mean()
    avg_weekly_cost_reactive = reactive_costs['weekly_cost'].mean()
    
    # Calculate annualized savings (extrapolate to full year)
    weeks_per_year = 52
    annual_savings = (savings / len(forecast_df)) * weeks_per_year if len(forecast_df) > 0 else 0
    
    return {
        'forecast_period_weeks': len(forecast_df),
        'total_forecast_cost': total_forecast_cost,
        'total_reactive_cost': total_reactive_cost,
        'total_savings': savings,
        'savings_percentage': savings_pct,
        'annualized_savings': annual_savings,
        'avg_agents_forecast': avg_agents_forecast,
        'avg_agents_reactive': avg_agents_reactive,
        'agents_saved': avg_agents_reactive - avg_agents_forecast,
        'avg_weekly_cost_forecast': avg_weekly_cost_forecast,
        'avg_weekly_cost_reactive': avg_weekly_cost_reactive,
        'avg_weekly_savings': avg_weekly_cost_reactive - avg_weekly_cost_forecast
    }


def identify_risk_periods(forecast_df, threshold_percentile=75):
    """
    Identify high-risk periods that need attention.
    
    Args:
        forecast_df: Forecast DataFrame
        threshold_percentile: Percentile above which is considered high-risk
    
    Returns:
        DataFrame with risk flags
    """
    threshold = forecast_df['yhat'].quantile(threshold_percentile / 100)
    max_val = forecast_df['yhat'].max()
    
    result = forecast_df.copy()
    result['is_high_risk'] = result['yhat'] >= threshold
    
    # Ensure bins are unique by adding small epsilon if needed
    low_bound = threshold * 0.8
    mid_bound = threshold
    high_bound = max_val + 1  # Add 1 to ensure it's always higher than max
    
    # Make sure bins are strictly increasing
    if mid_bound <= low_bound:
        mid_bound = low_bound + 1
    if high_bound <= mid_bound:
        high_bound = mid_bound + 1
    
    try:
        result['risk_level'] = pd.cut(
            result['yhat'],
            bins=[0, low_bound, mid_bound, high_bound],
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )
    except ValueError:
        # Fallback: simple classification
        result['risk_level'] = result['yhat'].apply(
            lambda x: 'High' if x >= threshold else ('Medium' if x >= threshold * 0.8 else 'Low')
        )
    
    return result

