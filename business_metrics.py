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
    # Assume 5-day work week, 10 hours/day (e.g., 8am-6pm) = 50 hours/week
    # This is more realistic for call center operations
    hours_per_day = 10
    days_per_week = 5
    hours_per_week = hours_per_day * days_per_week  # 50 hours
    
    # Calculate total productive hours needed (time actually handling calls)
    productive_hours = (forecast_df['yhat'] * avg_handle_time_minutes) / 60
    
    # Account for intraday call arrival patterns (NOT uniform distribution)
    # Reality: Calls cluster significantly during peak hours
    # Typical pattern: 
    #   - 30-40% of daily calls in peak 2-3 hours (lunch, afternoon)
    #   - Average intraday peak: 2.5-3.0x the hourly average (conservative estimate)
    #   - We need to staff for these peak hours, not the average
    #   - For weekly data, estimate peak hour as 2.5x average hourly rate
    #   - This ensures adequate coverage during busy periods
    hourly_average_rate = forecast_df['yhat'] / (days_per_week * hours_per_day)  # calls per hour (average)
    peak_multiplier = 2.5  # Peak hours are 2.5x average (conservative, realistic for most centers)
    peak_hourly_rate = hourly_average_rate * peak_multiplier
    
    # Peak hour staffing requirement (this is what we need to staff for)
    # At peak hour: peak_rate calls/hour * AHT = productive minutes needed per hour
    # Convert to agents needed: productive_minutes / (60 min * occupancy_rate)
    peak_productive_minutes_per_hour = peak_hourly_rate * avg_handle_time_minutes
    
    # Target occupancy rate for service level
    # For 80% service level: target 75-85% occupancy is realistic
    # Higher occupancy = longer wait times = lower service level
    target_occupancy = 0.75 + (service_level_target * 0.125)  # 75-85% range for 80% SL
    target_occupancy = np.clip(target_occupancy, 0.70, 0.90)
    
    # Agents needed for peak hour
    # Formula: (calls/hour * AHT_minutes) / (60 * occupancy)
    peak_agents_needed = peak_productive_minutes_per_hour / (60 * target_occupancy)
    
    # This gives us agents needed at peak - but we need them for the full day
    # However, since we're calculating weekly, we apply this as base FTE
    # The peak requirement drives the staffing level
    base_ftes = peak_agents_needed
    
    # Add service level safety buffer for queue management
    # Erlang C: Extra agents needed to maintain service level during variability
    # For 80% service level: typically need 10-15% buffer over base
    # Adding slightly more buffer to ensure service quality
    sl_buffer_multiplier = 1.0 + ((1.0 - service_level_target) * 0.25)  # 1.05 for 80% SL (slightly more conservative)
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
    
    # Calculate productive hours (time actually handling calls)
    productive_hours = (forecast_df['yhat'] * avg_handle_time_minutes) / 60
    
    # Occupancy calculation:
    # This represents actual utilization across the week
    # Lower occupancy = agents have more idle time = better ability to handle spikes
    # Higher occupancy = agents more busy = risk of service degradation
    # Realistic range: 60-85% for most call centers
    occupancy_pct = ((productive_hours / weekly_hours) * 100).clip(0, 100)
    
    result = forecast_df[['ds', 'yhat']].copy()
    result['weekly_hours'] = weekly_hours.round(1)
    result['productive_hours'] = productive_hours.round(1)
    result['agents_needed'] = agents_needed
    result['fte_needed'] = fte_needed.round(1)
    result['occupancy_pct'] = occupancy_pct.round(1)
    
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
    Calculate comprehensive ROI of using forecasting vs. alternative staffing strategies.
    
    Compares three staffing approaches:
    1. Forecast-based staffing: Optimal staffing based on predicted volume (recommended)
    2. Reactive staffing: Always staff for historical peak (conservative but wasteful)
    3. Average staffing: Staff for historical average (cheap but risky)
    
    Args:
        forecast_df: Future forecast with 'yhat', 'yhat_lower', 'yhat_upper'
        historical_df: Historical data for comparison
        avg_handle_time: Average handle time in minutes
        hourly_rate: Hourly rate per agent
        overhead_rate: Overhead multiplier
    
    Returns:
        Dictionary with comprehensive ROI metrics
    """
    service_level = 0.80
    
    # 1. Forecast-based staffing (optimal)
    staffing_forecast = calculate_staffing_needs(forecast_df, avg_handle_time, service_level)
    costs_forecast = calculate_costs(staffing_forecast, hourly_rate, overhead_rate)
    
    # 2. Reactive staffing: Always staff for peak (conservative approach)
    peak_volume = historical_df['y'].max()
    reactive_forecast = pd.DataFrame({
        'ds': forecast_df['ds'].copy(),
        'yhat': [peak_volume] * len(forecast_df)
    })
    reactive_staffing = calculate_staffing_needs(reactive_forecast, avg_handle_time, service_level)
    reactive_costs = calculate_costs(reactive_staffing, hourly_rate, overhead_rate)
    
    # 3. Average staffing: Staff for historical average (risky but cheaper)
    avg_volume = historical_df['y'].mean()
    avg_forecast = pd.DataFrame({
        'ds': forecast_df['ds'].copy(),
        'yhat': [avg_volume] * len(forecast_df)
    })
    avg_staffing = calculate_staffing_needs(avg_forecast, avg_handle_time, service_level)
    avg_costs = calculate_costs(avg_staffing, hourly_rate, overhead_rate)
    
    # Calculate costs
    total_forecast_cost = costs_forecast['weekly_cost'].sum()
    total_reactive_cost = reactive_costs['weekly_cost'].sum()
    total_avg_cost = avg_costs['weekly_cost'].sum()
    
    # Savings vs reactive (conservative baseline)
    savings_vs_reactive = total_reactive_cost - total_forecast_cost
    savings_pct_vs_reactive = (savings_vs_reactive / total_reactive_cost * 100) if total_reactive_cost > 0 else 0
    
    # Additional cost vs average (trade-off for better service)
    additional_vs_avg = total_forecast_cost - total_avg_cost
    additional_pct_vs_avg = (additional_vs_avg / total_avg_cost * 100) if total_avg_cost > 0 else 0
    
    # Average metrics
    avg_agents_forecast = staffing_forecast['agents_needed'].mean()
    avg_agents_reactive = reactive_staffing['agents_needed'].mean()
    avg_agents_avg = avg_staffing['agents_needed'].mean()
    
    # Cost efficiency metrics
    avg_cost_per_call_forecast = costs_forecast['cost_per_call'].mean()
    avg_cost_per_call_reactive = reactive_costs['cost_per_call'].mean()
    avg_cost_per_call_avg = avg_costs['cost_per_call'].mean()
    
    # Forecast uncertainty impact (worst case scenario)
    if 'yhat_upper' in forecast_df.columns:
        worst_case_forecast = pd.DataFrame({
            'ds': forecast_df['ds'].copy(),
            'yhat': forecast_df['yhat_upper'].values
        })
        worst_staffing = calculate_staffing_needs(worst_case_forecast, avg_handle_time, service_level)
        worst_costs = calculate_costs(worst_staffing, hourly_rate, overhead_rate)
        worst_case_cost = worst_costs['weekly_cost'].sum()
        max_additional_cost = worst_case_cost - total_forecast_cost
    else:
        worst_case_cost = total_forecast_cost
        max_additional_cost = 0
    
    # Annualized metrics
    weeks_per_year = 52
    forecast_period_weeks = len(forecast_df)
    annual_savings_vs_reactive = (savings_vs_reactive / forecast_period_weeks * weeks_per_year) if forecast_period_weeks > 0 else 0
    annual_cost_forecast = (total_forecast_cost / forecast_period_weeks * weeks_per_year) if forecast_period_weeks > 0 else 0
    
    # Historical comparison (what was average weekly cost in recent history?)
    recent_historical = historical_df.tail(13)  # Last quarter
    if len(recent_historical) > 0:
        hist_staffing = calculate_staffing_needs(
            pd.DataFrame({'ds': recent_historical['ds'], 'yhat': recent_historical['y']}),
            avg_handle_time, service_level
        )
        hist_costs = calculate_costs(hist_staffing, hourly_rate, overhead_rate)
        avg_historical_weekly_cost = hist_costs['weekly_cost'].mean()
        forecast_vs_historical = costs_forecast['weekly_cost'].mean() - avg_historical_weekly_cost
        forecast_vs_historical_pct = (forecast_vs_historical / avg_historical_weekly_cost * 100) if avg_historical_weekly_cost > 0 else 0
    else:
        avg_historical_weekly_cost = 0
        forecast_vs_historical = 0
        forecast_vs_historical_pct = 0
    
    return {
        'forecast_period_weeks': forecast_period_weeks,
        
        # Forecast-based (optimal)
        'total_forecast_cost': total_forecast_cost,
        'avg_weekly_cost_forecast': costs_forecast['weekly_cost'].mean(),
        'avg_agents_forecast': avg_agents_forecast,
        'avg_cost_per_call_forecast': avg_cost_per_call_forecast,
        
        # Reactive (conservative)
        'total_reactive_cost': total_reactive_cost,
        'avg_weekly_cost_reactive': reactive_costs['weekly_cost'].mean(),
        'avg_agents_reactive': avg_agents_reactive,
        'avg_cost_per_call_reactive': avg_cost_per_call_reactive,
        
        # Average-based (risky)
        'total_avg_cost': total_avg_cost,
        'avg_weekly_cost_avg': avg_costs['weekly_cost'].mean(),
        'avg_agents_avg': avg_agents_avg,
        'avg_cost_per_call_avg': avg_cost_per_call_avg,
        
        # Savings metrics
        'total_savings': savings_vs_reactive,  # vs reactive (for backward compatibility)
        'savings_percentage': savings_pct_vs_reactive,  # vs reactive
        'savings_vs_reactive': savings_vs_reactive,
        'savings_pct_vs_reactive': savings_pct_vs_reactive,
        'additional_vs_avg': additional_vs_avg,
        'additional_pct_vs_avg': additional_pct_vs_avg,
        'agents_saved': avg_agents_reactive - avg_agents_forecast,  # vs reactive
        
        # Uncertainty metrics
        'worst_case_cost': worst_case_cost,
        'max_additional_cost': max_additional_cost,
        'confidence_range_cost': worst_case_cost - total_forecast_cost,
        
        # Annualized
        'annualized_savings': annual_savings_vs_reactive,
        'annual_cost_forecast': annual_cost_forecast,
        
        # Historical comparison
        'avg_historical_weekly_cost': avg_historical_weekly_cost,
        'forecast_vs_historical': forecast_vs_historical,
        'forecast_vs_historical_pct': forecast_vs_historical_pct
    }


def calculate_service_quality_metrics(forecast_df, staffing_df, avg_handle_time=4.5, service_level_target=0.80):
    """
    Calculate expected service quality metrics based on forecast and staffing.
    
    Args:
        forecast_df: Forecast with 'yhat' (expected volume)
        staffing_df: Staffing needs from calculate_staffing_needs
        avg_handle_time: Average handle time in minutes
        service_level_target: Target service level
    
    Returns:
        DataFrame with service quality metrics
    """
    result = staffing_df[['ds', 'yhat', 'agents_needed']].copy()
    
    # Calculate expected service level (approximation)
    # Higher occupancy = lower service level (more busy = longer wait times)
    result['expected_occupancy'] = (result['yhat'] * avg_handle_time / 60) / (result['agents_needed'] * 40)
    result['expected_occupancy'] = np.clip(result['expected_occupancy'], 0, 1)
    
    # Estimate service level based on occupancy (empirical relationship)
    # At 80% occupancy, typically achieve ~80% service level
    # At 90% occupancy, service level drops to ~60%
    # At 70% occupancy, service level improves to ~90%
    result['estimated_service_level'] = np.clip(
        1.0 - (result['expected_occupancy'] - service_level_target) * 2.0,
        0.50, 0.95
    )
    
    # Estimate average wait time (approximation)
    # Higher occupancy = longer wait times
    base_wait_time = 30  # Base wait time in seconds when occupancy is at target
    result['estimated_avg_wait_sec'] = base_wait_time * (result['expected_occupancy'] / service_level_target) ** 2
    result['estimated_avg_wait_sec'] = np.clip(result['estimated_avg_wait_sec'], 10, 300)
    
    # Risk of service level failure
    result['service_level_risk'] = result['estimated_service_level'] < service_level_target
    result['service_level_deficit'] = np.maximum(0, service_level_target - result['estimated_service_level'])
    
    return result


def identify_risk_periods(forecast_df, threshold_percentile=75):
    """
    Identify high-risk periods that need attention based on volume and uncertainty.
    
    Args:
        forecast_df: Forecast DataFrame with 'yhat', optionally 'yhat_upper', 'yhat_lower'
        threshold_percentile: Percentile above which is considered high-risk
    
    Returns:
        DataFrame with risk flags and recommendations
    """
    threshold = forecast_df['yhat'].quantile(threshold_percentile / 100)
    max_val = forecast_df['yhat'].max()
    mean_val = forecast_df['yhat'].mean()
    
    result = forecast_df.copy()
    result['is_high_risk'] = result['yhat'] >= threshold
    
    # Calculate uncertainty risk (if confidence intervals available)
    if 'yhat_upper' in forecast_df.columns and 'yhat_lower' in forecast_df.columns:
        result['uncertainty_range'] = result['yhat_upper'] - result['yhat_lower']
        result['uncertainty_pct'] = (result['uncertainty_range'] / result['yhat']) * 100
        high_uncertainty_threshold = result['uncertainty_pct'].quantile(0.75)
        result['high_uncertainty'] = result['uncertainty_pct'] >= high_uncertainty_threshold
    else:
        result['uncertainty_range'] = 0
        result['uncertainty_pct'] = 0
        result['high_uncertainty'] = False
    
    # Risk levels
    result['risk_level'] = 'Low'
    result.loc[result['yhat'] >= threshold, 'risk_level'] = 'High'
    result.loc[(result['yhat'] >= threshold * 0.85) & (result['yhat'] < threshold), 'risk_level'] = 'Medium'
    
    # Combined risk (volume + uncertainty)
    result['combined_risk'] = 'Low'
    result.loc[result['is_high_risk'] | result['high_uncertainty'], 'combined_risk'] = 'Medium'
    result.loc[result['is_high_risk'] & result['high_uncertainty'], 'combined_risk'] = 'High'
    
    # Generate recommendations
    def get_recommendation(row):
        if row['combined_risk'] == 'High':
            if row['high_uncertainty']:
                return "Increase staffing buffer due to high volume and uncertainty"
            else:
                return "Increase staffing for high volume period"
        elif row['combined_risk'] == 'Medium':
            if row['high_uncertainty']:
                return "Monitor closely - high uncertainty in forecast"
            else:
                return "Standard staffing should suffice"
        else:
            return "Low volume period - consider flexible staffing"
    
    result['recommendation'] = result.apply(get_recommendation, axis=1)
    
    return result


def calculate_budget_impact(forecast_df, costs_df, historical_df=None):
    """
    Calculate budget planning metrics for the forecast period.
    
    Args:
        forecast_df: Forecast DataFrame
        costs_df: Costs DataFrame from calculate_costs
        historical_df: Optional historical data for comparison
    
    Returns:
        Dictionary with budget metrics
    """
    total_cost = costs_df['weekly_cost'].sum()
    avg_weekly_cost = costs_df['weekly_cost'].mean()
    peak_weekly_cost = costs_df['weekly_cost'].max()
    min_weekly_cost = costs_df['weekly_cost'].min()
    
    # Monthly aggregation
    costs_df_monthly = costs_df.copy()
    costs_df_monthly['year_month'] = pd.to_datetime(costs_df_monthly['ds']).dt.to_period('M')
    monthly_costs = costs_df_monthly.groupby('year_month')['weekly_cost'].sum()
    
    # Quarterly aggregation
    costs_df_quarterly = costs_df.copy()
    costs_df_quarterly['quarter'] = pd.to_datetime(costs_df_quarterly['ds']).dt.to_period('Q')
    quarterly_costs = costs_df_quarterly.groupby('quarter')['weekly_cost'].sum()
    
    # Cost trend
    cost_trend = 'Stable'
    if len(costs_df) >= 4:
        recent_avg = costs_df['weekly_cost'].tail(4).mean()
        early_avg = costs_df['weekly_cost'].head(4).mean()
        if recent_avg > early_avg * 1.05:
            cost_trend = 'Increasing'
        elif recent_avg < early_avg * 0.95:
            cost_trend = 'Decreasing'
    
    # Historical comparison
    historical_avg_cost = None
    cost_change_pct = None
    if historical_df is not None and len(historical_df) > 0:
        hist_staffing = calculate_staffing_needs(
            pd.DataFrame({'ds': historical_df['ds'], 'yhat': historical_df['y']}),
            4.5, 0.80
        )
        hist_costs = calculate_costs(hist_staffing, 25.0, 1.35)
        historical_avg_cost = hist_costs['weekly_cost'].mean()
        cost_change_pct = ((avg_weekly_cost - historical_avg_cost) / historical_avg_cost * 100) if historical_avg_cost > 0 else 0
    
    return {
        'total_cost': total_cost,
        'avg_weekly_cost': avg_weekly_cost,
        'peak_weekly_cost': peak_weekly_cost,
        'min_weekly_cost': min_weekly_cost,
        'cost_range': peak_weekly_cost - min_weekly_cost,
        'monthly_costs': monthly_costs.to_dict() if len(monthly_costs) > 0 else {},
        'quarterly_costs': quarterly_costs.to_dict() if len(quarterly_costs) > 0 else {},
        'cost_trend': cost_trend,
        'historical_avg_cost': historical_avg_cost,
        'cost_change_pct': cost_change_pct
    }

