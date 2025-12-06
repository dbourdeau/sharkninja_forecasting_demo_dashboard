"""
SharkNinja Customer Support Volume Forecasting Dashboard.

Interactive dashboard to visualize call center forecasts, staffing needs, and business impact.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

from forecast_model import CallVolumeForecaster, train_test_split, compare_forecasts
from business_metrics import calculate_staffing_needs, calculate_costs, calculate_roi, identify_risk_periods

# Page configuration
st.set_page_config(
    page_title="SharkNinja Support Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SharkNinja brand colors
SHARK_TEAL = "#00A0AF"
NINJA_ORANGE = "#FF6B35"
DARK_GRAY = "#2D3436"

# Custom CSS with SharkNinja branding
st.markdown(f"""
    <style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, {SHARK_TEAL}, {NINJA_ORANGE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: {DARK_GRAY};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {SHARK_TEAL};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SHARK_TEAL};
        color: white;
    }}
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load combined data from CSV. Auto-generate if missing."""
    if not os.path.exists('data/combined_data.csv'):
        # Auto-generate data for deployment
        with st.spinner("Generating synthetic data (first time setup)..."):
            import generate_data
            generate_data.main()
        
        st.success("Data generated successfully!")
    
    df = pd.read_csv('data/combined_data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df


@st.cache_resource
def train_model(df_train, use_exogenous=True):
    """
    Train and cache the SARIMAX model.
    Uses Streamlit cache to avoid retraining when parameters don't change.
    """
    forecaster = CallVolumeForecaster(use_exogenous=use_exogenous)
    forecaster.fit(df_train)
    return forecaster


def plot_forecast(historical_df, forecast_df, title="Call Volume Forecast"):
    """Create interactive forecast plot."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines+markers',
        name='Historical Volume',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    # Lower bound line
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='Lower Bound (80% CI)',
        line=dict(color='rgba(255, 127, 14, 0.5)', width=1, dash='dot')
    ))
    
    # Upper bound line
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Bound (80% CI)',
        line=dict(color='rgba(255, 127, 14, 0.5)', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Call Volume",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_components(forecast_df, model):
    """Plot forecast components (trend, seasonality, etc.)."""
    components = model.get_components(forecast_df)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trend', 'Weekly Seasonality', 'Yearly Seasonality', 'Axiom Ray AI Contribution'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=forecast_df['ds'], y=components['trend'], name='Trend', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Weekly seasonality
    if 'weekly' in components.columns:
        fig.add_trace(
            go.Scatter(x=forecast_df['ds'], y=components['weekly'], name='Weekly', line=dict(color='#ff7f0e')),
            row=1, col=2
        )
    
    # Yearly seasonality
    if 'yearly' in components.columns:
        fig.add_trace(
            go.Scatter(x=forecast_df['ds'], y=components['yearly'], name='Yearly', line=dict(color='#2ca02c')),
            row=2, col=1
        )
    
    # Axiom Ray contribution
    axiom_col = [col for col in components.columns if 'axiom' in col.lower()]
    if axiom_col:
        fig.add_trace(
            go.Scatter(x=forecast_df['ds'], y=forecast_df[axiom_col[0]], name='Axiom Ray Score', 
                      line=dict(color='#d62728'), mode='lines+markers'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, template='plotly_white')
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Contribution", row=1, col=1)
    fig.update_yaxes(title_text="Contribution", row=1, col=2)
    fig.update_yaxes(title_text="Contribution", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)
    
    return fig


def plot_axiom_ray_correlation(df):
    """Plot relationship between Axiom Ray predictions and call volume."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Call Volume vs Axiom Ray AI Score', 'Correlation Analysis'),
        vertical_spacing=0.15
    )
    
    # Time series comparison
    fig.add_trace(
        go.Scatter(x=df['ds'], y=df['y'], name='Call Volume', line=dict(color='#1f77b4'), yaxis='y'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['ds'], y=df['axiom_ray_score'], name='Axiom Ray Score', 
                  line=dict(color='#d62728'), yaxis='y2'),
        row=1, col=1
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['axiom_ray_score'],
            y=df['y'],
            mode='markers',
            name='Volume vs Score',
            marker=dict(color='#2ca02c', size=6, opacity=0.6)
        ),
        row=2, col=1
    )
    
    # Calculate correlation
    correlation = df['y'].corr(df['axiom_ray_score'])
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Call Volume", row=1, col=1)
    fig.update_yaxes(title_text="Axiom Ray Score (0-100)", row=1, col=1, side='right', overlaying='y')
    fig.update_xaxes(title_text="Axiom Ray Score", row=2, col=1)
    fig.update_yaxes(title_text="Call Volume", row=2, col=1)
    
    # Add correlation annotation
    fig.add_annotation(
        text=f"Correlation: {correlation:.3f}",
        xref="x2", yref="y2",
        x=0.5, y=0.95,
        xanchor="center",
        showarrow=False,
        font=dict(size=14, color='black')
    )
    
    return fig, correlation


def main():
    """Main dashboard application."""
    # Header with SharkNinja branding
    st.markdown('<p class="main-header">SharkNinja Customer Support Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Axiom Ray AI | Predictive Analytics for Workforce Optimization</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar with SharkNinja context
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/SharkNinja_logo.svg/200px-SharkNinja_logo.svg.png", width=150)
    st.sidebar.header("Forecast Configuration")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar controls
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    forecast_periods = st.sidebar.slider("Forecast Weeks", 4, 26, 13, 1)  # Default: 13 weeks (3 months)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Parameters")
    changepoint_prior = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05, 0.01)
    seasonality_prior = st.sidebar.slider("Seasonality Prior Scale", 1.0, 20.0, 10.0, 1.0)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size)
    
    # Train model with progress visibility
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    def update_status(message, progress=0):
        """Update status message and progress bar"""
        status_container.info(f"**Status:** {message}")
        progress_bar.progress(progress)
    
    import time
    with st.spinner("Training SARIMAX models (baseline and enhanced with Axiom Ray)..."):
        try:
            update_status("Initializing models...", 10)
            
            # Train BASELINE model (without Axiom Ray)
            update_status("Training BASELINE model (no Axiom Ray)...", 20)
            start_time = time.time()
            forecaster_baseline = CallVolumeForecaster(use_exogenous=False)
            forecaster_baseline.fit(train_df)
            baseline_time = time.time() - start_time
            
            # Train ENHANCED model (with Axiom Ray)
            update_status("Training ENHANCED model (with Axiom Ray)...", 50)
            start_time = time.time()
            forecaster = CallVolumeForecaster(use_exogenous=True)
            forecaster.fit(train_df)
            enhanced_time = time.time() - start_time
            
            update_status("Comparing model performance...", 80)
            
            # Get comparison metrics
            comparison = compare_forecasts(train_df, test_df, forecast_periods)
            
            total_time = baseline_time + enhanced_time
            st.success(f"✓ **Both models trained in {total_time:.1f} seconds!**")
            
            update_status("Model training complete! Generating forecasts...", 90)
            progress_bar.progress(100)
            
            # Show model comparison in expandable section
            with st.expander("📊 Model Comparison & Diagnostics (click to view)", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline Model (No Axiom Ray)**")
                    st.write(f"- MAPE: {comparison['baseline']['metrics']['MAPE']:.2f}%")
                    st.write(f"- MAE: {comparison['baseline']['metrics']['MAE']:.1f}")
                    st.write(f"- Training time: {baseline_time:.1f}s")
                with col2:
                    st.markdown("**Enhanced Model (With Axiom Ray)**")
                    if 'enhanced' in comparison:
                        st.write(f"- MAPE: {comparison['enhanced']['metrics']['MAPE']:.2f}%")
                        st.write(f"- MAE: {comparison['enhanced']['metrics']['MAE']:.1f}")
                        st.write(f"- Training time: {enhanced_time:.1f}s")
                        st.markdown(f"**Improvement: {comparison['improvement']['MAPE_pct_improvement']:.1f}% better MAPE**")
                    else:
                        st.info("Axiom Ray data not available")
                    
        except Exception as e:
            update_status(f"Error during training: {str(e)}", 0)
            st.error(f"Training failed: {str(e)}")
            st.stop()
    
    # Clear status after training
    status_container.empty()
    progress_bar.empty()
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Historical Weeks", len(df))
    with col2:
        st.metric("Training Weeks", len(train_df))
    with col3:
        st.metric("Test Weeks", len(test_df))
    with col4:
        avg_volume = df['y'].mean()
        st.metric("Avg Weekly Volume", f"{avg_volume:.0f}")
    
    st.markdown("---")
    
    # Generate forecast and metrics for all tabs
    future_forecast = forecaster.forecast_future(periods=forecast_periods, future_exog=test_df)
    metrics, eval_df = forecaster.evaluate(test_df)
    
    # Also get baseline forecast for comparison
    future_forecast_baseline = forecaster_baseline.forecast_future(periods=forecast_periods)
    metrics_baseline, eval_df_baseline = forecaster_baseline.evaluate(test_df)
    
    # Tabs - Added Executive Summary first for VP presentation
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Executive Summary", 
        "Forecast Comparison", 
        "Scenario Planning",
        "Product Breakdown",
        "Business Impact", 
        "Components", 
        "Axiom Ray Analysis", 
        "Model Performance"
    ])
    
    # Executive Summary Tab
    with tab0:
        st.header("Executive Summary")
        st.markdown("### Key Insights & Business Impact")
        
        # Calculate business metrics (use default business parameters for exec summary)
        default_hourly_rate = 25.0
        default_overhead = 1.35
        default_aht = 4.5
        roi_metrics = calculate_roi(future_forecast, train_df, default_aht, default_hourly_rate, default_overhead)
        risk_periods = identify_risk_periods(future_forecast)
        high_risk_count = risk_periods['is_high_risk'].sum()
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Forecast Accuracy", 
                f"{100 - metrics['MAPE']:.1f}%",
                delta=f"{metrics['MAPE']:.1f}% error",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Potential Savings",
                f"${roi_metrics['total_savings']:,.0f}",
                delta=f"{roi_metrics['savings_percentage']:.1f}% vs reactive",
                delta_color="normal"
            )
        with col3:
            st.metric(
                "High-Risk Periods",
                f"{high_risk_count} weeks",
                delta=f"Next {forecast_periods} weeks",
                delta_color="inverse"
            )
        with col4:
            avg_forecast = future_forecast['yhat'].mean()
            avg_historical = train_df['y'].mean()
            change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100
            st.metric(
                "Volume Trend",
                f"{avg_forecast:.0f} calls/week",
                delta=f"{change_pct:+.1f}% vs historical",
                delta_color="normal" if change_pct > 0 else "inverse"
            )
        
        st.markdown("---")
        
        # Executive Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecast Overview")
            st.markdown(f"""
            - **Forecast Period**: Next {forecast_periods} weeks
            - **Average Volume**: {future_forecast['yhat'].mean():.0f} calls/week
            - **Peak Volume**: {future_forecast['yhat'].max():.0f} calls/week
            - **Model Confidence**: {metrics['Within_CI_%']:.1f}% of actuals within prediction bounds
            """)
            
            # Key Dates
            peak_week = future_forecast.loc[future_forecast['yhat'].idxmax(), 'ds']
            st.markdown(f"""
            **Peak Volume Alert**
            - **Date**: {peak_week.strftime('%B %d, %Y')}
            - **Expected Volume**: {future_forecast['yhat'].max():.0f} calls
            - **Action Required**: Increase staffing capacity
            """)
        
        with col2:
            st.subheader("Business Impact")
            st.markdown(f"""
            - **Cost Optimization**: ${roi_metrics['total_savings']:,.0f} savings vs reactive staffing
            - **Efficiency Gain**: {roi_metrics['agents_saved']:.1f} fewer agents needed on average
            - **ROI**: {roi_metrics['savings_percentage']:.1f}% cost reduction
            - **Forecast Period Cost**: ${roi_metrics['total_forecast_cost']:,.0f}
            """)
            
            # Axiom Ray Value
            if 'axiom_ray_score' in df.columns:
                recent_axiom = df['axiom_ray_score'].tail(4).mean()
                st.markdown(f"""
                **Axiom Ray AI Early Warning**
                - **Current Score**: {recent_axiom:.0f}/100
                - **Lead Time**: 2 weeks advance notice
                - **Correlation**: {df['y'].corr(df['axiom_ray_score']):.2f} with volume
                """)
        
        st.markdown("---")
        
        # Action Items
        st.subheader("Recommended Actions")
        
        # Identify top risk periods
        top_risks = risk_periods.nlargest(3, 'yhat')[['ds', 'yhat', 'yhat_upper']]
        
        action_items = []
        for idx, row in top_risks.iterrows():
            action_items.append({
                'Priority': 'High' if row['yhat'] > future_forecast['yhat'].quantile(0.9) else 'Medium',
                'Date': row['ds'].strftime('%Y-%m-%d'),
                'Expected Volume': f"{row['yhat']:.0f}",
                'Upper Bound': f"{row['yhat_upper']:.0f}",
                'Action': 'Increase staffing by 20-30%' if row['yhat'] > future_forecast['yhat'].quantile(0.9) else 'Monitor closely'
            })
        
        action_df = pd.DataFrame(action_items)
        st.dataframe(action_df, use_container_width=True, hide_index=True)
        
        # Export button
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = future_forecast.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col2:
            # Create summary report
            summary_text = f"""
EXECUTIVE SUMMARY - Call Center Volume Forecast
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

KEY METRICS:
- Forecast Period: {forecast_periods} weeks
- Average Volume: {future_forecast['yhat'].mean():.0f} calls/week
- Model Accuracy: {100 - metrics['MAPE']:.1f}%
- Potential Savings: ${roi_metrics['total_savings']:,.0f}

HIGH-RISK PERIODS:
{chr(10).join([f"- {row['Date']}: {row['Expected Volume']} calls (Action: {row['Action']})" for _, row in action_df.iterrows()])}

RECOMMENDATIONS:
1. Proactively staff for identified high-volume periods
2. Monitor Axiom Ray AI scores for early warning signals
3. Implement dynamic staffing based on forecast
4. Review and adjust forecast weekly
            """
            st.download_button(
                label="Download Summary Report",
                data=summary_text,
                file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with tab1:
        st.header("Forecast Comparison: With vs Without Axiom Ray")
        st.markdown("**Compare the forecasting accuracy with and without using Axiom Ray as a leading indicator.**")
        
        # Model comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mape_improvement = metrics_baseline['MAPE'] - metrics['MAPE']
            st.metric("Baseline MAPE", f"{metrics_baseline['MAPE']:.1f}%", 
                     delta=f"Without Axiom Ray", delta_color="off")
        with col2:
            st.metric("Enhanced MAPE", f"{metrics['MAPE']:.1f}%", 
                     delta=f"{-mape_improvement:.1f}% better", delta_color="normal")
        with col3:
            mae_improvement = metrics_baseline['MAE'] - metrics['MAE']
            st.metric("MAE Improvement", f"{mae_improvement:.0f} calls",
                     delta=f"{(mae_improvement/metrics_baseline['MAE']*100):.1f}% reduction")
        with col4:
            accuracy_enhanced = 100 - metrics['MAPE']
            st.metric("Enhanced Accuracy", f"{accuracy_enhanced:.1f}%",
                     delta="With Axiom Ray")
        
        st.markdown("---")
        
        # Comparison chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=train_df['ds'], y=train_df['y'],
            name='Historical Volume', line=dict(color='#1f77b4', width=2),
            mode='lines'
        ))
        
        # Baseline forecast (without Axiom Ray)
        fig.add_trace(go.Scatter(
            x=future_forecast_baseline['ds'], y=future_forecast_baseline['yhat'],
            name='Baseline Forecast (No Axiom Ray)', 
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            mode='lines'
        ))
        
        # Enhanced forecast (with Axiom Ray)
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat'],
            name='Enhanced Forecast (With Axiom Ray)', 
            line=dict(color='#2ca02c', width=3),
            mode='lines'
        ))
        
        # Confidence interval for enhanced
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat_upper'],
            name='Upper Bound', line=dict(width=0), mode='lines', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat_lower'],
            name='Confidence Interval', line=dict(width=0), mode='lines',
            fill='tonexty', fillcolor='rgba(44, 160, 44, 0.2)'
        ))
        
        fig.update_layout(
            title=f"Forecast Comparison - Next {forecast_periods} Weeks",
            xaxis_title="Date",
            yaxis_title="Call Volume",
            height=500,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Side-by-side forecast tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseline Forecast (No Axiom Ray)")
            baseline_table = pd.DataFrame({
                'Date': future_forecast_baseline['ds'].dt.strftime('%Y-%m-%d'),
                'Forecast': future_forecast_baseline['yhat'].round(0).astype(int)
            })
            st.dataframe(baseline_table, use_container_width=True, height=300)
        
        with col2:
            st.subheader("Enhanced Forecast (With Axiom Ray)")
            enhanced_table = pd.DataFrame({
                'Date': future_forecast['ds'].dt.strftime('%Y-%m-%d'),
                'Forecast': future_forecast['yhat'].round(0).astype(int),
                'Difference': (future_forecast['yhat'] - future_forecast_baseline['yhat']).round(0).astype(int)
            })
            st.dataframe(enhanced_table, use_container_width=True, height=300)
        
        # Why Axiom Ray helps
        st.markdown("---")
        st.subheader("Why Axiom Ray Improves Forecasting")
        st.markdown("""
        **Axiom Ray is a 2-week leading indicator** that detects support volume changes before they happen:
        
        | Signal Source | What It Detects | Lead Time |
        |--------------|-----------------|-----------|
        | Social Media | Customer complaints & viral issues | 1-2 weeks |
        | Product Reviews | Rating drops & negative sentiment | 1-3 weeks |
        | Warranty Claims | Defect patterns & returns | 2-4 weeks |
        | Search Trends | "Product X not working" queries | 1-2 weeks |
        
        By incorporating these early warning signals, the forecast can anticipate volume spikes 
        **before** they occur, enabling proactive staffing adjustments.
        """)
    
    # Scenario Planning Tab
    with tab2:
        st.header("Scenario Planning & What-If Analysis")
        st.markdown("Explore different scenarios to understand how changes in volume affect staffing and costs.")
        
        # Scenario selection
        col1, col2 = st.columns(2)
        with col1:
            scenario = st.selectbox(
                "Select Scenario",
                ["Baseline Forecast", "High Growth (+20%)", "Low Growth (-15%)", 
                 "Holiday Surge (+35%)", "Economic Downturn (-25%)", "Custom"]
            )
        
        with col2:
            if scenario == "Custom":
                custom_adjustment = st.slider("Custom Volume Adjustment (%)", -50, 100, 0, 5)
            else:
                custom_adjustment = 0
        
        # Apply scenario multiplier
        scenario_multipliers = {
            "Baseline Forecast": 1.0,
            "High Growth (+20%)": 1.20,
            "Low Growth (-15%)": 0.85,
            "Holiday Surge (+35%)": 1.35,
            "Economic Downturn (-25%)": 0.75,
            "Custom": 1 + (custom_adjustment / 100)
        }
        
        multiplier = scenario_multipliers[scenario]
        
        # Create scenario forecast
        scenario_forecast = future_forecast.copy()
        scenario_forecast['yhat'] = scenario_forecast['yhat'] * multiplier
        scenario_forecast['yhat_lower'] = scenario_forecast['yhat_lower'] * multiplier
        scenario_forecast['yhat_upper'] = scenario_forecast['yhat_upper'] * multiplier
        
        # Display comparison
        st.subheader(f"Scenario: {scenario}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            baseline_avg = future_forecast['yhat'].mean()
            scenario_avg = scenario_forecast['yhat'].mean()
            st.metric("Avg Weekly Volume", f"{scenario_avg:.0f}", 
                     delta=f"{scenario_avg - baseline_avg:+.0f} vs baseline")
        with col2:
            # Staffing impact (simplified)
            baseline_agents = baseline_avg / 80  # ~80 calls per agent per week
            scenario_agents = scenario_avg / 80
            st.metric("Agents Needed", f"{scenario_agents:.1f}",
                     delta=f"{scenario_agents - baseline_agents:+.1f}")
        with col3:
            # Cost impact
            hourly_rate_default = 25
            baseline_cost = baseline_agents * 40 * hourly_rate_default * 1.35 * forecast_periods
            scenario_cost = scenario_agents * 40 * hourly_rate_default * 1.35 * forecast_periods
            st.metric("Total Labor Cost", f"${scenario_cost:,.0f}",
                     delta=f"${scenario_cost - baseline_cost:+,.0f}")
        with col4:
            st.metric("Volume Change", f"{(multiplier - 1) * 100:+.0f}%",
                     delta=f"{scenario_forecast['yhat'].sum() - future_forecast['yhat'].sum():+.0f} calls")
        
        st.markdown("---")
        
        # Scenario comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat'],
            name='Baseline Forecast', line=dict(color='#1f77b4', dash='dash'),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_forecast['ds'], y=scenario_forecast['yhat'],
            name=f'{scenario}', line=dict(color='#ff7f0e', width=3),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_forecast['ds'], y=scenario_forecast['yhat_upper'],
            name='Upper Bound', line=dict(color='#ff7f0e', width=0),
            mode='lines', showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_forecast['ds'], y=scenario_forecast['yhat_lower'],
            name='Lower Bound', line=dict(color='#ff7f0e', width=0),
            fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)',
            mode='lines', showlegend=False
        ))
        
        fig.update_layout(
            title=f"Scenario Comparison: {scenario}",
            xaxis_title="Date",
            yaxis_title="Call Volume",
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario summary table
        st.subheader("Weekly Breakdown")
        scenario_table = pd.DataFrame({
            'Week': scenario_forecast['ds'].dt.strftime('%Y-%m-%d'),
            'Baseline': future_forecast['yhat'].astype(int),
            'Scenario': scenario_forecast['yhat'].astype(int),
            'Difference': (scenario_forecast['yhat'] - future_forecast['yhat']).astype(int),
            'Agents (Scenario)': (scenario_forecast['yhat'] / 80).round(1)
        })
        st.dataframe(scenario_table, use_container_width=True, height=300)
    
    # Product Breakdown Tab
    with tab3:
        st.header("Product Category Breakdown")
        st.markdown("Analyze call volume by SharkNinja product lines: **Shark** (vacuums, hair care, air purifiers) vs **Ninja** (kitchen appliances).")
        
        # Check if product breakdown data exists
        if 'shark_volume' in df.columns and 'ninja_volume' in df.columns:
            # Historical product split
            total_shark = df['shark_volume'].sum()
            total_ninja = df['ninja_volume'].sum()
            total_volume = total_shark + total_ninja
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Shark Volume", f"{total_shark:,}", 
                         delta=f"{total_shark/total_volume*100:.1f}% of total")
            with col2:
                st.metric("Ninja Volume", f"{total_ninja:,}",
                         delta=f"{total_ninja/total_volume*100:.1f}% of total")
            with col3:
                shark_trend = df['shark_volume'].tail(13).mean() - df['shark_volume'].head(13).mean()
                st.metric("Shark Trend", f"{shark_trend:+.0f}/week",
                         delta="vs 1 year ago")
            with col4:
                ninja_trend = df['ninja_volume'].tail(13).mean() - df['ninja_volume'].head(13).mean()
                st.metric("Ninja Trend", f"{ninja_trend:+.0f}/week",
                         delta="vs 1 year ago")
            
            st.markdown("---")
            
            # Product volume over time
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=("Volume by Brand Over Time", "Brand Split Pie Chart",
                                             "Monthly Seasonality by Brand", "Product Subcategories"),
                               specs=[[{"type": "xy"}, {"type": "domain"}],
                                      [{"type": "xy"}, {"type": "xy"}]])
            
            # Time series by brand
            fig.add_trace(
                go.Scatter(x=df['ds'], y=df['shark_volume'], name='Shark', 
                          line=dict(color='#1f77b4')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['ds'], y=df['ninja_volume'], name='Ninja',
                          line=dict(color='#ff7f0e')),
                row=1, col=1
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(labels=['Shark', 'Ninja'], values=[total_shark, total_ninja],
                      marker_colors=['#1f77b4', '#ff7f0e'], hole=0.4),
                row=1, col=2
            )
            
            # Monthly seasonality
            df_monthly = df.copy()
            df_monthly['month'] = df_monthly['ds'].dt.month
            monthly_shark = df_monthly.groupby('month')['shark_volume'].mean()
            monthly_ninja = df_monthly.groupby('month')['ninja_volume'].mean()
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig.add_trace(
                go.Bar(x=months, y=monthly_shark.values, name='Shark (Monthly)', 
                      marker_color='#1f77b4', showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=months, y=monthly_ninja.values, name='Ninja (Monthly)',
                      marker_color='#ff7f0e', showlegend=False),
                row=2, col=1
            )
            
            # Subcategories (if available)
            subcategories = []
            subcategory_values = []
            colors = []
            
            shark_subs = ['shark_vacuums', 'shark_haircare', 'shark_airpurifiers']
            ninja_subs = ['ninja_airfryers', 'ninja_blenders', 'ninja_coffee', 'ninja_grills']
            
            shark_colors = ['#1f77b4', '#4a90d9', '#7eb3ed']
            ninja_colors = ['#ff7f0e', '#ffaa4d', '#ffc67d', '#ffe0ad']
            
            for i, col in enumerate(shark_subs):
                if col in df.columns:
                    subcategories.append(col.replace('shark_', 'Shark ').title())
                    subcategory_values.append(df[col].sum())
                    colors.append(shark_colors[i % len(shark_colors)])
            
            for i, col in enumerate(ninja_subs):
                if col in df.columns:
                    subcategories.append(col.replace('ninja_', 'Ninja ').title())
                    subcategory_values.append(df[col].sum())
                    colors.append(ninja_colors[i % len(ninja_colors)])
            
            if subcategories:
                fig.add_trace(
                    go.Bar(x=subcategories, y=subcategory_values, 
                          marker_color=colors, showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(height=700, showlegend=True, barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecasted product breakdown
            st.subheader("Forecasted Product Breakdown")
            st.markdown("Estimated future volume by product line based on historical patterns.")
            
            # Calculate average split from recent data
            recent_shark_pct = df['shark_volume'].tail(26).sum() / df['y'].tail(26).sum()
            recent_ninja_pct = 1 - recent_shark_pct
            
            forecast_breakdown = pd.DataFrame({
                'Week': future_forecast['ds'].dt.strftime('%Y-%m-%d'),
                'Total Volume': future_forecast['yhat'].astype(int),
                'Shark (Est.)': (future_forecast['yhat'] * recent_shark_pct).astype(int),
                'Ninja (Est.)': (future_forecast['yhat'] * recent_ninja_pct).astype(int)
            })
            st.dataframe(forecast_breakdown, use_container_width=True, height=300)
            
        else:
            st.info("Product breakdown data not available. Add 'shark_volume' and 'ninja_volume' columns to enable this analysis.")
            
            # Show placeholder
            st.markdown("""
            **Expected data columns:**
            - `shark_volume`: Weekly call volume for Shark products (vacuums, hair care, air purifiers)
            - `ninja_volume`: Weekly call volume for Ninja products (air fryers, blenders, coffee makers, grills)
            
            **Benefits of product-level forecasting:**
            - Target training resources by product expertise
            - Plan for product launch support needs
            - Identify seasonal patterns by product line
            """)
    
    # Business Impact Tab (renumbered to tab4)
    with tab4:
        st.header("Business Impact & Staffing Analysis")
        
        # Sidebar for business parameters
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Business Parameters")
        hourly_rate = st.sidebar.slider("Hourly Rate ($)", 15, 50, 25, 1)
        overhead_rate = st.sidebar.slider("Overhead Multiplier", 1.2, 1.5, 1.35, 0.05)
        avg_handle_time = st.sidebar.slider("Avg Handle Time (min)", 3.0, 7.0, 4.5, 0.1)
        service_level = st.sidebar.slider("Service Level Target", 0.70, 0.95, 0.80, 0.01)
        
        # Calculate staffing needs
        staffing_df = calculate_staffing_needs(future_forecast, avg_handle_time, service_level)
        costs_df = calculate_costs(staffing_df, hourly_rate, overhead_rate)
        roi_metrics = calculate_roi(future_forecast, train_df, avg_handle_time, hourly_rate, overhead_rate)
        
        # Key Business Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_cost = costs_df['weekly_cost'].sum()
            st.metric("Total Forecast Cost", f"${total_cost:,.0f}", 
                     delta=f"${costs_df['weekly_cost'].mean():,.0f}/week avg")
        with col2:
            avg_agents = staffing_df['agents_needed'].mean()
            st.metric("Avg Agents Needed", f"{avg_agents:.1f}", 
                     delta=f"{staffing_df['agents_needed'].max():.0f} peak")
        with col3:
            st.metric("Cost Savings", f"${roi_metrics['total_savings']:,.0f}",
                     delta=f"{roi_metrics['savings_percentage']:.1f}%")
        with col4:
            st.metric("Agents Saved", f"{roi_metrics['agents_saved']:.1f}",
                     delta="vs reactive staffing")
        
        st.markdown("---")
        
        # Staffing Forecast Chart
        fig_staffing = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_staffing.add_trace(
            go.Scatter(x=staffing_df['ds'], y=staffing_df['yhat'], 
                      name='Call Volume', line=dict(color='#1f77b4')),
            secondary_y=False
        )
        
        fig_staffing.add_trace(
            go.Scatter(x=staffing_df['ds'], y=staffing_df['agents_needed'],
                      name='Agents Needed', line=dict(color='#ff7f0e', width=2),
                      fill='tonexty'),
            secondary_y=True
        )
        
        fig_staffing.update_xaxes(title_text="Date")
        fig_staffing.update_yaxes(title_text="Call Volume", secondary_y=False)
        fig_staffing.update_yaxes(title_text="Agents Needed", secondary_y=True)
        fig_staffing.update_layout(
            title="Call Volume vs Staffing Requirements",
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig_staffing, use_container_width=True)
        
        # Cost Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Staffing Requirements")
            staffing_display = staffing_df[['ds', 'yhat', 'agents_needed', 'fte_needed']].copy()
            staffing_display['ds'] = staffing_display['ds'].dt.strftime('%Y-%m-%d')
            staffing_display.columns = ['Date', 'Calls', 'Agents', 'FTE']
            st.dataframe(staffing_display, use_container_width=True, height=400)
        
        with col2:
            st.subheader("Cost Breakdown")
            costs_display = costs_df[['ds', 'weekly_cost', 'monthly_cost']].copy()
            costs_display['ds'] = costs_display['ds'].dt.strftime('%Y-%m-%d')
            costs_display.columns = ['Date', 'Weekly Cost ($)', 'Monthly Cost ($)']
            costs_display['Weekly Cost ($)'] = costs_display['Weekly Cost ($)'].apply(lambda x: f"${x:,.0f}")
            costs_display['Monthly Cost ($)'] = costs_display['Monthly Cost ($)'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(costs_display, use_container_width=True, height=400)
        
        # ROI Analysis
        st.markdown("---")
        st.subheader("ROI Analysis")
        
        roi_col1, roi_col2 = st.columns(2)
        with roi_col1:
            avg_weekly_forecast = roi_metrics['total_forecast_cost'] / roi_metrics['forecast_period_weeks'] if roi_metrics['forecast_period_weeks'] > 0 else 0
            st.markdown(f"""
            **Forecast-Based Staffing:**
            - Total Cost: ${roi_metrics['total_forecast_cost']:,.0f}
            - Avg Agents: {roi_metrics['avg_agents_forecast']:.1f}
            - Avg Weekly Cost: ${avg_weekly_forecast:,.0f}
            """)
        
        with roi_col2:
            avg_weekly_reactive = roi_metrics['total_reactive_cost'] / roi_metrics['forecast_period_weeks'] if roi_metrics['forecast_period_weeks'] > 0 else 0
            st.markdown(f"""
            **Reactive Staffing (Baseline):**
            - Total Cost: ${roi_metrics['total_reactive_cost']:,.0f}
            - Avg Agents: {roi_metrics['avg_agents_reactive']:.1f}
            - Avg Weekly Cost: ${avg_weekly_reactive:,.0f}
            """)
        
        # Additional ROI insights
        if roi_metrics.get('annualized_savings', 0) > 0:
            st.markdown(f"""
            **Annualized Impact:** ${roi_metrics['annualized_savings']:,.0f} potential annual savings with forecast-based staffing
            """)
        
        # Risk Periods
        st.markdown("---")
        st.subheader("High-Risk Periods Requiring Attention")
        risk_periods = identify_risk_periods(future_forecast)
        high_risk = risk_periods[risk_periods['is_high_risk']][['ds', 'yhat', 'yhat_upper', 'risk_level']].copy()
        high_risk['ds'] = high_risk['ds'].dt.strftime('%Y-%m-%d')
        high_risk.columns = ['Date', 'Forecasted Volume', 'Upper Bound', 'Risk Level']
        if len(high_risk) > 0:
            st.dataframe(high_risk, use_container_width=True)
        else:
            st.info("No high-risk periods identified in forecast period.")
    
    with tab5:
        st.header("Forecast Components")
        st.markdown("Breakdown of forecast components: trend, seasonality, and external regressors.")
        
        # Get full forecast including history for components
        full_forecast = forecaster.predict(train_df)
        future_forecast = forecaster.forecast_future(periods=forecast_periods)
        combined_forecast = pd.concat([full_forecast, future_forecast]).sort_values('ds')
        
        fig = plot_components(combined_forecast, forecaster)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("Axiom Ray AI Analysis")
        st.markdown("**2-Week Leading Indicator** - Axiom Ray score at week T predicts call volume at week T+2.")
        
        # Check if we have Axiom Ray data
        has_axiom = 'axiom_ray_score' in df.columns
        
        if has_axiom:
            # Calculate LEADING correlations (this is the key insight!)
            axiom = df['axiom_ray_score'].values
            volume = df['y'].values
            
            # Concurrent correlation (axiom[t] vs volume[t])
            concurrent_corr = np.corrcoef(axiom[:-2], volume[:-2])[0,1]
            
            # Leading correlation (axiom[t] vs volume[t+2]) - should be HIGHER
            leading_corr = np.corrcoef(axiom[:-2], volume[2:])[0,1]
            
            correlation = leading_corr  # Use leading correlation as primary
            
            # Key metrics row - emphasize LEADING indicator nature
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("2-Week Lead Correlation", f"{leading_corr:.2f}", 
                         delta=f"+{(leading_corr - concurrent_corr):.2f} vs concurrent")
            with col2:
                st.metric("Concurrent Correlation", f"{concurrent_corr:.2f}",
                         delta="Same-week comparison", delta_color="off")
            with col3:
                recent_score = df['axiom_ray_score'].tail(4).mean()
                st.metric("Current Signal", f"{recent_score:.0f}/100", 
                         delta="Predicts next 2 weeks", delta_color="off")
            with col4:
                r_squared = leading_corr ** 2
                st.metric("Predictive Power (R²)", f"{r_squared:.1%}",
                         delta="Leading indicator")
            
            st.markdown("---")
            
            # Dual-axis time series chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=df['ds'], y=df['y'], name='Call Volume', 
                          line=dict(color='#1f77b4', width=2)),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=df['ds'], y=df['axiom_ray_score'], name='Axiom Ray Score', 
                          line=dict(color='#d62728', width=2)),
                secondary_y=True
            )
            
            fig.update_layout(
                title="Axiom Ray Signal vs Actual Call Volume",
                height=400,
                hovermode='x unified'
            )
            fig.update_yaxes(title_text="Call Volume", secondary_y=False)
            fig.update_yaxes(title_text="Axiom Ray Score (0-100)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal components breakdown
            st.subheader("Signal Component Analysis")
            
            signal_cols = ['social_sentiment', 'review_trend', 'warranty_claims', 'search_trends', 'retail_alerts']
            available_signals = [c for c in signal_cols if c in df.columns]
            
            if available_signals:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Component correlations
                    st.markdown("**Individual Signal Correlations with Volume:**")
                    signal_corrs = {}
                    for sig in available_signals:
                        corr = df['y'].corr(df[sig])
                        signal_corrs[sig] = corr
                    
                    signal_df = pd.DataFrame({
                        'Signal': [s.replace('_', ' ').title() for s in signal_corrs.keys()],
                        'Correlation': list(signal_corrs.values()),
                        'Strength': ['Strong' if abs(c) > 0.5 else 'Moderate' if abs(c) > 0.3 else 'Weak' 
                                    for c in signal_corrs.values()]
                    })
                    st.dataframe(signal_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Current signal levels
                    st.markdown("**Current Signal Levels (Last 4 Weeks Avg):**")
                    current_signals = {s.replace('_', ' ').title(): df[s].tail(4).mean() 
                                      for s in available_signals}
                    
                    fig_bar = go.Figure(go.Bar(
                        x=list(current_signals.values()),
                        y=list(current_signals.keys()),
                        orientation='h',
                        marker_color=['#ff6b6b' if v > 60 else '#ffd93d' if v > 40 else '#6bcb77' 
                                     for v in current_signals.values()]
                    ))
                    fig_bar.update_layout(
                        height=250,
                        xaxis_title="Score (0-100)",
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Scatter plot with regression
            st.subheader("Correlation Analysis")
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=df['axiom_ray_score'],
                y=df['y'],
                mode='markers',
                name='Weekly Data Points',
                marker=dict(color='#1f77b4', size=8, opacity=0.6)
            ))
            
            # Add trend line
            z = np.polyfit(df['axiom_ray_score'], df['y'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['axiom_ray_score'].min(), df['axiom_ray_score'].max(), 100)
            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                name=f'Trend (r={correlation:.2f})',
                line=dict(color='#d62728', width=2, dash='dash')
            ))
            
            fig_scatter.update_layout(
                title="Axiom Ray Score vs Call Volume (Weekly)",
                xaxis_title="Axiom Ray Score",
                yaxis_title="Call Volume",
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            # Generate synthetic Axiom Ray data for display
            st.warning("Axiom Ray data not found in dataset. Regenerate data to include Axiom Ray signals.")
        
        st.markdown("---")
        st.markdown("""
        **How Axiom Ray Works for SharkNinja:**
        
        | Data Source | What It Monitors | Lead Time |
        |-------------|------------------|-----------|
        | Social Sentiment | Twitter, Reddit, Facebook mentions | 1-2 weeks |
        | Review Trends | Amazon, Best Buy, Target reviews | 1-3 weeks |
        | Warranty Claims | Return rates, replacement requests | 2-4 weeks |
        | Search Trends | Google searches for product issues | 1-2 weeks |
        | Retail Alerts | Partner feedback, inventory returns | 1-2 weeks |
        
        **Key Benefits:**
        - Proactive staffing adjustments before volume spikes
        - Early detection of product quality issues
        - Reduced overtime costs through better planning
        - Improved customer satisfaction via faster response
        """)
    
    with tab7:
        st.header("Model Performance on Test Set")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.1f}")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.1f}")
        with col3:
            st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
        with col4:
            st.metric("Within CI (%)", f"{metrics['Within_CI_%']:.1f}%")
        
        # Plot predictions vs actuals
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=eval_df['ds'],
            y=eval_df['y'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=eval_df['ds'],
            y=eval_df['yhat'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=eval_df['ds'].tolist() + eval_df['ds'].tolist()[::-1],
            y=eval_df['yhat_upper'].tolist() + eval_df['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            title="Predictions vs Actuals on Test Set",
            xaxis_title="Date",
            yaxis_title="Call Volume",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        eval_df['residual'] = eval_df['y'] - eval_df['yhat']
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=eval_df['ds'],
            y=eval_df['residual'],
            mode='lines+markers',
            name='Residuals',
            line=dict(color='#d62728', width=1),
            marker=dict(size=4)
        ))
        
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig2.update_layout(
            title="Residuals Plot",
            xaxis_title="Date",
            yaxis_title="Residual (Actual - Predicted)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Footer with SharkNinja branding
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: {DARK_GRAY}; padding: 20px;'>
        <p style='font-size: 1.1rem; font-weight: bold;'>SharkNinja Customer Support Forecasting</p>
        <p>Powered by <span style='color: {SHARK_TEAL};'>Axiom Ray AI</span> | Predictive Workforce Analytics</p>
        <p style='font-size: 0.9rem; color: gray;'>Demo for illustration purposes | Data is synthetic</p>
        <p style='font-size: 0.8rem; color: gray; margin-top: 10px;'>
            Supporting Shark (vacuums, hair care, air purifiers) & Ninja (blenders, air fryers, coffee makers, grills)
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()

