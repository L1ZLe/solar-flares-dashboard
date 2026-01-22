# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(page_title="Solar Flare EDA Dashboard", layout="wide")

# Title
st.title("🌞 Solar Flare Exploratory Analysis Dashboard")
st.markdown("**GOES-18 EUVS Data Analysis** | Interactive exploration of solar activity patterns")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load and parse the CSV file"""
    df = pd.read_csv('euvs_data_converted_summary.csv')
    
    # Parse timestamp (ISO format: 2022-09-24T123600Z)
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], format='%Y-%m-%dT%H%M%SZ', utc=True)
    
    # Extract time components
    df['date'] = df['timestamp_utc'].dt.date
    df['hour'] = df['timestamp_utc'].dt.hour
    df['month'] = df['timestamp_utc'].dt.month
    df['year'] = df['timestamp_utc'].dt.year
    df['day_of_week'] = df['timestamp_utc'].dt.day_name()
    
    # Convert flux to scientific notation for readability
    df['xrs_b_flux_sci'] = df['xrs_b_flux'].apply(lambda x: f'{x:.2e}')
    
    return df

df = load_data()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🎚️ Filters & Controls")

# Date range slider
min_date = df['timestamp_utc'].min()
max_date = df['timestamp_utc'].max()

date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, min_date + pd.Timedelta(days=30)),
    format="YYYY-MM-DD"
)

# Filter by date range
df_filtered = df[(df['timestamp_utc'] >= pd.Timestamp(date_range[0], tz='UTC')) & 
                  (df['timestamp_utc'] <= pd.Timestamp(date_range[1], tz='UTC'))]

# Flare class filter
flare_classes = ['All'] + sorted(df['flare_class'].dropna().unique().tolist())
selected_flare = st.sidebar.selectbox("Flare Class Filter:", flare_classes)

if selected_flare != 'All':
    df_filtered = df_filtered[df_filtered['flare_class'] == selected_flare]

# Flux range threshold
st.sidebar.markdown("### High-Activity Detection")
flux_threshold = st.sidebar.number_input(
    "XRS-B Flux Threshold (W/m²):",
    value=1e-6,
    format="%.2e"
)

df_filtered['is_high_activity'] = df_filtered['xrs_b_flux'] > flux_threshold

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS CARDS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("## 📊 Quick Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Records",
        value=f"{len(df_filtered):,}",
        delta=f"({len(df_filtered)/len(df)*100:.1f}% of total)"
    )

with col2:
    high_activity = df_filtered['is_high_activity'].sum()
    st.metric(
        label="High-Activity Minutes",
        value=f"{high_activity:,}",
        delta=f"{high_activity/len(df_filtered)*100:.1f}% of range"
    )

with col3:
    event_count = df_filtered['status'].notna().sum()
    st.metric(
        label="Detected Events",
        value=f"{event_count:,}",
        delta="EVENTSTART/PEAK/END"
    )

with col4:
    avg_flux = df_filtered['xrs_b_flux'].mean()
    st.metric(
        label="Mean XRS-B Flux",
        value=f"{avg_flux:.2e}",
        delta=f"W/m²"
    )

# ═══════════════════════════════════════════════════════════════════════════
# VIZ 1: TIME SERIES WITH FLARES HIGHLIGHTED
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📈 Time Series: XRS-B Flux Over Time")

# Create time series plot
fig_ts = go.Figure()

# Background trace (all data in gray)
fig_ts.add_trace(go.Scatter(
    x=df_filtered['timestamp_utc'],
    y=df_filtered['xrs_b_flux'],
    mode='lines',
    name='XRS-B Flux',
    line=dict(color='rgba(100, 150, 200, 0.6)', width=1),
    hovertemplate='<b>Time:</b> %{x|%Y-%m-%d %H:%M}<br><b>Flux:</b> %{y:.2e} W/m²<extra></extra>'
))

# High-activity highlights (red)
high_activity_data = df_filtered[df_filtered['is_high_activity']]
if len(high_activity_data) > 0:
    fig_ts.add_trace(go.Scatter(
        x=high_activity_data['timestamp_utc'],
        y=high_activity_data['xrs_b_flux'],
        mode='markers',
        name='High Activity',
        marker=dict(color='red', size=6, opacity=0.7),
        hovertemplate='<b>High Activity</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>Flux: %{y:.2e} W/m²<extra></extra>'
    ))

# Add threshold line
fig_ts.add_hline(
    y=flux_threshold,
    line_dash="dash",
    line_color="orange",
    annotation_text=f"Threshold: {flux_threshold:.2e}",
    annotation_position="right"
)

fig_ts.update_layout(
    title="XRS-B Flux Time Series",
    xaxis_title="Time (UTC)",
    yaxis_title="Flux (W/m²)",
    hovermode='x unified',
    height=400,
    template='plotly_white'
)

st.plotly_chart(fig_ts, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# VIZ 2: FLUX DISTRIBUTION (HISTOGRAM)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📊 Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    # Histogram with log scale
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=df_filtered['xrs_b_flux'],
        nbinsx=50,
        name='XRS-B Flux',
        marker=dict(color='steelblue', line=dict(color='darkblue', width=1)),
        hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    fig_hist.update_xaxes(type='log')
    fig_hist.update_layout(
        title="XRS-B Flux Distribution (Log Scale)",
        xaxis_title="Flux (W/m²)",
        yaxis_title="Frequency",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Flare class pie chart
    flare_counts = df_filtered['flare_class'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=flare_counts.index,
        values=flare_counts.values,
        hole=0.3,
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig_pie.update_layout(
        title="Flare Class Distribution",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# VIZ 3: DAILY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📅 Daily Activity Summary")

daily_stats = df_filtered.groupby('date').agg({
    'xrs_b_flux': ['max', 'mean', 'count'],
    'is_high_activity': 'sum'
}).reset_index()

daily_stats.columns = ['date', 'max_flux', 'avg_flux', 'total_minutes', 'high_activity_minutes']

fig_daily = go.Figure()

fig_daily.add_trace(go.Bar(
    x=daily_stats['date'],
    y=daily_stats['max_flux'],
    name='Daily Max Flux',
    marker=dict(color='darkred'),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Max Flux: %{y:.2e} W/m²<extra></extra>'
))

fig_daily.add_trace(go.Scatter(
    x=daily_stats['date'],
    y=daily_stats['high_activity_minutes'],
    name='High-Activity Minutes',
    yaxis='y2',
    mode='lines+markers',
    line=dict(color='orange', width=2),
    marker=dict(size=6),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>High-Activity Minutes: %{y}<extra></extra>'
))

fig_daily.update_layout(
    title="Daily Maximum Flux & High-Activity Minutes",
    xaxis_title="Date",
    yaxis_title="Max Flux (W/m²)",
    yaxis2=dict(
        title="High-Activity Minutes",
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    height=400,
    template='plotly_white'
)

st.plotly_chart(fig_daily, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# VIZ 4: EVENT STATUS BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🎯 Event Status Analysis")

status_counts = df_filtered['status'].value_counts()

fig_status = px.bar(
    x=status_counts.index,
    y=status_counts.values,
    labels={'x': 'Event Status', 'y': 'Count'},
    color=status_counts.index,
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_status.update_layout(
    title="Distribution of Event Status Types",
    height=350,
    template='plotly_white',
    showlegend=False
)

st.plotly_chart(fig_status, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🔍 Data Explorer")

# Show sample records
if st.checkbox("Show raw data sample"):
    st.dataframe(
        df_filtered[['timestamp_utc', 'xrs_b_flux', 'status', 'flare_class', 'integrated_flux']].head(100),
        use_container_width=True
    )

# Download button
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=df_filtered.to_csv(index=False),
    file_name=f"solar_flares_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
**Data Source:** GOES-18 EUVS 1-minute Averages  
**Dashboard Built For:** Florida Poly & New College Datathon  
**Analysis Type:** Exploratory Data Analysis (EDA)  
**Last Updated:** Jan 22, 2026
""")
