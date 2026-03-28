import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

# Safe plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

from src.config import DATA_RAW

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

from app.ui_utils import apply_custom_css, update_plotly_layout
apply_custom_css()

# ────────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">📊 Data Exploration Dashboard</div>
    <div class="header-subtitle">Analyze weather patterns across Australia (2008-2017)</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Load Data
# ────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_RAW)
    df['Date']  = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    return df

df = load_data()

# ────────────────────────────────────────────────────────────────────────
# Sidebar Filters
# ────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🎛️ Filters")
locations = st.sidebar.multiselect(
    "Select Locations",
    df['Location'].unique().tolist(),
    default=df['Location'].unique().tolist()[:5],
    help="Choose one or more locations to analyze"
)

years = st.sidebar.slider(
    "Year Range",
    int(df['Year'].min()), int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max())),
    help="Slide to select year range"
)

# Apply filters
mask = (df['Location'].isin(locations)) & (df['Year'].between(*years))
dff = df[mask]

# ────────────────────────────────────────────────────────────────────────
# Check Plotly Availability
# ────────────────────────────────────────────────────────────────────────
if not PLOTLY_AVAILABLE:
    st.error("❌ This page requires Plotly for visualization.")
    st.info("📚 Plotly is required for all charts on this page.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────
# KPI Cards
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(dff):,}",
        f"{len(dff)/len(df)*100:.1f}% of dataset",
        help="Number of records in filtered dataset"
    )

with col2:
    rain_days = (dff['RainTomorrow']=='Yes').sum()
    st.metric(
        "Rain Days",
        f"{rain_days:,}",
        f"{dff['RainTomorrow'].value_counts().get('Yes', 0)/len(dff)*100:.1f}% rainy",
        help="Days where it rained tomorrow"
    )

with col3:
    rain_rate = (dff['RainTomorrow']=='Yes').mean()*100
    st.metric(
        "Rain Probability",
        f"{rain_rate:.1f}%",
        "in selected period",
        help="Percentage of days with rain tomorrow"
    )

with col4:
    st.metric(
        "Locations",
        f"{dff['Location'].nunique()}",
        f"out of {df['Location'].nunique()} total",
        help="Number of unique locations in filter"
    )

st.divider()

# ────────────────────────────────────────────────────────────────────────
# Target Distribution
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Target Variable Distribution</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    counts = dff['RainTomorrow'].value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=['No Rain', 'Rain'] if 'No' in counts.index else ['Rain', 'No Rain'],
        title='Rain Distribution',
        color_discrete_map={'No Rain': '#10b981', 'Rain': '#3b82f6'},
        hole=0.4,
        height=350,
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie = update_plotly_layout(fig_pie)
    st.plotly_chart(fig_pie, width='stretch', key='rain_dist_chart')

with col2:
    monthly = dff.groupby('Month')['RainTomorrow'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    monthly.columns = ['Month', 'RainRate']
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly['MonthName'] = monthly['Month'].map(month_names)

    fig_bar = px.bar(
        monthly,
        x='MonthName',
        y='RainRate',
        title='Seasonal Rain Patterns',
        color='RainRate',
        color_continuous_scale='Blues',
        labels={'RainRate': 'Rain Probability (%)'},
        height=350,
    )
    fig_bar = update_plotly_layout(fig_bar)
    fig_bar.update_layout(
        hovermode='closest',
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width='stretch', key='monthly_rain_chart')

# ────────────────────────────────────────────────────────────────────────
# Temperature Analysis
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌡️ Temperature Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Temperature distribution
    fig_temp = px.histogram(
        dff.dropna(subset=['MaxTemp']),
        x='MaxTemp',
        nbins=50,
        title='Maximum Temperature Distribution',
        color_discrete_sequence=['#f97316'],
        labels={'MaxTemp': 'Max Temperature (°C)', 'count': 'Frequency'},
        height=350,
    )
    fig_temp = update_plotly_layout(fig_temp)
    st.plotly_chart(fig_temp, width='stretch', key='maxtemp_hist_chart')

with col2:
    # Temp by rain
    fig_temp_rain = px.box(
        dff.dropna(subset=['MaxTemp']),
        x='RainTomorrow',
        y='MaxTemp',
        title='Temperature by Rain Occurrence',
        color='RainTomorrow',
        color_discrete_map={'Yes': '#3b82f6', 'No': '#10b981'},
        height=350,
    )
    fig_temp_rain = update_plotly_layout(fig_temp_rain)
    fig_temp_rain.update_layout(showlegend=False)
    st.plotly_chart(fig_temp_rain, width='stretch', key='temp_rain_chart')

# ────────────────────────────────────────────────────────────────────────
# Rainfall Analysis
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💧 Rainfall Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Rainfall distribution
    fig_rain = px.histogram(
        dff.dropna(subset=['Rainfall']),
        x='Rainfall',
        nbins=50,
        title='Rainfall Distribution',
        color_discrete_sequence=['#0ea5e9'],
        labels={'Rainfall': 'Rainfall (mm)', 'count': 'Frequency'},
        height=350,
    )
    fig_rain = update_plotly_layout(fig_rain)
    st.plotly_chart(fig_rain, width='stretch', key='rainfall_hist_chart')

with col2:
    # Rainfall by location (top 10)
    location_rain = dff.groupby('Location')['Rainfall'].mean().nlargest(10).reset_index()
    fig_loc_rain = px.bar(
        location_rain,
        x='Rainfall',
        y='Location',
        orientation='h',
        title='Top 10 Locations by Avg Rainfall',
        color='Rainfall',
        color_continuous_scale='Blues',
        labels={'Rainfall': 'Average Rainfall (mm)'},
        height=350,
    )
    fig_loc_rain = update_plotly_layout(fig_loc_rain)
    fig_loc_rain.update_layout(showlegend=False)
    st.plotly_chart(fig_loc_rain, width='stretch', key='location_rain_chart')

# ────────────────────────────────────────────────────────────────────────
# Humidity & Wind Analysis
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💨 Humidity & Wind Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Humidity by rain
    fig_humid = go.Figure()
    for rain_val, label in [('Yes', 'Rainy Days'), ('No', 'No Rain Days')]:
        data = dff[dff['RainTomorrow']==rain_val]['Humidity9am'].dropna()
        fig_humid.add_trace(go.Box(
            y=data,
            name=label,
            marker_color='#3b82f6' if rain_val == 'Yes' else '#10b981'
        ))
    fig_humid = update_plotly_layout(fig_humid)
    fig_humid.update_layout(
        title='Humidity Levels by Rain Occurrence',
        yaxis_title='Humidity 9am (%)',
        height=350,
    )
    st.plotly_chart(fig_humid, width='stretch', key='humidity_rain_chart')

with col2:
    # Wind speed distribution
    fig_wind = px.histogram(
        dff.dropna(subset=['WindGustSpeed']),
        x='WindGustSpeed',
        nbins=50,
        title='Wind Gust Speed Distribution',
        color_discrete_sequence=['#8b5cf6'],
        labels={'WindGustSpeed': 'Wind Gust Speed (km/h)', 'count': 'Frequency'},
        height=350,
    )
    fig_wind = update_plotly_layout(fig_wind)
    st.plotly_chart(fig_wind, width='stretch', key='windspeed_hist_chart')

# ────────────────────────────────────────────────────────────────────────
# Correlation Heatmap
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Correlation Analysis</div>', unsafe_allow_html=True)

numeric_cols = ['MinTemp','MaxTemp','Rainfall','Humidity9am',
                'Humidity3pm','WindGustSpeed','Pressure9am']
corr_data = dff[numeric_cols].corr().round(2)

fig_heatmap = px.imshow(
    corr_data,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    aspect='auto',
    title='Feature Correlation Heatmap',
    height=450,
    labels=dict(color='Correlation')
)
fig_heatmap = update_plotly_layout(fig_heatmap)
st.plotly_chart(fig_heatmap, width='stretch', key='corr_heatmap_chart')

# ────────────────────────────────────────────────────────────────────────
# Rain by Location
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Rain Rate by Location</div>', unsafe_allow_html=True)

loc_rain = dff.groupby('Location')['RainTomorrow'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
loc_rain.columns = ['Location', 'RainRate']
loc_rain = loc_rain.sort_values('RainRate', ascending=True)

fig_loc = px.bar(
    loc_rain,
    x='RainRate',
    y='Location',
    orientation='h',
    title='Rain Probability by Location',
    color='RainRate',
    color_continuous_scale='Blues',
    labels={'RainRate': 'Rain Probability (%)'},
    height=600,
)
fig_loc = update_plotly_layout(fig_loc)
fig_loc.update_layout(showlegend=False)
st.plotly_chart(fig_loc, width='stretch', key='location_rain_prob_chart')

# ────────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align: center; color: #94a3b8; font-size: 12px; padding: 20px 0;">
    <p>EDA Dashboard • Dataset: weatherAUS.csv (145,460 records)</p>
    <p>Data span: 2008-2017 • 49 locations across Australia</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Target Distribution")
    counts = dff['RainTomorrow'].value_counts()
    fig = px.pie(
        values=counts.values, names=counts.index,
        color_discrete_map={'Yes': '#2563EB', 'No': '#DC2626'},
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig = update_plotly_layout(fig)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Rain Rate by Month")
    monthly = dff.groupby('Month')['RainTomorrow'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    monthly.columns = ['Month', 'RainRate']
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly['MonthName'] = monthly['Month'].map(month_names)
    fig = px.bar(
        monthly, x='MonthName', y='RainRate',
        color='RainRate', color_continuous_scale='Blues',
        labels={'RainRate': 'Rain Rate (%)'}
    )
    fig = update_plotly_layout(fig)
    st.plotly_chart(fig, width='stretch')

# ── Numerical distributions ───────────────────────────────────
st.subheader("Feature Distributions by Rain")
num_cols = ['MinTemp','MaxTemp','Rainfall','Humidity9am',
            'Humidity3pm','WindGustSpeed','Pressure9am']
selected_col = st.selectbox("Select feature", num_cols)

fig = px.histogram(
    dff.dropna(subset=[selected_col]),
    x=selected_col, color='RainTomorrow',
    barmode='overlay', nbins=50, opacity=0.7,
    labels={'RainTomorrow': 'Rain Tomorrow'}
)
fig = update_plotly_layout(fig)
st.plotly_chart(fig, width='stretch')

# ── Correlation heatmap ───────────────────────────────────────
st.subheader("Correlation Heatmap")
num_df = dff[num_cols].corr().round(2)
fig = px.imshow(
    num_df, text_auto=True, color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1, aspect='auto'
)
fig = update_plotly_layout(fig)
fig.update_layout(height=500)
st.plotly_chart(fig, width='stretch')

# ── Rain by location ──────────────────────────────────────────
st.subheader("Rain Rate by Location")
loc_rain = dff.groupby('Location')['RainTomorrow'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
loc_rain.columns = ['Location', 'RainRate']
loc_rain = loc_rain.sort_values('RainRate', ascending=True)
fig = px.bar(
    loc_rain, x='RainRate', y='Location',
    orientation='h', color='RainRate',
    color_continuous_scale='Blues',
    labels={'RainRate': 'Rain Rate (%)'}
)
fig = update_plotly_layout(fig)
fig.update_layout(height=600)
st.plotly_chart(fig, width='stretch')
