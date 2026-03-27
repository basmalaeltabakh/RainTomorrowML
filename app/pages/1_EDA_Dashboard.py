import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import DATA_RAW

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 EDA Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_RAW)
    df['Date']  = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.header("Filters")
locations = st.sidebar.multiselect(
    "Locations", df['Location'].unique().tolist(),
    default=df['Location'].unique().tolist()[:5]
)
years = st.sidebar.slider(
    "Year range",
    int(df['Year'].min()), int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

mask = (df['Location'].isin(locations)) & (df['Year'].between(*years))
dff  = df[mask]

# ── KPIs ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",     f"{len(dff):,}")
c2.metric("Rain Days",         f"{(dff['RainTomorrow']=='Yes').sum():,}")
c3.metric("Rain Rate",         f"{(dff['RainTomorrow']=='Yes').mean()*100:.1f}%")
c4.metric("Locations Selected",f"{dff['Location'].nunique()}")

st.divider()

# ── Target distribution ───────────────────────────────────────
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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

# ── Numerical distributions ───────────────────────────────────
st.subheader("Feature Distributions by Rain")
num_cols = ['MinTemp','MaxTemp','Rainfall','Humidity9am',
            'Humidity3pm','WindGustSpeed','Pressure9am']
selected_col = st.selectbox("Select feature", num_cols)

fig = px.histogram(
    dff.dropna(subset=[selected_col]),
    x=selected_col, color='RainTomorrow',
    barmode='overlay', nbins=50, opacity=0.7,
    color_discrete_map={'Yes': '#2563EB', 'No': '#DC2626'},
    labels={'RainTomorrow': 'Rain Tomorrow'}
)
st.plotly_chart(fig, use_container_width=True)

# ── Correlation heatmap ───────────────────────────────────────
st.subheader("Correlation Heatmap")
num_df = dff[num_cols].corr().round(2)
fig = px.imshow(
    num_df, text_auto=True, color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1, aspect='auto'
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

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
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)