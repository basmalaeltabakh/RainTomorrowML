import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from src.config import MODELS_DIR

# ────────────────────────────────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RainTomorrowML Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────
# Custom CSS Styling
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1e40af;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
    }

    * {
        margin: 0;
        padding: 0;
    }

    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
    }

    /* Main container */
    .main {
        background-color: #f8fafc;
    }

    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1);
    }

    .header-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .header-subtitle {
        font-size: 16px;
        opacity: 0.95;
        font-weight: 500;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #2563eb;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 32px;
        font-weight: 700;
    }

    .metric-suffix {
        color: #94a3b8;
        font-size: 16px;
        font-weight: 500;
    }

    /* Section Headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 12px;
    }

    /* Card Container */
    .card-container {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Badge Styling */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }

    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
    }

    .badge-info {
        background-color: #dbeafe;
        color: #0c4a6e;
    }

    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">🌧️ RainTomorrowML</div>
    <div class="header-subtitle">AI-Powered Rain Prediction for Australia • Real-time Weather Analytics</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Load Results
# ────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_csv(MODELS_DIR / 'results.csv')

results_df = load_results()
best_model = results_df.iloc[0]

# ────────────────────────────────────────────────────────────────────────
# Main KPI Section
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Performance Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best Model</div>
        <div class="metric-value">{best_model['model']}</div>
        <div class="metric-suffix">{best_model['type']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ROC-AUC Score</div>
        <div class="metric-value">{best_model['roc_auc']:.4f}</div>
        <div class="metric-suffix">Excellent Performance ✓</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">F1-Score</div>
        <div class="metric-value">{best_model['f1']:.4f}</div>
        <div class="metric-suffix">vs {results_df.iloc[-1]['f1']:.4f} worst</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{best_model['accuracy']:.2%}</div>
        <div class="metric-suffix">Overall Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# All Models Comparison
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 All Models Leaderboard</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Top models visualization
    top_models = results_df.nlargest(5, 'roc_auc')

    fig = px.bar(
        top_models,
        x='roc_auc',
        y='model',
        orientation='h',
        color='roc_auc',
        color_continuous_scale=['#fbbf24', '#60a5fa', '#3b82f6', '#1e40af', '#1e3a8a'],
        text='roc_auc',
        title='Top 5 Models by ROC-AUC',
        height=320,
        hover_data={'accuracy': ':.2%', 'f1': ':.4f'},
    )
    fig.update_traces(textposition='outside', texttemplate='%{text:.4f}')
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, key='top_models_chart')

with col2:
    st.markdown("""
    <div class="card-container">
        <div class="metric-label">Dataset Stats</div>
        <ul style="list-style: none; padding: 12px 0;">
            <li style="margin: 8px 0; font-size: 14px; color: #334155;"><strong>Total Samples:</strong> 145,460</li>
            <li style="margin: 8px 0; font-size: 14px; color: #334155;"><strong>Features:</strong> 23</li>
            <li style="margin: 8px 0; font-size: 14px; color: #334155;"><strong>Rain Days:</strong> 31,877 (21.9%)</li>
            <li style="margin: 8px 0; font-size: 14px; color: #334155;"><strong>No Rain Days:</strong> 113,583 (78.1%)</li>
            <li style="margin: 8px 0; font-size: 14px; color: #334155;"><strong>Locations:</strong> 49</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Detailed Metrics Table
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Detailed Metrics Table</div>', unsafe_allow_html=True)

# Format and display results
results_display = results_df.copy()
results_display['accuracy'] = results_display['accuracy'].apply(lambda x: f"{x:.2%}")
results_display['f1'] = results_display['f1'].apply(lambda x: f"{x:.4f}")
results_display['roc_auc'] = results_display['roc_auc'].apply(lambda x: f"{x:.4f}")

st.dataframe(
    results_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        'model': st.column_config.TextColumn('Model Name', width=150),
        'type': st.column_config.TextColumn('Type', width=120),
        'accuracy': st.column_config.TextColumn('Accuracy', width=120),
        'f1': st.column_config.TextColumn('F1-Score', width=120),
        'roc_auc': st.column_config.TextColumn('ROC-AUC', width=120),
    }
)

# ────────────────────────────────────────────────────────────────────────
# Model Distribution
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Model Type Distribution</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Pie chart
    type_counts = results_df['type'].value_counts()
    colors = ['#3b82f6', '#8b5cf6']  # Blue and Purple
    fig_pie = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.3,
        marker=dict(colors=colors),
        textinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    fig_pie.update_layout(
        title='Models by Type',
        height=400,
        showlegend=True,
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
    )
    st.plotly_chart(fig_pie, use_container_width=True, key='model_type_chart')

with col2:
    # Metrics comparison
    fig_metrics = px.box(
        results_df,
        y='roc_auc',
        x='type',
        color='type',
        color_discrete_map={'Bagging': '#3b82f6', 'Boosting': '#8b5cf6'},
        title='ROC-AUC Distribution by Type',
        height=400,
    )
    fig_metrics.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        hovermode='closest',
    )
    st.plotly_chart(fig_metrics, use_container_width=True, key='auc_distribution_chart')

# ────────────────────────────────────────────────────────────────────────
# Quick Navigation
# ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🚀 Quick Navigation</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('📊 EDA Dashboard', use_container_width=True, key='nav_eda'):
        st.switch_page("pages/1_EDA_Dashboard.py")

with col2:
    if st.button('🔮 Single Prediction', use_container_width=True, key='nav_single'):
        st.switch_page("pages/2_Single_Prediction.py")

with col3:
    if st.button('📂 Batch Predictions', use_container_width=True, key='nav_batch'):
        st.switch_page("pages/3_Batch_Prediction.py")

with col4:
    if st.button('🏆 Model Comparison', use_container_width=True, key='nav_comparison'):
        st.switch_page("pages/4_Model_Comparison.py")

# ────────────────────────────────────────────────────────────────────────
# Footer Information
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align: center; color: #94a3b8; font-size: 12px; padding: 20px 0;">
    <p><strong>RainTomorrowML v1.0</strong> • ML-powered weather prediction for Australia</p>
    <p>Trained on 145,460 weather records • 8 ensemble models • Real-time predictions</p>
    <p style="margin-top: 12px; font-size: 11px;">
        <span class="badge badge-info">8 Models</span>
        <span class="badge badge-success">89.5% Best AUC</span>
        <span class="badge badge-info">49 Locations</span>
    </p>
</div>
""", unsafe_allow_html=True)
