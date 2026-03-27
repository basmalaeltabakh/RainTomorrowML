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

from app.ui_utils import apply_custom_css, update_plotly_layout, style_dataframe
apply_custom_css()
# ────────────────────────────────────────────────────────────────────────
# Removed Custom CSS block - Now using app.ui_utils
# ────────────────────────────────────────────────────────────────────────

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
    fig = update_plotly_layout(fig)
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
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
results_display = results_display[['model', 'type', 'accuracy', 'f1', 'roc_auc']]
results_display.columns = ['Model Name', 'Type', 'Accuracy', 'F1-Score', 'ROC-AUC']
results_display['Accuracy'] = results_display['Accuracy'].apply(lambda x: f"{x:.2%}")
results_display['F1-Score'] = results_display['F1-Score'].apply(lambda x: f"{x:.4f}")
results_display['ROC-AUC'] = results_display['ROC-AUC'].apply(lambda x: f"{x:.4f}")
results_display = results_display.set_index('Model Name')

styled_table = style_dataframe(results_display).to_html()
st.markdown(styled_table, unsafe_allow_html=True)

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
    fig_pie = update_plotly_layout(fig_pie)
    fig_pie.update_layout(
        title='Models by Type',
        height=400,
        showlegend=True,
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
    fig_metrics = update_plotly_layout(fig_metrics)
    fig_metrics.update_layout(
        showlegend=False,
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
