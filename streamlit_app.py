import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Minimal imports first
import streamlit as st
import pandas as pd

# Import plotly AFTER streamlit to avoid conflicts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotly not available: {e}")
    PLOTLY_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────
# Page Configuration (MUST BE FIRST)
# ────────────────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="RainTomorrowML Dashboard",
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception as e:
    print(f"Error setting page config: {e}")

# ────────────────────────────────────────────────────────────────────────
# Import Config and Utils (with error handling)
# ────────────────────────────────────────────────────────────────────────
try:
    from src.config import MODELS_DIR
except Exception as e:
    st.error(f"❌ Could not import config: {e}")
    st.stop()

# Try to import UI utils, but continue without them if it fails
ui_functions_available = False
try:
    from app.ui_utils import apply_custom_css, update_plotly_layout, style_dataframe
    apply_custom_css()
    ui_functions_available = True
except Exception as e:
    print(f"Warning: Could not import UI utilities: {e}")
    # Define fallback functions
    def update_plotly_layout(fig):
        return fig
    
    def style_dataframe(df):
        return df

# ────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #6366f1; margin: 0;">🌧️ RainTomorrowML</h1>
    <p style="color: #64748b; font-size: 16px; margin: 10px 0 0 0;">
        AI-Powered Rain Prediction for Australia • Real-time Weather Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Load Results with Error Handling
# ────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    try:
        results_path = MODELS_DIR / 'results.csv'
        if not results_path.exists():
            return None
        df = pd.read_csv(results_path)
        return df if not df.empty else None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

# Try to load results
results_df = load_results()

# If no results, show placeholder
if results_df is None or results_df.empty:
    st.error("⚠️ Could not load model results.")
    st.info(f"📁 Models directory: {MODELS_DIR}")
    st.stop()

best_model = results_df.iloc[0]

# ────────────────────────────────────────────────────────────────────────
# Main KPI Section
# ────────────────────────────────────────────────────────────────────────
st.markdown('### 📈 Performance Overview', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Best Model", best_model['model'], f"{best_model['type']}")

with col2:
    st.metric("ROC-AUC Score", f"{best_model['roc_auc']:.4f}", "Excellent ✓")

with col3:
    st.metric("F1-Score", f"{best_model['f1']:.4f}", f"vs {results_df.iloc[-1]['f1']:.4f}")

with col4:
    st.metric("Accuracy", f"{best_model['accuracy']:.2%}", "Overall")

# ────────────────────────────────────────────────────────────────────────
# All Models Comparison (with Plotly if available)
# ────────────────────────────────────────────────────────────────────────
st.markdown('### 🏆 All Models Leaderboard', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    if PLOTLY_AVAILABLE:
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
        )
        fig.update_traces(textposition='outside', texttemplate='%{text:.4f}')
        fig = update_plotly_layout(fig)
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width='stretch', key='top_models_chart')
    else:
        top_models = results_df.nlargest(5, 'roc_auc')
        st.write("**Top 5 Models by ROC-AUC:**")
        st.dataframe(top_models[['model', 'roc_auc', 'accuracy', 'f1']], width='stretch')

with col2:
    st.markdown("""
    **Dataset Stats**
    - Total Samples: 145,460
    - Features: 23
    - Rain Days: 31,877 (21.9%)
    - No Rain Days: 113,583 (78.1%)
    - Locations: 49
    """)

# ────────────────────────────────────────────────────────────────────────
# Detailed Metrics Table
# ────────────────────────────────────────────────────────────────────────
st.markdown('### 📊 Detailed Metrics Table', unsafe_allow_html=True)

# Format and display results
results_display = results_df.copy()
results_display = results_display[['model', 'type', 'accuracy', 'f1', 'roc_auc']]
results_display.columns = ['Model', 'Type', 'Accuracy', 'F1-Score', 'ROC-AUC']
results_display['Accuracy'] = results_display['Accuracy'].apply(lambda x: f"{x:.2%}")
results_display['F1-Score'] = results_display['F1-Score'].apply(lambda x: f"{x:.4f}")
results_display['ROC-AUC'] = results_display['ROC-AUC'].apply(lambda x: f"{x:.4f}")

st.dataframe(results_display, width='stretch')

# ────────────────────────────────────────────────────────────────────────
# Model Distribution (Fallback if Plotly not available)
# ────────────────────────────────────────────────────────────────────────
st.markdown('### 📈 Model Type Distribution', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if PLOTLY_AVAILABLE:
        # Pie chart
        type_counts = results_df['type'].value_counts()
        colors = ['#3b82f6', '#8b5cf6']
        fig_pie = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.3,
            marker=dict(colors=colors),
        )])
        fig_pie = update_plotly_layout(fig_pie)
        fig_pie.update_layout(title='Models by Type', height=350)
        st.plotly_chart(fig_pie, width='stretch', key='model_type_chart')
    else:
        type_counts = results_df['type'].value_counts()
        st.write("**Models by Type:**")
        st.bar_chart(type_counts)

with col2:
    if PLOTLY_AVAILABLE:
        # Metrics comparison
        fig_metrics = px.box(
            results_df,
            y='roc_auc',
            x='type',
            color='type',
            color_discrete_map={'Bagging': '#3b82f6', 'Boosting': '#8b5cf6'},
            title='ROC-AUC by Type',
            height=350,
        )
        fig_metrics = update_plotly_layout(fig_metrics)
        fig_metrics.update_layout(showlegend=False)
        st.plotly_chart(fig_metrics, width='stretch', key='auc_distribution_chart')
    else:
        st.write("**ROC-AUC by Type:**")
        type_auc = results_df.groupby('type')['roc_auc'].apply(list)
        st.write(type_auc)

# ────────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 12px;">
    <p><strong>RainTomorrowML v1.0</strong> • ML-powered weather prediction</p>
    <p>Trained on 145,460 records • 8 ensemble models • Real-time predictions</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Quick Navigation
# ────────────────────────────────────────────────────────────────────────
st.markdown('### 🚀 Quick Navigation', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button('📊 EDA Dashboard', width='stretch', key='nav_eda'):
        st.switch_page("pages/1_EDA_Dashboard.py")

with col2:
    if st.button('🔮 Single Prediction', width='stretch', key='nav_single'):
        st.switch_page("pages/2_Single_Prediction.py")

with col3:
    if st.button('📂 Batch Predictions', width='stretch', key='nav_batch'):
        st.switch_page("pages/3_Batch_Prediction.py")

with col4:
    if st.button('🏆 Model Comparison', width='stretch', key='nav_comparison'):
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
