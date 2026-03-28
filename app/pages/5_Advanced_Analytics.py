import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Safe plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

from sklearn.metrics import roc_curve, confusion_matrix, classification_report
from src.config import MODELS_DIR, DATA_RAW

try:
    from src.predict import predict_batch
    PREDICT_AVAILABLE = True
except ImportError as e:
    PREDICT_AVAILABLE = False
    st.error(f"❌ Prediction module not available: {e}")

st.set_page_config(page_title="Advanced Analytics", page_icon="📈", layout="wide")

from app.ui_utils import apply_custom_css, update_plotly_layout
apply_custom_css()

# ────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">📈 Advanced Analytics</div>
    <div class="header-subtitle">Deep dive into model performance and predictions</div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────
# Load Data
# ────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    results_df = pd.read_csv(MODELS_DIR / 'results.csv')
    with open(MODELS_DIR / 'test_data.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)
    return results_df, X_test, y_test

results_df, X_test, y_test = load_data()

# ────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Advanced Settings")

model_name = st.sidebar.selectbox(
    "Select Model for Analysis",
    [p.stem for p in MODELS_DIR.glob("*.pkl")
     if p.stem not in ['location_map','oe_wind','oe_season','feature_cols','test_data']],
    index=0
)

analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Model Metrics", "Predictions Heatmap", "Feature Importance", "Error Analysis"]
)

# ────────────────────────────────────────────────────────────────────────
# TAB 1: Model Metrics Analysis
# ────────────────────────────────────────────────────────────────────────
if analysis_type == "Model Metrics":
    st.markdown('<div class="section-header">Model Metrics Analysis</div>', unsafe_allow_html=True)

    # Load model
    with open(MODELS_DIR / f'{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Display metrics in cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10b981;">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{accuracy:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3b82f6;">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{precision:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{recall:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #8b5cf6;">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">{f1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ec4899;">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">{auc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ROC Curve
    st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.4f})',
        line=dict(color='#3b82f6', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#94a3b8', width=2, dash='dash'),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    fig_roc = update_plotly_layout(fig_roc)
    st.plotly_chart(fig_roc, use_container_width=True, key='roc_curve_chart')

    # Confusion Matrix
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Rain', 'Predicted: Rain'],
        y=['Actual: No Rain', 'Actual: Rain'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=16, color='white'),
        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
    ))
    fig_cm = update_plotly_layout(fig_cm)
    st.plotly_chart(fig_cm, use_container_width=True, key='confusion_matrix_chart')

# ────────────────────────────────────────────────────────────────────────
# TAB 2: Predictions Heatmap
# ────────────────────────────────────────────────────────────────────────
elif analysis_type == "Predictions Heatmap":
    st.markdown('<div class="section-header">Prediction Probability Distribution</div>', unsafe_allow_html=True)

    with open(MODELS_DIR / f'{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    y_proba = model.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig_hist = px.histogram(
            x=y_proba,
            nbins=50,
            title='Prediction Probability Distribution',
            labels={'x': 'Prediction Probability', 'y': 'Frequency'},
            color_discrete_sequence=['#3b82f6'],
            height=400,
        )
        fig_hist = update_plotly_layout(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True, key='pred_hist_chart')

    with col2:
        # Statistics
        st.markdown("""
        <div class="section-header" style="margin-top: 0; margin-bottom: 20px;">
            Prediction Statistics
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Mean Probability", f"{y_proba.mean():.4f}")
            st.metric("Median Probability", f"{np.median(y_proba):.4f}")
        with col_b:
            st.metric("Std Deviation", f"{y_proba.std():.4f}")
            st.metric("Range", f"{y_proba.min():.4f} - {y_proba.max():.4f}")

# ────────────────────────────────────────────────────────────────────────
# TAB 3: Feature Importance
# ────────────────────────────────────────────────────────────────────────
elif analysis_type == "Feature Importance":
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

    with open(MODELS_DIR / f'{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / 'feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(15)

        fig = px.bar(
            feature_importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Most Important Features',
            color='importance',
            color_continuous_scale='Blues',
            height=500,
        )
        fig = update_plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True, key='feature_importance_chart')
    else:
        st.info("Feature importance not available for this model type")

# ────────────────────────────────────────────────────────────────────────
# TAB 4: Error Analysis
# ────────────────────────────────────────────────────────────────────────
elif analysis_type == "Error Analysis":
    st.markdown('<div class="section-header">Error Analysis</div>', unsafe_allow_html=True)

    with open(MODELS_DIR / f'{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate errors
    correct = y_pred == y_test
    incorrect = y_pred != y_test

    false_positives = (y_pred == 1) & (y_test == 0)
    false_negatives = (y_pred == 0) & (y_test == 1)
    true_positives = (y_pred == 1) & (y_test == 1)
    true_negatives = (y_pred == 0) & (y_test == 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("True Positives", int(true_positives.sum()), f"{true_positives.sum()/len(y_test)*100:.1f}%")
    with col2:
        st.metric("True Negatives", int(true_negatives.sum()), f"{true_negatives.sum()/len(y_test)*100:.1f}%")
    with col3:
        st.metric("False Positives", int(false_positives.sum()), f"{false_positives.sum()/len(y_test)*100:.1f}%")
    with col4:
        st.metric("False Negatives", int(false_negatives.sum()), f"{false_negatives.sum()/len(y_test)*100:.1f}%")

    # Error types pie chart
    st.markdown('<div class="section-header">Error Distribution</div>', unsafe_allow_html=True)

    error_data = {
        'Type': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
        'Count': [
            int(true_positives.sum()),
            int(true_negatives.sum()),
            int(false_positives.sum()),
            int(false_negatives.sum())
        ]
    }
    error_df = pd.DataFrame(error_data)

    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    fig_pie = px.pie(
        error_df,
        values='Count',
        names='Type',
        color_discrete_sequence=colors,
        title='Prediction Result Distribution',
        height=450,
        hover_data={'Count': ':,'}
    )
    fig_pie = update_plotly_layout(fig_pie)
    st.plotly_chart(fig_pie, use_container_width=True, key='error_pie_chart')

# ────────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align: center; color: #94a3b8; font-size: 12px; padding: 20px 0;">
    <p>Advanced Analytics Dashboard • Model: """ + model_name + """</p>
</div>
""", unsafe_allow_html=True)
