import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Safe plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from src.config import MODELS_DIR

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")

from app.ui_utils import apply_custom_css, update_plotly_layout, style_dataframe
apply_custom_css()

st.markdown("""
<div class="header-container">
    <div class="header-title">🏆 Model Comparison</div>
    <div class="header-subtitle">Compare performance and evaluation metrics</div>
</div>
""", unsafe_allow_html=True)

# ── Load results ──────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_csv(MODELS_DIR / 'results.csv')

@st.cache_resource
def load_test_data():
    with open(MODELS_DIR / 'test_data.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(name):
    with open(MODELS_DIR / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)

results_df         = load_results()
X_test, y_test     = load_test_data()

# ── Leaderboard ───────────────────────────────────────────────
st.markdown('<div class="section-header">🥇 Leaderboard</div>', unsafe_allow_html=True)
styled = style_dataframe(results_df).format({'accuracy':'{:.4f}','f1':'{:.4f}','roc_auc':'{:.4f}'})
st.markdown(styled.to_html(), unsafe_allow_html=True)

st.divider()

# ── Metrics bar chart ─────────────────────────────────────────
st.markdown('<div class="section-header">📊 Metrics Comparison</div>', unsafe_allow_html=True)
metric = st.radio("Select Metric", ['roc_auc','f1','accuracy'], horizontal=True)

fig = px.bar(
    results_df.sort_values(metric, ascending=True),
    x=metric, y='model', orientation='h',
    color='type',
    color_discrete_map={'Bagging':'#2563EB','Boosting':'#7C3AED'},
    text=results_df.sort_values(metric)[metric].round(4),
    labels={metric: metric.upper(), 'model': 'Model'}
)
fig.update_traces(textposition='outside')
fig = update_plotly_layout(fig)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── ROC Curves ────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 ROC Curves — All Models</div>', unsafe_allow_html=True)
fig_roc = go.Figure()
colors  = px.colors.qualitative.Plotly

for i, row in results_df.iterrows():
    mdl       = load_model(row['model'])
    y_proba   = mdl.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, name=f"{row['model']} (AUC={row['roc_auc']:.3f})",
        line=dict(color=colors[i % len(colors)], width=2)
    ))

fig_roc.add_trace(go.Scatter(
    x=[0,1], y=[0,1], name='Random',
    line=dict(color='gray', dash='dash'), showlegend=True
))
fig_roc.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=500
)
fig_roc = update_plotly_layout(fig_roc)
st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# ── Confusion Matrix ──────────────────────────────────────────
st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
selected = st.selectbox("Select Model", results_df['model'].tolist())

mdl    = load_model(selected)
y_pred = mdl.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)

fig_cm = px.imshow(
    cm, text_auto=True, color_continuous_scale='Blues',
    x=['No Rain','Rain'], y=['No Rain','Rain'],
    labels=dict(x='Predicted', y='Actual'),
    title=f'Confusion Matrix — {selected}'
)
fig_cm = update_plotly_layout(fig_cm)
fig_cm.update_layout(height=450, width=450)
st.plotly_chart(fig_cm)