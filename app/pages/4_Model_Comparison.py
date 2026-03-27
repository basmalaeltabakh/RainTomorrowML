import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from src.config import MODELS_DIR

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison")

# ── Load results ──────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_csv(MODELS_DIR / 'results.csv')

@st.cache_resource
def load_test_data():
    return joblib.load(MODELS_DIR / 'test_data.pkl')

@st.cache_resource
def load_model(name):
    return joblib.load(MODELS_DIR / f'{name}.pkl')

results_df         = load_results()
X_test, y_test     = load_test_data()

# ── Leaderboard ───────────────────────────────────────────────
st.subheader("🥇 Leaderboard")
styled = results_df.style.background_gradient(
    subset=['accuracy','f1','roc_auc'], cmap='Blues'
).format({'accuracy':'{:.4f}','f1':'{:.4f}','roc_auc':'{:.4f}'})
st.dataframe(styled, use_container_width=True)

st.divider()

# ── Metrics bar chart ─────────────────────────────────────────
st.subheader("📊 Metrics Comparison")
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
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── ROC Curves ────────────────────────────────────────────────
st.subheader("📈 ROC Curves — All Models")
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
st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# ── Confusion Matrix ──────────────────────────────────────────
st.subheader("🔲 Confusion Matrix")
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
fig_cm.update_layout(height=400, width=400)
st.plotly_chart(fig_cm)