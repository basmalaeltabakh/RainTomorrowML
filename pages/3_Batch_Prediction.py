import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict import predict_batch
from src.config import MODELS_DIR

st.set_page_config(page_title="Batch Prediction", page_icon="📂", layout="wide")

from app.ui_utils import apply_custom_css, update_plotly_layout, style_dataframe
apply_custom_css()

st.markdown("""
<div class="header-container">
    <div class="header-title">📂 Batch Prediction</div>
    <div class="header-subtitle">Run predictions on multiple rows of data via CSV upload</div>
</div>
""", unsafe_allow_html=True)

available_models = [
    p.stem for p in MODELS_DIR.glob("*.pkl")
    if p.stem not in ['location_map','oe_wind','oe_season','feature_cols','test_data']
]
model_name = st.sidebar.selectbox("Select Model", available_models)

st.info("Upload a CSV file with the same columns as weatherAUS.csv (without RainTomorrow).")

uploaded = st.file_uploader("Upload CSV", type=['csv'])

if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.markdown('<div class="section-header">Preview — Uploaded Data</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.write(f"Shape: {df_raw.shape}")

    if st.button("🚀 Run Batch Prediction", use_container_width=True):
        with st.spinner("Running predictions..."):

            # Encode RainToday if present as string
            if 'RainToday' in df_raw.columns:
                df_raw['RainToday'] = df_raw['RainToday'].map(
                    {'Yes': 1.0, 'No': 0.0}
                ).fillna(0.0)

            # Add Month/Year if Date column exists
            if 'Date' in df_raw.columns:
                df_raw['Date']  = pd.to_datetime(df_raw['Date'])
                df_raw['Month'] = df_raw['Date'].dt.month
                df_raw['Year']  = df_raw['Date'].dt.year
                df_raw = df_raw.drop(columns=['Date'])

            result_df = predict_batch(df_raw, model_name)

        st.success("Done!")
        st.divider()

        # ── KPIs
        c1, c2, c3, c4 = st.columns(4)
        total     = len(result_df)
        rain_cnt  = int(result_df['Prediction'].sum())
        norain    = total - rain_cnt
        rain_rate = rain_cnt / total * 100

        c1.metric("Total Rows",   f"{total:,}")
        c2.metric("Rain 🌧️",      f"{rain_cnt:,}")
        c3.metric("No Rain ☀️",   f"{norain:,}")
        c4.metric("Rain Rate",    f"{rain_rate:.1f}%")

        # ── Charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                result_df, names='Label',
                color_discrete_map={'Rain 🌧️':'#2563EB','No Rain ☀️':'#DC2626'},
                title="Prediction Distribution", hole=0.4
            )
            fig = update_plotly_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                result_df, x='Probability', nbins=50,
                color='Label',
                color_discrete_map={'Rain 🌧️':'#2563EB','No Rain ☀️':'#DC2626'},
                title="Probability Distribution"
            )
            fig = update_plotly_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── Table
        st.markdown('<div class="section-header">Results Table</div>', unsafe_allow_html=True)
        display_df = result_df[['Prediction','Probability','Label']].copy()
        styled_table = style_dataframe(display_df).to_html()
        st.markdown(styled_table, unsafe_allow_html=True)

        # ── Download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv, file_name='predictions.csv',
            mime='text/csv', use_container_width=True
        )