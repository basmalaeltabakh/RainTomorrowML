import streamlit as st

st.set_page_config(
    page_title = "RainTomorrowML",
    page_icon  = "🌧️",
    layout     = "wide",
)

st.title("🌧️ RainTomorrowML")
st.subheader("Predicting Rain in Australia using Bagging & Boosting")

st.markdown("""
### Welcome!
Use the sidebar to navigate between pages:

| Page | Description |
|------|-------------|
| 📊 EDA Dashboard | Explore the dataset visually |
| 🔮 Single Prediction | Predict rain for one day |
| 📂 Batch Prediction | Upload CSV and predict in bulk |
| 🏆 Model Comparison | Compare all 8 trained models |

---
**Dataset:** Rain in Australia (145,460 rows · 23 features)  
**Models:** Random Forest, Extra Trees, Bagging DT, GBM, AdaBoost, XGBoost, LightGBM, CatBoost
""")

st.info("⬅️ Select a page from the sidebar to get started.", icon="👈")