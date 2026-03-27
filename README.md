# 🌧️ RainTomorrow-ML
**Predicting Tomorrow's Rain Across Australia with Advanced Machine Learning**

<div align="center">

  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/ML%20Models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-orange.svg" alt="Models">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg" alt="Kaggle">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">

  <br><br>

  <a href="YOUR_STREAMLIT_LINK">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>

</div>

---

##  Dashboard Demo

https://github.com/user-attachments/assets/4d6fe080-03d1-432b-82f5-c362ec85d502

---

##  What is RainTomorrow-ML?

RainTomorrow-ML is an **end-to-end machine learning system** that predicts whether it will rain tomorrow across **49 locations in Australia**. Built on 145,000+ real weather records spanning 10 years, the project covers the full ML lifecycle — from raw data to a deployed interactive application.

-  **8 ensemble models** compared (Bagging + Boosting)
-  **Production-ready REST API** via FastAPI
-  **Interactive 4-page Streamlit dashboard**
-  **Full EDA + feature engineering pipeline**

---

## 📸 Screenshots

| Home | EDA Dashboard |
|------|--------------|
| <img width="1327" alt="Home" src="https://github.com/user-attachments/assets/3ddc4dbb-0892-4e7f-a9da-7102444ad748" /> | <img width="1185" alt="EDA Dashboard" src="https://github.com/user-attachments/assets/61d90399-059b-4904-8eb7-7639219dae0c" /> |

| Single Prediction | Model Comparison |
|------------------|-----------------|
| <img width="1142" alt="Single Prediction" src="https://github.com/user-attachments/assets/59a1ca93-6606-49c9-bc04-df2999e4e984" /> | <img width="1413" alt="Model Comparison" src="https://github.com/user-attachments/assets/9146d9b2-8a8b-48ae-9111-4e9626b10c34" /> |


---

## 🏆 Model Performance

| Rank | Model | Type | AUC | F1-Score | Accuracy |
|------|-------|------|-----|----------|----------|
| 🥇 | **LightGBM** | Boosting | **0.8954** | **0.6581** | **81.93%** |
| 🥈 | XGBoost | Boosting | 0.8901 | 0.6517 | 81.36% |
| 🥉 | Random Forest | Bagging | 0.8872 | 0.6566 | 84.12% |
| 4 | CatBoost | Boosting | 0.8863 | 0.6409 | 80.55% |
| 5 | Gradient Boosting | Boosting | 0.8852 | 0.6260 | 86.00% |
| 6 | Extra Trees | Bagging | — | — | — |
| 7 | AdaBoost | Boosting | — | — | — |
| 8 | Bagging (DT) | Bagging | — | — | — |

**LightGBM** leads with the best AUC and F1-Score balance.
> All models were evaluated on a held-out 20% test set with stratified splitting.

---

##  ML Pipeline

### 1️⃣ Data Cleaning
- Group-aware imputation (median/mode **per location**, not global)
- IQR winsorization (1%–99%) on skewed features: `Rainfall`, `WindGustSpeed`, `Evaporation`
- Dropped rows with missing target only (~3,200 rows)

### 2️⃣ Feature Engineering
5 new features derived from existing columns:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `TempRange` | MaxTemp − MinTemp | Daily heat spread |
| `TempChange` | Temp3pm − Temp9am | Temperature trend |
| `HumidityDrop` | Humidity9am − Humidity3pm | Moisture loss |
| `PressureDrop` | Pressure9am − Pressure3pm | Pressure change |
| `WindSpeedDiff` | WindSpeed3pm − WindSpeed9am | Wind trend |

### 3️⃣ Encoding Strategy

| Feature | Method | Reason |
|---------|--------|--------|
| `Location` (49 cats) | Target Encoding (OOF) | High cardinality — avoids leakage |
| `WindGustDir`, `WindDir9am`, `WindDir3pm` | Ordinal Encoding | Directional order |
| `Season` | Ordinal Encoding | Cyclical mapping |
| `RainToday` | Binary (0/1) | Already categorical binary |

>  **No scaling applied** — all models are tree-based and scale-invariant.

---

##  API Endpoints

The FastAPI backend runs at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/models` | List available trained models |
| `GET` | `/results` | Training metrics for all models |
| `POST` | `/predict` | Single prediction (JSON input) |
| `POST` | `/predict/batch` | Batch prediction (CSV upload) |

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Location": "Sydney",
           "MinTemp": 13.4,
           "MaxTemp": 22.9,
           "Humidity9am": 71,
           "Humidity3pm": 22,
           "WindGustDir": "W",
           "WindGustSpeed": 44,
           "WindDir9am": "W",
           "WindDir3pm": "WNW",
           "WindSpeed9am": 20,
           "WindSpeed3pm": 24,
           "Rainfall": 0.6,
           "Temp9am": 16.9,
           "Temp3pm": 21.8,
           "RainToday": "No",
           "Pressure9am": 1007.7,
           "Pressure3pm": 1007.1,
           "Month": 12,
           "Year": 2024,
           "model_name": "LightGBM"
         }'
```

**Example response:**
```json
{
  "prediction": 0,
  "label": "No Rain ☀️",
  "probability": 0.1342,
  "confidence": 86.58,
  "model_used": "LightGBM"
}
```

---

##  Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/RainTomorrowML.git
cd RainTomorrowML
pip install -r requirements.txt
```

### 2. Get the Dataset
Download from Kaggle and place it at `data/raw/weatherAUS.csv`:

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Rain%20in%20Australia-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

### 3. Train Models
```bash
python -m src.train
```
This saves all 8 trained models + encoders to `models/`.

### 4. Run the App
```bash
# Streamlit dashboard
streamlit run app/main.py

# FastAPI (in a separate terminal)
uvicorn api.main:app --reload --port 8000
```

---

##  Project Structure

```
RainTomorrowML/
│
├── 📂 data/
│   ├── raw/                  # weatherAUS.csv (download from Kaggle)
│   └── processed/            # Intermediate processed files
│
├── 📂 notebooks/
│   └── 01_eda_and_modeling.ipynb
│
├── 📂 src/
│   ├── config.py             # Paths, constants, feature lists
│   ├── preprocessing.py      # Full cleaning + encoding pipeline
│   ├── train.py              # Train all 8 models + save artifacts
│   └── predict.py            # Inference for single & batch
│
├── 📂 api/
│   └── main.py               # FastAPI — 5 endpoints
│
├── 📂 app/
│   ├── main.py               # Streamlit home page
│   └── pages/
│       ├── 1_EDA_Dashboard.py
│       ├── 2_Single_Prediction.py
│       ├── 3_Batch_Prediction.py
│       └── 4_Model_Comparison.py
│
├── 📂 models/                # Saved .pkl files (git-ignored)
├── 📂 screenshots/           # App screenshots for README
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| ML Models | scikit-learn, XGBoost, LightGBM, CatBoost |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn, SHAP |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Serialization | Joblib |

---

##  Dataset

- **Source:** [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Rain%20in%20Australia-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Size:** 145,460 rows × 23 columns
- **Period:** 2008 – 2017
- **Locations:** 49 weather stations across Australia
- **Target:** `RainTomorrow` (binary: Yes/No → 1/0)
- **Class balance:** ~78% No Rain / ~22% Rain

---
