# RainTomorrowML

Predict if it will rain tomorrow in Australia using machine learning with 8 different models (Bagging & Boosting).

## 📊 Overview

- **Dataset**: Weather in Australia (145,460 rows × 23 columns)
- **Target**: RainTomorrow (Binary Classification: Yes/No)
- **Best Model**: LightGBM (AUC: 0.8954)
- **All Models**: RandomForest, ExtraTrees, BaggingDT, GBM, AdaBoost, XGBoost, LightGBM, CatBoost

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python -m src.train
```
This trains all 8 models and saves results to `models/results.csv`.

### 3. Run Streamlit App
```bash
streamlit run app/main.py
```
Access at `http://localhost:8501`

Pages:
- 📊 EDA Dashboard - Explore the dataset
- 🔮 Single Prediction - Predict for one day
- 📂 Batch Prediction - Upload CSV for bulk predictions
- 🏆 Model Comparison - Compare all 8 models

### 4. Run FastAPI Server
```bash
uvicorn api.main:app --reload
```
Access at `http://localhost:8000/docs`

Endpoints:
- `GET /` - Health check
- `GET /models` - List available models
- `GET /results` - Training results
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch prediction (CSV upload)

## 📁 Project Structure

```
RainTomorrowML/
├── src/
│   ├── config.py           # Paths & configuration
│   ├── preprocessing.py    # Data pipeline
│   ├── train.py            # Model training
│   └── predict.py          # Prediction functions
├── api/
│   └── main.py             # FastAPI endpoints
├── app/
│   ├── main.py             # Streamlit main page
│   └── pages/
│       ├── 1_EDA_Dashboard.py
│       ├── 2_Single_Prediction.py
│       ├── 3_Batch_Prediction.py
│       └── 4_Model_Comparison.py
├── models/                 # Trained models & encoders
├── Data/
│   ├── Raw/                # weatherAUS.csv
│   └── processed/          # Processed data
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── requirements.txt
└── README.md
```

## 🎯 Model Performance

| Model | AUC | F1 | Accuracy |
|-------|-----|-----|----------|
| **LightGBM** | 0.8954 | 0.6581 | 81.93% |
| XGBoost | 0.8901 | 0.6517 | 81.36% |
| RandomForest | 0.8872 | 0.6566 | 84.12% |
| CatBoost | 0.8863 | 0.6409 | 80.55% |
| GradientBoosting | 0.8852 | 0.6260 | 86.00% |
| BaggingDT | 0.8774 | 0.6420 | 81.79% |
| ExtraTrees | 0.8732 | 0.6284 | 80.51% |
| AdaBoost | 0.8708 | 0.5925 | 85.13% |

## 🔧 Data Preprocessing

The preprocessing pipeline includes:
1. **Data Cleaning**: Handle missing values, drop null targets
2. **Imputation**: Group-aware median imputation for numerical features
3. **Outlier Capping**: Cap extreme values using IQR method
4. **Feature Engineering**: Create derived features (TempRange, HumidityDrop, etc.)
5. **Encoding**: Target encoding for Location, Ordinal encoding for Wind/Season

## 📝 API Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Location": "Sydney",
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "Evaporation": 5.0,
    "Sunshine": 7.0,
    "WindGustDir": "W",
    "WindGustSpeed": 44.0,
    "WindDir9am": "W",
    "WindDir3pm": "WNW",
    "WindSpeed9am": 20.0,
    "WindSpeed3pm": 24.0,
    "Humidity9am": 71.0,
    "Humidity3pm": 22.0,
    "Pressure9am": 1007.7,
    "Pressure3pm": 1007.1,
    "Cloud9am": 8.0,
    "Cloud3pm": 5.0,
    "Temp9am": 16.9,
    "Temp3pm": 21.8,
    "RainToday": "No",
    "Month": 12,
    "Year": 2024,
    "model_name": "LightGBM"
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@data.csv" \
  -F "model_name=LightGBM"
```

## 📊 Jupyter Notebook

Explore the data and models in the notebook:
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## 🛠️ Training Details

- **Train/Test Split**: 80/20
- **Random State**: 42 (reproducibility)
- **Class Balance**: Handled via class weights in models
- **Cross-Validation**: 5-fold KFold for Location encoding

## 📦 Dependencies

See `requirements.txt` for all packages. Key dependencies:
- pandas, numpy (data processing)
- scikit-learn (ML models)
- xgboost, lightgbm, catboost (gradient boosting)
- fastapi, uvicorn (API)
- streamlit (web app)
- plotly, seaborn (visualization)

## 💡 Usage Tips

1. **For Production**: Use LightGBM (best AUC, fastest)
2. **For Explainability**: Use RandomForest or GradientBoosting
3. **For Robustness**: Ensemble predictions from multiple models
4. **For Real-Time**: Use FastAPI with model caching

## 📄 License

Open source - feel free to use and modify.

## 🎓 Author

RainTomorrowML - Machine Learning Project
