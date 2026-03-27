import joblib
import pandas as pd
import numpy as np
from src.config import MODELS_DIR
from src.preprocessing import (
    impute, cap_outliers, engineer_features, encode
)


def load_model(model_name: str):
    path = MODELS_DIR / f'{model_name}.pkl'
    if not path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found. Run train.py first.")
    return joblib.load(path)


def load_feature_cols() -> list:
    return joblib.load(MODELS_DIR / 'feature_cols.pkl')


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the same pipeline as training — but fit=False (use saved encoders).
    Input df must have the raw columns (same as weatherAUS.csv minus Date & target).
    """
    df = df.copy()

    # Convert RainToday to numeric if it's string
    if 'RainToday' in df.columns:
        if df['RainToday'].dtype == 'object':
            df['RainToday'] = (df['RainToday'] == 'Yes').astype(float)

    # Month & Year from input (user provides Month directly in single prediction)
    if 'Month' not in df.columns:
        df['Month'] = 6   # fallback
    if 'Year' not in df.columns:
        df['Year'] = 2024

    df = impute(df)
    df = cap_outliers(df)
    df = engineer_features(df)
    df = encode(df, fit=False)

    # Align columns to training order
    feature_cols = load_feature_cols()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    return df


def predict_single(input_dict: dict, model_name: str) -> dict:
    df  = pd.DataFrame([input_dict])
    X   = preprocess_input(df)
    mdl = load_model(model_name)

    proba     = mdl.predict_proba(X)[0][1]
    pred      = int(proba >= 0.5)
    label     = "Rain 🌧️" if pred == 1 else "No Rain ☀️"

    return {
        "prediction" : pred,
        "label"      : label,
        "probability": round(float(proba), 4),
        "confidence" : round(float(max(proba, 1 - proba)) * 100, 2),
    }


def predict_batch(df_raw: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df_raw = df_raw.copy()
    X      = preprocess_input(df_raw)
    mdl    = load_model(model_name)

    probas       = mdl.predict_proba(X)[:, 1]
    preds        = (probas >= 0.5).astype(int)

    df_raw['Prediction']  = preds
    df_raw['Probability'] = probas.round(4)
    df_raw['Label']       = df_raw['Prediction'].map({1: 'Rain 🌧️', 0: 'No Rain ☀️'})

    return df_raw