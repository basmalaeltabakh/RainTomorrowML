import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from src.config import (
    TARGET, RANDOM_STATE, WIND_ORDER,
    SEASON_ORDER, OUTLIER_COLS, CAT_WIND_COLS,
    MODELS_DIR
)

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # ── Date features
    df['Date']  = pd.to_datetime(df['Date'])
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.drop(columns=['Date'])

    # ── Encode binary columns
    df[TARGET]       = (df[TARGET]       == 'Yes').astype(float)
    df['RainToday']  = (df['RainToday']  == 'Yes').astype(float)

    # ── Drop rows with null target
    df = df.dropna(subset=[TARGET])

    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include='number').columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET]

    # Group-aware numerical imputation
    for col in num_cols:
        df[col] = df.groupby('Location')[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

    # Categorical wind imputation
    for col in CAT_WIND_COLS:
        df[col] = df.groupby('Location')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
        )
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTLIER_COLS:
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        IQR    = Q3 - Q1
        upper  = Q3 + 1.5 * IQR
        lower  = max(0, Q1 - 1.5 * IQR)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['TempRange']     = df['MaxTemp']      - df['MinTemp']
    df['TempChange']    = df['Temp3pm']      - df['Temp9am']
    df['HumidityDrop']  = df['Humidity9am']  - df['Humidity3pm']
    df['PressureDrop']  = df['Pressure9am']  - df['Pressure3pm']
    df['WindSpeedDiff'] = df['WindSpeed3pm'] - df['WindSpeed9am']
    df['Season']        = df['Month'].map({
        12:'Summer',1:'Summer', 2:'Summer',
        3:'Autumn', 4:'Autumn', 5:'Autumn',
        6:'Winter', 7:'Winter', 8:'Winter',
        9:'Spring', 10:'Spring',11:'Spring'
    })
    return df


def encode(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    fit=True  → during training (saves encoders)
    fit=False → during inference (loads saved encoders)
    """
    if fit:
        # ── Target encoding for Location (OOF)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        df['Location_encoded'] = 0.0
        for tr_idx, val_idx in kf.split(df):
            loc_mean = df.iloc[tr_idx].groupby('Location')[TARGET].mean()
            df.loc[df.index[val_idx], 'Location_encoded'] = \
                df.iloc[val_idx]['Location'].map(loc_mean)
        df['Location_encoded'] = df['Location_encoded'].fillna(df[TARGET].mean())

        # Save location mean map
        loc_map = df.groupby('Location')[TARGET].mean().to_dict()
        joblib.dump(loc_map, MODELS_DIR / 'location_map.pkl')

        # ── Ordinal encoding — wind + season
        oe_wind   = OrdinalEncoder(
            categories=[WIND_ORDER]*3,
            handle_unknown='use_encoded_value', unknown_value=-1
        )
        oe_season = OrdinalEncoder(categories=SEASON_ORDER)

        df[['WindGustDir_enc','WindDir9am_enc','WindDir3pm_enc']] = \
            oe_wind.fit_transform(df[CAT_WIND_COLS])
        df['Season_enc'] = oe_season.fit_transform(df[['Season']])

        joblib.dump(oe_wind,   MODELS_DIR / 'oe_wind.pkl')
        joblib.dump(oe_season, MODELS_DIR / 'oe_season.pkl')

    else:
        loc_map   = joblib.load(MODELS_DIR / 'location_map.pkl')
        oe_wind   = joblib.load(MODELS_DIR / 'oe_wind.pkl')
        oe_season = joblib.load(MODELS_DIR / 'oe_season.pkl')

        global_mean = sum(loc_map.values()) / len(loc_map)
        df['Location_encoded'] = df['Location'].map(loc_map).fillna(global_mean)

        df[['WindGustDir_enc','WindDir9am_enc','WindDir3pm_enc']] = \
            oe_wind.transform(df[CAT_WIND_COLS])
        df['Season_enc'] = oe_season.transform(df[['Season']])

    # ── Drop raw categorical columns
    df = df.drop(columns=['Location'] + CAT_WIND_COLS + ['Season'])

    return df


def full_pipeline(filepath: str, fit: bool = True) -> tuple:
    df = load_and_clean(filepath)
    df = impute(df)
    df = cap_outliers(df)
    df = engineer_features(df)
    df = encode(df, fit=fit)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Save feature columns order
    if fit:
        joblib.dump(X.columns.tolist(), MODELS_DIR / 'feature_cols.pkl')

    return X, y