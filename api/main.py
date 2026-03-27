from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import io

from src.predict import predict_single, predict_batch
from src.config import MODELS_DIR

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "RainTomorrowML API",
    description = "Predict rain in Australia using Bagging & Boosting models",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────
class WeatherInput(BaseModel):
    Location     : str            = Field(..., example="Sydney")
    MinTemp      : float          = Field(..., example=13.4)
    MaxTemp      : float          = Field(..., example=22.9)
    Rainfall     : float          = Field(..., example=0.6)
    Evaporation  : Optional[float]= Field(None, example=5.0)
    Sunshine     : Optional[float]= Field(None, example=7.0)
    WindGustDir  : str            = Field(..., example="W")
    WindGustSpeed: float          = Field(..., example=44.0)
    WindDir9am   : str            = Field(..., example="W")
    WindDir3pm   : str            = Field(..., example="WNW")
    WindSpeed9am : float          = Field(..., example=20.0)
    WindSpeed3pm : float          = Field(..., example=24.0)
    Humidity9am  : float          = Field(..., example=71.0)
    Humidity3pm  : float          = Field(..., example=22.0)
    Pressure9am  : Optional[float]= Field(None, example=1007.7)
    Pressure3pm  : Optional[float]= Field(None, example=1007.1)
    Cloud9am     : Optional[float]= Field(None, example=8.0)
    Cloud3pm     : Optional[float]= Field(None, example=5.0)
    Temp9am      : float          = Field(..., example=16.9)
    Temp3pm      : float          = Field(..., example=21.8)
    RainToday    : str            = Field(..., example="No")
    Month        : int            = Field(..., example=12)
    Year         : int            = Field(..., example=2024)
    model_name   : str            = Field("LightGBM", example="LightGBM")


class PredictionResponse(BaseModel):
    prediction  : int
    label       : str
    probability : float
    confidence  : float
    model_used  : str


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RainTomorrowML API is running 🌧️"}


@app.get("/models", tags=["Models"])
def list_models():
    """List all available trained models."""
    models = [p.stem for p in MODELS_DIR.glob("*.pkl")
              if p.stem not in ['location_map','oe_wind','oe_season','feature_cols','test_data']]
    return {"available_models": models}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(input_data: WeatherInput):
    """Single prediction from JSON input."""
    try:
        data       = input_data.dict()
        model_name = data.pop("model_name")

        # Convert RainToday to float
        data['RainToday'] = 1.0 if data['RainToday'] == 'Yes' else 0.0

        result = predict_single(data, model_name)
        result['model_used'] = model_name
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch_endpoint(
    file      : UploadFile = File(...),
    model_name: str        = "LightGBM"
):
    """Batch prediction from uploaded CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    try:
        contents = await file.read()
        df       = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # RainToday encode
        if 'RainToday' in df.columns:
            df['RainToday'] = df['RainToday'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0)

        result_df = predict_batch(df, model_name)

        return {
            "total_rows" : len(result_df),
            "rain_count" : int(result_df['Prediction'].sum()),
            "no_rain_count": int((result_df['Prediction'] == 0).sum()),
            "results"    : result_df[['Prediction','Probability','Label']].to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results", tags=["Models"])
def get_results():
    """Return training results for all models."""
    import pandas as pd
    results_path = MODELS_DIR / 'results.csv'
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Run training first.")
    df = pd.read_csv(results_path)
    return df.to_dict(orient='records')