from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT_DIR / "Data" / "Raw" / "weatherAUS.csv"
DATA_PROCESSED= ROOT_DIR / "Data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"

# ── Model settings ────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
TARGET        = "RainTomorrow"

# ── Feature lists ─────────────────────────────────────────────
WIND_ORDER = [
    'N','NNE','NE','ENE','E','ESE','SE','SSE',
    'S','SSW','SW','WSW','W','WNW','NW','NNW','Unknown'
]

SEASON_ORDER = [['Summer','Autumn','Winter','Spring']]

OUTLIER_COLS = [
    'Rainfall','Evaporation','WindGustSpeed',
    'WindSpeed9am','WindSpeed3pm'
]

CAT_WIND_COLS = ['WindGustDir','WindDir9am','WindDir3pm']

ENGINEERED_FEATURES = [
    'TempRange','TempChange','HumidityDrop',
    'PressureDrop','WindSpeedDiff'
]

# ── Best model name (update after training) ───────────────────
BEST_MODEL_NAME = "LightGBM"