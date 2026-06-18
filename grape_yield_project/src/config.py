"""Configurable column names and paths for the grape yield project.

Column mappings are derived from the user's .xlsx files (Hebrew + English headers).
Adjust these dictionaries if your source files use different names.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
METRICS = OUTPUTS / "metrics"

# Default: parent wine folder where .xlsx files live (override via env or CLI).
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT.parent

VARIETY_FILTER = ["CS", "Cabernet Sauvignon", "Cabernet"]

# Year-based splits (configurable).
TRAIN_YEARS = [2020, 2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]

# Optional: restrict to specific vineyards (None = all).
VINEYARD_FILTER = None  # e.g. ["ENZ", "ELR", "KDZ"]

# --- ID / target columns (normalized output schema) ---
COL_VINE_ID = "vine_id"
COL_YEAR = "year"
COL_VINEYARD = "vineyard"
COL_BLOCK = "block"
COL_ROW = "row"
COL_VINE_POSITION = "vine_position"
COL_VARIETY = "variety"
COL_SHOOTS = "shoots_per_vine"
COL_CANOPY = "canopy_size"
COL_PRED_YIELD = "predicted_yield"
COL_ACTUAL_YIELD = "actual_yield"
COL_PREV_ACTUAL = "previous_year_actual_yield"
COL_PREV_PRED = "previous_year_predicted_yield"
COL_PREV_ERROR = "previous_year_yield_error"
COL_YIELD_ERROR = "yield_error"

ID_COLUMNS = [COL_VINE_ID, COL_YEAR, COL_VINEYARD, COL_BLOCK, COL_ROW, COL_VINE_POSITION, COL_VARIETY]
TARGET_COLUMN = COL_ACTUAL_YIELD

# Categorical columns for encoding (present columns are encoded automatically).
CATEGORICAL_COLUMNS = [
    COL_VINEYARD,
    COL_BLOCK,
    COL_ROW,
    "treatment",
    "soil_type",
    "winery",
    "root_stock",
    "clone",
    "weather_station",
    "general_direction",
    "color",
]

# Feature groups for PG-AN attention (matched against available columns).
FEATURE_GROUPS = {
    "vigor": [COL_SHOOTS, COL_CANOPY, "shoots_per_canopy_area", "cluster_count", "cluster_weight"],
    "historical_yield": [
        COL_PREV_ACTUAL,
        COL_PREV_PRED,
        COL_PREV_ERROR,
        COL_PRED_YIELD,
        COL_YIELD_ERROR,
    ],
    "weather": [
        "weather_temp_max_mean",
        "weather_temp_min_mean",
        "weather_rain_daily_sum",
        "weather_gdd_from_mar_max",
        "weather_vpd_max_mean",
        "weather_humidity_max_mean",
        "weather_solar_rad_mean",
        "weather_cold_hours_sum",
        "weather_penman_daily_mean",
    ],
    "spatial_management": [
        COL_VINEYARD,
        COL_BLOCK,
        COL_ROW,
        COL_VINE_POSITION,
        "size",
        "block_score",
        "rr_w",
        "distance_between_rows",
        "distance_between_vines",
        "vines_per_quarter_acre",
        "planting_year",
        "altitude_m",
        "degree",
        "brix",
        "ta",
        "ph",
    ],
}

# --- Source file patterns (glob) ---
POT_FILE_PATTERN = "*POT*.xlsx"
HARVEST_FILE_PATTERN = "*harvest*.xlsx"
CLUSTER_FILE_PATTERN = "*אשכול*.xlsx"
ARCHIVE_FILE_PATTERN = "*ארכיון*.xlsx"
WEATHER_FILE_PATTERN = "*מטאורולוגים*.xlsx"

POT_RENAME = {
    "HARVEST YEAR": COL_YEAR,
    "VINEYARD": COL_VINEYARD,
    "VARIETY": COL_VARIETY,
    "BLOCK": COL_BLOCK,
    "SIZE": "size",
    "BLOCK SCORE": "block_score",
    "RR/W": "rr_w",
    "YIELD ESTIMATION": COL_PRED_YIELD,
    "YIELD": COL_ACTUAL_YIELD,
}

HARVEST_RENAME = {
    "כרם": COL_VINEYARD,
    "קוד זן": COL_VARIETY,
    "צבע": "color",
    "חלקה": COL_BLOCK,
    "גודל מחושב": "size",
    "שנה": COL_YEAR,
    "תאריך": "harvest_date",
    "יבול ט/ד": "harvest_yield_t_per_d",
    "BX": "brix",
    "TA": "ta",
    "PH": "ph",
    "כמות ענבים": "grape_count",
    "BLOCK SCORE": "block_score_harvest",
    "דרוג יין": "wine_grade",
    "RR/W": "rr_w_harvest",
    "צבע ענבים": "grape_color",
}

CLUSTER_RENAME = {
    "VINEYARD": COL_VINEYARD,
    "VARIETY": COL_VARIETY,
    "COLOR": "color",
    "BLOCK": COL_BLOCK,
    "SIZE": "size_cluster",
    "HARVEST YEAR": COL_YEAR,
    "CLUSTER COUNT": "cluster_count",
    "CLUSTER WEIGHT": "cluster_weight",
}

ARCHIVE_RENAME = {
    "BLOCK": COL_BLOCK,
    "SIZE": "size_archive",
    "WINERY": "winery",
    "VARIETY": COL_VARIETY,
    "ROOT STOCK": "root_stock",
    "CLONE": "clone",
    "VINEYARD": COL_VINEYARD,
    "DISTANCE BETWEEN ROWS": "distance_between_rows",
    "DISTANCE BETWEEN VINES": "distance_between_vines",
    "VINES PER 1/4 ACRE": "vines_per_quarter_acre",
    "PLANTING YEAR": "planting_year",
    "WEATHER STATION": "weather_station",
    "DEGREE": "degree",
    "GENERAL DIRECTION": "general_direction",
    "ALTITUDE ABOVE SEA LEVEL M": "altitude_m",
    "COLOR": "color_archive",
}

WEATHER_RENAME = {
    "תאריך": "date",
    "קוד תחנה": "station",
    "טמפ מקס": "temp_max",
    "טמפ מינ": "temp_min",
    "לחות מקס": "humidity_max",
    "לחות מינ": "humidity_min",
    "VPD מקס": "vpd_max",
    "VPD מינ": "vpd_min",
    "מהירות רוח מקסימלית": "wind_speed_max",
    "קרינת השמש": "solar_rad",
    "טמפ קרקע מקס": "soil_temp_max",
    "טמפ קרקע מינ": "soil_temp_min",
    "רטיבות עלים": "leaf_wetness",
    "גשם יומי": "rain_daily",
    "גשם מצטבר": "rain_cumulative",
    "PENMAN_24": "penman_24",
    "PENMAN_DAILY": "penman_daily",
    "DD": "dd",
    "DEGDAY": "degday",
    "DEGDAY from 01/03": "gdd_from_mar",
    "שעות קור": "cold_hours",
    "שעות קור מצטברות": "cold_hours_cumulative",
    "קרינה יומית": "daily_radiation",
    "פנמן סטנדרט": "penman_standard",
}

WEATHER_AGG = {
    "temp_max": "mean",
    "temp_min": "mean",
    "humidity_max": "mean",
    "humidity_min": "mean",
    "vpd_max": "mean",
    "vpd_min": "mean",
    "rain_daily": "sum",
    "gdd_from_mar": "max",
    "degday": "max",
    "solar_rad": "mean",
    "cold_hours": "sum",
    "penman_daily": "mean",
    "wind_speed_max": "mean",
    "leaf_wetness": "mean",
}
