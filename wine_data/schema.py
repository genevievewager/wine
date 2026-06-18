"""Column names and feature-group definitions for the CS yield dataset."""

VARIETY_CS = "CS"

# Vine / canopy static and block-level attributes (from agricultural archive).
VINE_COLUMNS = [
    "block",
    "vineyard",
    "variety",
    "size",
    "winery",
    "root_stock",
    "clone",
    "distance_between_rows",
    "distance_between_vines",
    "vines_per_quarter_acre",
    "planting_year",
    "weather_station",
    "degree",
    "general_direction",
    "altitude_m",
    "color",
]

# Cluster / canopy measurements (optional per block-year).
CLUSTER_COLUMNS = [
    "cluster_count",
    "cluster_weight",
]

# Weather columns (daily raw; aggregated with ``weather_`` prefix in modeling table).
WEATHER_RAW_COLUMNS = [
    "date",
    "station",
    "temp_max",
    "temp_min",
    "humidity_max",
    "humidity_min",
    "vpd_max",
    "vpd_min",
    "wind_speed_max",
    "solar_rad",
    "soil_temp_max",
    "soil_temp_min",
    "leaf_wetness",
    "rain_daily",
    "rain_cumulative",
    "penman_24",
    "penman_daily",
    "dd",
    "degday",
    "gdd_from_mar",
    "cold_hours",
    "cold_hours_cumulative",
    "daily_radiation",
    "penman_standard",
]

WEATHER_AGG_COLUMNS = [
    "temp_max_mean",
    "temp_min_mean",
    "vpd_max_mean",
    "rain_daily_sum",
    "gdd_from_mar_max",
    "solar_rad_mean",
    "cold_hours_sum",
]

PREVIOUS_YIELD_COLUMNS = ["yield_prev_year"]

# Predicted yield — kept separate; use as baseline, not a default model feature.
PREDICTED_YIELD_COLUMNS = ["yield_estimation"]

TARGET_COLUMN = "yield_actual"

ID_COLUMNS = ["harvest_year", "block", "vineyard", "variety"]

HARVEST_RENAME = {
    "כרם": "vineyard",
    "קוד זן": "variety",
    "צבע": "color",
    "חלקה": "block",
    "גודל מחושב": "size",
    "שנה": "harvest_year",
    "תאריך": "date",
    "יבול ט/ד": "harvest_yield_t_per_d",
    "BX": "brix",
    "TA": "ta",
    "PH": "ph",
    "כמות ענבים": "grape_count",
    "BLOCK SCORE": "block_score",
    "דרוג יין": "wine_grade",
    "RR/W": "rr_w",
    "צבע ענבים": "grape_color",
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

POT_RENAME = {
    "HARVEST YEAR": "harvest_year",
    "VINEYARD": "vineyard",
    "VARIETY": "variety",
    "BLOCK": "block",
    "SIZE": "size",
    "BLOCK SCORE": "block_score",
    "RR/W": "rr_w",
    "YIELD ESTIMATION": "yield_estimation",
    "YIELD": "yield_actual",
}

VINE_ARCHIVE_RENAME = {
    "BLOCK": "block",
    "SIZE": "size",
    "WINERY": "winery",
    "VARIETY": "variety",
    "ROOT STOCK": "root_stock",
    "CLONE": "clone",
    "VINEYARD": "vineyard",
    "DISTANCE BETWEEN ROWS": "distance_between_rows",
    "DISTANCE BETWEEN VINES": "distance_between_vines",
    "VINES PER 1/4 ACRE": "vines_per_quarter_acre",
    "PLANTING YEAR": "planting_year",
    "WEATHER STATION": "weather_station",
    "DEGREE": "degree",
    "GENERAL DIRECTION": "general_direction",
    "ALTITUDE ABOVE SEA LEVEL M": "altitude_m",
    "COLOR": "color",
}

CLUSTER_RENAME = {
    "VINEYARD": "vineyard",
    "VARIETY": "variety",
    "COLOR": "color",
    "BLOCK": "block",
    "SIZE": "size",
    "HARVEST YEAR": "harvest_year",
    "CLUSTER COUNT": "cluster_count",
    "CLUSTER WEIGHT": "cluster_weight",
}
