"""Project paths and dataset schema."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dengue.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

DATE = "date"
CITY = "city"
WARD = "ward_id"
TARGET = "dengue_cases"

STATIC_COLUMNS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
]
WEATHER_COLUMNS = [
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
]
FEATURE_INPUT_COLUMNS = [DATE, CITY, WARD, *STATIC_COLUMNS, *WEATHER_COLUMNS]
REQUIRED_COLUMNS = [*FEATURE_INPUT_COLUMNS, TARGET]

HORIZONS = (2, 3, 4)

# Feature pipeline (docs/decisions.md, D1)
MODEL_TARGET = "target"
TARGET_DATE = "target_date"
MAX_LAG = 8
LAGS = tuple(range(MAX_LAG + 1))
RAIN_SUM_WINDOWS = (2, 4, 8)
MEAN_WINDOWS = (4, 8)
MEAN_COLUMNS = ("avg_temp_c", "relative_humidity_pct")
WEEKS_PER_YEAR = 52.1775

# Chronological split by target date (docs/decisions.md, D3)
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"
PURGE_WEEKS = 4

# Tiers and intervals (docs/decisions.md, D4 and D6)
TIER_NAMES = ("Low", "Medium", "High", "Critical")
TIER_QUANTILES = (0.60, 0.85, 0.95)
MIN_HIGH_CRITICAL_RECALL = 0.60
INTERVAL_QUANTILES = (0.05, 0.95)

SEED = 42
LGBM_PARAMS = {
    "objective": "poisson",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 100
