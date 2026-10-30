"""Paths and planning assumptions for the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "train.csv"
SQL_DIR = ROOT / "sql"
OUT_DIR = ROOT / "outputs"
CHART_DIR = ROOT / "charts"

# Backtest windows (weeks start on Monday)
CALIB_START = "2016-07-04"  # forecast errors here set safety stock
TEST_START = "2017-01-02"   # inventory is simulated on 2017, never used for fitting

# Replenishment assumptions
LEAD_TIME_WEEKS = 2         # order placed at end of week t arrives before week t+3
REVIEW_WEEKS = 1            # order every week
PROTECTION_WEEKS = LEAD_TIME_WEEKS + REVIEW_WEEKS

TARGET_SERVICE = 0.95       # cycle service target: share of weeks with no stockout
SERVICE_GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]

FEATURES = [
    "store", "item",
    "lag_3", "lag_4", "lag_5", "lag_6",
    "ma_4", "ma_8", "ma_13",
    "lag_51", "lag_52", "lag_53", "ma_4_last_year",
    "week_of_year", "month",
]
SEED = 42
