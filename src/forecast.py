"""Baseline forecasts vs a LightGBM model, all using demand from 3+ weeks back only.

Two fits keep the backtest honest:
  * model A is trained on data before CALIB_START and forecasts the calibration half-year.
    Its errors set safety stock.
  * model B is retrained on everything before TEST_START and forecasts 2017.
Model settings were fixed up front; 2017 is never used for fitting or tuning.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CALIB_START, FEATURES, SEED, TEST_START

PARAMS = dict(
    objective="poisson", learning_rate=0.05, num_leaves=63, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1,
)
N_ROUNDS = 600


def _fit_predict(train, score):
    cat = ["store", "item"]
    ds = lgb.Dataset(train[FEATURES], train["demand"], categorical_feature=cat)
    model = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
    return model, model.predict(score[FEATURES])


def run(feats):
    df = feats.dropna(subset=FEATURES).copy()   # first ~58 weeks lack a full year of history
    calib_start, test_start = pd.Timestamp(CALIB_START), pd.Timestamp(TEST_START)

    # Baselines a planner would build in a spreadsheet, with the same 3-week information lag
    df["fc_moving_avg"] = df["ma_4"]           # average of the last 4 known weeks
    df["fc_seasonal_naive"] = df["lag_52"]     # same week last year

    df["fc_lightgbm"] = np.nan
    calib = df.week_start >= calib_start
    _, df.loc[calib & (df.week_start < test_start), "fc_lightgbm"] = _fit_predict(
        df[df.week_start < calib_start], df[calib & (df.week_start < test_start)])
    model, df.loc[df.week_start >= test_start, "fc_lightgbm"] = _fit_predict(
        df[df.week_start < test_start], df[df.week_start >= test_start])

    importance = (pd.Series(model.feature_importance("gain"), index=FEATURES)
                  .sort_values(ascending=False) / model.feature_importance("gain").sum())
    return df[df.week_start >= calib_start].reset_index(drop=True), importance


METHODS = {
    "fc_moving_avg": "4-week moving average",
    "fc_seasonal_naive": "Same week last year",
    "fc_lightgbm": "LightGBM",
}


def accuracy(df):
    """WAPE and bias per method: WAPE = sum|error| / sum(actual)."""
    rows = []
    for col, name in METHODS.items():
        err = df[col] - df["demand"]
        rows.append({
            "method": name,
            "wape": err.abs().sum() / df["demand"].sum(),
            "bias": err.sum() / df["demand"].sum(),
            "rmse": float(np.sqrt((err ** 2).mean())),
        })
    return pd.DataFrame(rows)
