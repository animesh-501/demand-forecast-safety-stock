"""Turn forecasts into safety stock and order-up-to levels, then replay 2017 to see what happens.

Policy (periodic review, order-up-to):
  every week, order enough to bring inventory position up to
      S = forecast for the next 3 weeks + safety stock
  safety stock = z * RMSE_weekly * sqrt(3)
  where RMSE_weekly is that SKU-location's forecast error on the calibration half-year.

Simulation: lost sales (no backorders), 2-week lead time, starts in steady state.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import CALIB_START, PROTECTION_WEEKS, TEST_START
from forecast import METHODS

KEYS = ["store", "item"]


def error_rmse(df, col):
    calib = df[(df.week_start >= CALIB_START) & (df.week_start < TEST_START)]
    return calib.assign(e2=(calib[col] - calib.demand) ** 2).groupby(KEYS)["e2"].mean().pow(0.5)


def _matrices(df, col):
    test = df[df.week_start >= TEST_START].sort_values(KEYS + ["week_start"])
    d = test.pivot_table(index=KEYS, columns="week_start", values="demand").to_numpy()
    f = test.pivot_table(index=KEYS, columns="week_start", values=col).to_numpy()
    # Orders placed in the final 3 weeks arrive after the window, so padding forecasts
    # past the last week only affects orders that never land inside 2017.
    f = np.hstack([f, np.repeat(f[:, -1:], PROTECTION_WEEKS, axis=1)])
    return d, f


def simulate(d, f, ss):
    n, T = d.shape
    L = PROTECTION_WEEKS
    arrive = np.zeros((n, T + L))
    arrive[:, 1], arrive[:, 2] = f[:, 1], f[:, 2]   # orders already in transit at the start
    on_hand = ss + f[:, 0]

    sold = np.zeros((n, T))
    inv = np.zeros((n, T))
    for t in range(T):
        on_hand = on_hand + (arrive[:, t] if t > 0 else 0)
        start = on_hand.copy()
        sold[:, t] = np.minimum(on_hand, d[:, t])
        on_hand = on_hand - sold[:, t]
        inv[:, t] = (start + on_hand) / 2                       # average on hand during the week
        position = on_hand + arrive[:, t + 1:t + L].sum(axis=1)
        target = f[:, t + 1:t + 1 + L].sum(axis=1) + ss
        arrive[:, t + L] += np.maximum(0, target - position)
    return sold, inv


def evaluate(df, service_levels):
    rows = []
    for col, name in METHODS.items():
        d, f = _matrices(df, col)
        sigma = error_rmse(df, col).sort_index().to_numpy()
        for sl in service_levels:
            ss = norm.ppf(sl) * sigma * np.sqrt(PROTECTION_WEEKS)
            sold, inv = simulate(d, f, ss)
            rows.append({
                "method": name,
                "target_service": sl,
                "fill_rate": sold.sum() / d.sum(),
                "weeks_without_stockout": (sold >= d - 1e-9).mean(),
                "avg_inventory_units": inv.sum(axis=0).mean(),   # all SKU-locations, avg week
                "weeks_of_supply": inv.mean() / d.mean(),
                "safety_stock_units": ss.sum(),
            })
    return pd.DataFrame(rows)


def inventory_needed(results, method, in_stock_target):
    """Interpolate the average inventory a method needs to hit a share of stockout-free weeks."""
    r = results[results.method == method].sort_values("weeks_without_stockout")
    x = r.weeks_without_stockout
    if in_stock_target < x.min() or in_stock_target > x.max():
        return np.nan
    return float(np.interp(in_stock_target, x, r.avg_inventory_units))


def policy_table(df, col, service):
    """Per SKU-location parameters a planner would load: weekly forecast, error, safety stock."""
    sigma = error_rmse(df, col).rename("weekly_forecast_rmse")
    test = df[df.week_start >= TEST_START]
    avg_fc = test.groupby(KEYS)[col].mean().rename("avg_weekly_forecast")
    out = pd.concat([avg_fc, sigma], axis=1).reset_index()
    out["safety_stock_units"] = norm.ppf(service) * out.weekly_forecast_rmse * np.sqrt(PROTECTION_WEEKS)
    out["safety_stock_weeks"] = out.safety_stock_units / out.avg_weekly_forecast
    out["avg_order_up_to_level"] = out.avg_weekly_forecast * PROTECTION_WEEKS + out.safety_stock_units
    return out.round(2)
