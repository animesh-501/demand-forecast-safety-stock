"""Safety stock from the measured 3-week forecast error vs the textbook weekly error x sqrt(3).

Works from outputs/forecasts_2016h2_2017.csv, so it runs without the raw data:
    python protection_window.py

Writes:
  outputs/protection_window_error.csv   how much bigger the real 3-week error is, and why
  outputs/protection_window_replay.csv  2017 replay at every service target, both sizings
  charts/06_protection_window.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

import charts  # noqa: E402
import inventory  # noqa: E402
from config import (CALIB_START, CHART_DIR, OUT_DIR, PROTECTION_WEEKS,  # noqa: E402
                    SERVICE_GRID, TARGET_SERVICE, TEST_START)
from forecast import METHODS  # noqa: E402

KEYS = ["store", "item"]


def error_profile(fc):
    """Per method: textbook vs measured 3-week error, and the two reasons they differ."""
    calib = fc[(fc.week_start >= CALIB_START) & (fc.week_start < TEST_START)]
    calib = calib.sort_values(KEYS + ["week_start"])
    rows = []
    for col, name in METHODS.items():
        sqrt_rule = inventory.protection_sigma(fc, col, "sqrt")
        window = inventory.window_error_rmse(fc, col)
        e = calib.assign(e=calib[col] - calib.demand).groupby(KEYS)["e"]
        w = calib.assign(e=calib[col] - calib.demand).assign(
            w=lambda x: x.groupby(KEYS)["e"].transform(lambda s: s.rolling(PROTECTION_WEEKS).sum())
        ).dropna(subset=["w"]).groupby(KEYS)["w"]
        rows.append({
            "method": name,
            "sqrt_rule_error_units": sqrt_rule.sum(),
            "measured_3wk_error_units": window.sum(),
            "measured_vs_sqrt_rule": window.sum() / sqrt_rule.sum(),
            "week_to_week_error_correlation": e.apply(lambda s: s.autocorr(1)).mean(),
            "bias_share_of_3wk_error": 1 - (w.var(ddof=0)).sum() / (w.apply(lambda s: (s ** 2).mean())).sum(),
            "windows_per_series": int(w.size().min()),
        })
    return pd.DataFrame(rows)


def main():
    fc = pd.read_csv(OUT_DIR / "forecasts_2016h2_2017.csv", parse_dates=["week_start"])

    profile = error_profile(fc)
    print("Calibration error, all 500 SKU-locations summed\n", profile.round(3).to_string(index=False))

    rep = pd.concat([inventory.evaluate(fc, SERVICE_GRID, sizing=s).assign(sizing=s)
                     for s in ("sqrt", "window")], ignore_index=True)
    at_target = rep[rep.target_service == TARGET_SERVICE].pivot(
        index="method", columns="sizing",
        values=["weeks_without_stockout", "fill_rate", "avg_inventory_units"]).loc[list(METHODS.values())]
    print(f"\n2017 replay at the {TARGET_SERVICE:.0%} target setting\n", at_target.round(4).to_string())

    need = pd.DataFrame({s: {m: inventory.inventory_needed(rep[rep.sizing == s], m, TARGET_SERVICE)
                             for m in METHODS.values()} for s in ("sqrt", "window")})
    print(f"\nInventory needed for {TARGET_SERVICE:.0%} stockout-free weeks (read off each curve)\n",
          need.round(0).to_string())

    profile.to_csv(OUT_DIR / "protection_window_error.csv", index=False)
    rep.to_csv(OUT_DIR / "protection_window_replay.csv", index=False)
    CHART_DIR.mkdir(exist_ok=True)
    charts.protection_window(rep)
    print("\nOutputs written to outputs and charts")


if __name__ == "__main__":
    main()
