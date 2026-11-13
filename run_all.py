"""Run the full pipeline: SQL prep -> forecasts -> safety stock -> 2017 inventory replay -> charts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

import charts  # noqa: E402
import data  # noqa: E402
import forecast  # noqa: E402
import inventory  # noqa: E402
from config import CHART_DIR, OUT_DIR, SERVICE_GRID, TARGET_SERVICE, TEST_START  # noqa: E402


def main():
    OUT_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    weekly, feats = data.build()
    fc, importance = forecast.run(feats)

    test = fc[fc.week_start >= TEST_START]
    calib = fc[fc.week_start < TEST_START]
    acc = forecast.accuracy(test).assign(window="2017 test")
    acc_calib = forecast.accuracy(calib).assign(window="2016 H2 calibration")
    accuracy = pd.concat([acc_calib, acc])
    print("\nForecast accuracy\n", accuracy.round(4).to_string(index=False))

    sim = inventory.evaluate(fc, SERVICE_GRID)
    print("\nInventory replay, 2017\n", sim.round(4).to_string(index=False))

    # Same service, different stock: inventory each method needs for 95% stockout-free weeks
    need = {m: inventory.inventory_needed(sim, m, TARGET_SERVICE) for m in forecast.METHODS.values()}
    a = acc.set_index("method")
    ma, sn, gbm = "4-week moving average", "Same week last year", "LightGBM"
    summary = {
        "series": int(fc.groupby(["store", "item"]).ngroups),
        "wape_moving_avg_2017": a.loc[ma, "wape"],
        "wape_last_year_2017": a.loc[sn, "wape"],
        "wape_lightgbm_2017": a.loc[gbm, "wape"],
        "in_stock_weeks_target": TARGET_SERVICE,
        "inventory_needed_moving_avg": need[ma],
        "inventory_needed_last_year": need[sn],
        "inventory_needed_lightgbm": need[gbm],
        "reduction_vs_moving_avg": 1 - need[gbm] / need[ma],
        "reduction_vs_last_year": 1 - need[gbm] / need[sn],
    }
    print("\nSummary\n", pd.Series(summary).round(4).to_string())

    accuracy.to_csv(OUT_DIR / "forecast_accuracy.csv", index=False)
    sim.to_csv(OUT_DIR / "service_vs_inventory.csv", index=False)
    pd.Series(summary).to_csv(OUT_DIR / "summary.csv", header=["value"])
    importance.rename("gain_share").to_csv(OUT_DIR / "feature_importance.csv")
    inventory.policy_table(fc, "fc_lightgbm", TARGET_SERVICE).to_csv(
        OUT_DIR / "sku_location_policy.csv", index=False)
    fc[["store", "item", "week_start", "demand", *forecast.METHODS]].round(
        {c: 1 for c in forecast.METHODS}).to_csv(
        OUT_DIR / "forecasts_2016h2_2017.csv", index=False)

    charts.make_all(weekly, fc, accuracy, sim, importance, summary)
    print("\nOutputs written to outputs and charts")


if __name__ == "__main__":
    main()
