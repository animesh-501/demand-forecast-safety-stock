"""Static charts for the README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

from config import CHART_DIR, TARGET_SERVICE, TEST_START  # noqa: E402
from forecast import METHODS  # noqa: E402

COLORS = {"4-week moving average": "#9aa0a6", "Same week last year": "#e8a33d", "LightGBM": "#1f6feb"}
INK, MUTED = "#1f2328", "#656d76"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": "#d0d7de", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#eaeef2", "grid.linewidth": 0.8, "legend.frameon": False,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, bbox_inches="tight")
    plt.close(fig)


def demand_history(weekly):
    total = weekly.groupby("week_start").demand.sum()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(total.index, total.values / 1000, color=INK, lw=1.4)
    ax.set_title("Total weekly demand, 500 SKU-locations")
    ax.set_ylabel("Units (thousands)")
    ax.text(0, -0.22, "Strong summer peak every year and steady year-over-year growth.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _save(fig, "01_demand_history.png")


def forecast_vs_actual(fc):
    test = fc[fc.week_start >= TEST_START].groupby("week_start")[["demand", *METHODS]].sum() / 1000
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(test.index, test.demand, color=INK, lw=2.2, label="Actual")
    for col, name in METHODS.items():
        ax.plot(test.index, test[col], color=COLORS[name], lw=1.5, label=name,
                ls="-" if name == "LightGBM" else "--")
    ax.set_title("2017: actual demand vs forecasts made 3 weeks ahead")
    ax.set_ylabel("Units (thousands)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    _save(fig, "02_forecast_vs_actual_2017.png")


def wape_bars(accuracy):
    a = accuracy[accuracy.window == "2017 test"].set_index("method").loc[list(METHODS.values())]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    bars = ax.barh(a.index, a.wape * 100, color=[COLORS[m] for m in a.index])
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in a.wape], padding=4, color=INK)
    ax.set_title("Forecast error on 2017 (WAPE, lower is better)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, a.wape.max() * 100 * 1.2)
    _save(fig, "03_forecast_error.png")


def service_vs_inventory(sim, summary):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name in METHODS.values():
        r = sim[sim.method == name].sort_values("avg_inventory_units")
        ax.plot(r.avg_inventory_units / 1000, r.weeks_without_stockout * 100, marker="o", ms=4,
                color=COLORS[name], lw=2 if name == "LightGBM" else 1.5, label=name)
    key = {"4-week moving average": "inventory_needed_moving_avg",
           "Same week last year": "inventory_needed_last_year", "LightGBM": "inventory_needed_lightgbm"}
    for name, k in key.items():
        ax.scatter(summary[k] / 1000, TARGET_SERVICE * 100, s=90, facecolor="white",
                   edgecolor=COLORS[name], lw=2, zorder=5)
    ax.axhline(TARGET_SERVICE * 100, color=MUTED, lw=0.8, ls=":")
    ax.set_title("In-stock weeks you get for the inventory you hold (2017 replay)")
    ax.set_xlabel("Average units on hand across all SKU-locations (thousands)")
    ax.set_ylabel("Weeks without a stockout")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(60, 100.5)
    ax.legend(loc="lower right")
    ax.text(0, -0.2, f"Circles = stock each method needs to stay in stock {TARGET_SERVICE:.0%} of weeks. "
            "Up and to the left is better.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    _save(fig, "04_service_vs_inventory.png")


LABELS = {
    "lag_52": "Same week last year", "lag_53": "Week before, last year", "lag_51": "Week after, last year",
    "lag_3": "Demand 3 weeks ago", "lag_4": "Demand 4 weeks ago", "lag_5": "Demand 5 weeks ago",
    "lag_6": "Demand 6 weeks ago", "ma_4": "4-week average", "ma_8": "8-week average",
    "ma_13": "13-week average", "ma_4_last_year": "4-week average, last year",
    "week_of_year": "Week of year", "month": "Month", "item": "Item", "store": "Store",
}


def importance_bars(importance):
    top = importance.head(8)[::-1].rename(LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(top.index, top.values * 100, color=COLORS["LightGBM"])
    ax.set_title("What the model leans on (share of split gain)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", visible=False)
    _save(fig, "05_feature_importance.png")


def make_all(weekly, fc, accuracy, sim, importance, summary):
    demand_history(weekly)
    forecast_vs_actual(fc)
    wape_bars(accuracy)
    service_vs_inventory(sim, summary)
    importance_bars(importance)
