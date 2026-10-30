"""Load raw sales into DuckDB, run quality checks, build weekly demand and features in SQL."""
import duckdb
import pandas as pd

from config import RAW_FILE, SQL_DIR


def _statements(path):
    lines = [ln for ln in path.read_text().splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def build():
    con = duckdb.connect()
    con.execute(f"CREATE TABLE raw_sales AS SELECT * FROM read_csv_auto('{RAW_FILE.as_posix()}')")

    print("Data quality checks")
    for stmt in _statements(SQL_DIR / "01_quality_checks.sql"):
        print("  ", con.execute(stmt).df().to_dict("records")[0])

    for name in ["02_weekly_demand.sql", "03_features.sql"]:
        for stmt in _statements(SQL_DIR / name):
            con.execute(stmt)

    weekly = con.execute("SELECT * FROM weekly_demand").df()
    feats = con.execute("SELECT * FROM features").df()
    for df in (weekly, feats):
        df["week_start"] = pd.to_datetime(df["week_start"])
    print(f"   weekly rows: {len(weekly):,} | series: {weekly.groupby(['store','item']).ngroups} "
          f"| weeks: {weekly.week_start.min().date()} to {weekly.week_start.max().date()}")
    return weekly, feats
