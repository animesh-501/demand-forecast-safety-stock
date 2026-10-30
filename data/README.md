# Data

**Source:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) (Kaggle, public).
Daily unit sales for 50 items across 10 stores, 2013-01-01 to 2017-12-31 (913,000 rows).

The raw file isn't committed here because of Kaggle's data terms. To run the project:

```bash
pip install kaggle
kaggle competitions download -c demand-forecasting-kernels-only -f train.csv -p data/raw
unzip -o data/raw/train.csv.zip -d data/raw   # if it downloads zipped
```

Expected file: `data/raw/train.csv` with columns `date, store, item, sales`.
