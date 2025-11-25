from functools import reduce
from pathlib import Path
import pandas as pd

from modules.data_services.data_models import Pair


def load_single_ticker(ticker: str, start: str, end: str, interval: str, base_dir: Path) -> pd.DataFrame:
    """Load data for a single asset and return as a DataFrame."""
    ticker_dir = base_dir / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ticker_dir}")

    files = list(ticker_dir.glob(f"*_{interval}.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV file with interval '{interval}' found in {ticker_dir}")

    df = pd.read_csv(files[0], parse_dates=["open_time", "close_time"])

    first_date = df["open_time"].min()
    last_date = df["open_time"].max()
    if first_date > pd.Timestamp(start) or last_date < pd.Timestamp(end):
        raise ValueError(f"Data not found for {start}-{end} date range")

    return df.set_index("open_time")[["close"]].rename(columns={"close": ticker})


def load_data(tickers: list[str], start: str, end: str, interval: str, data_dir: str = "data") -> pd.DataFrame:
    """Load data for a list of assets and return as DataFrame."""
    base_dir = Path().resolve().parent / data_dir

    dfs = [load_single_ticker(t, start, end, interval, base_dir) for t in tickers]

    data = pd.concat(dfs, axis=1)
    data = data[(data.index >= start) & (data.index <= end)]

    if data.empty:
        raise ValueError(f"No data available for tickers {tickers} in range {start} to {end}")

    return data


def load_pair(x: str, y: str, start: str, end: str, interval: str, data_dir: str = "data") -> Pair:
    """Load data for a single pair and return as Pair."""
    base_dir = Path().resolve().parent / data_dir

    df_x = load_single_ticker(x, start, end, interval, base_dir)
    df_y = load_single_ticker(y, start, end, interval, base_dir)

    data = pd.concat([df_x, df_y], axis=1)
    data = data[(data.index >= start) & (data.index <= end)]

    if data.empty:
        raise ValueError(f"No data available for tickers {[x, y]} in range {start} to {end}")

    return Pair(x=x, y=y, start=start, end=end, interval=interval, data=data)


def merge_by_pair(dfs: list[pd.DataFrame], keep_cols: list[list[str]]) -> pd.DataFrame:
    trimmed = []
    for df, cols in zip(dfs, keep_cols):
        trimmed.append(df[['pair'] + cols])

    merged = reduce(lambda left, right: pd.merge(left, right, on='pair', how='outer'), trimmed)
    return merged
