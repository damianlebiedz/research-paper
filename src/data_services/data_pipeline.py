from functools import reduce
from pathlib import Path
from typing import List
import pandas as pd

from src.data_services.data_models import PairData


def minmax_scale(series: pd.Series) -> pd.Series:
    """Scale a series to range [0, 1]."""
    return (series - series.min()) / (series.max() - series.min())


def load_data(tickers: List[str], start: str, end: str, interval: str, data_dir: str = "data") -> pd.DataFrame:
    """Load data for a list of assets and return as DataFrame."""
    dfs = []
    base_dir = Path().resolve().parent / data_dir

    for ticker in tickers:
        ticker_dir = base_dir / ticker
        if not ticker_dir.exists():
            raise FileNotFoundError(f"Directory not found: {ticker_dir}")

        files = list(ticker_dir.glob(f"*_{interval}.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV file with interval '{interval}' found in {ticker_dir}")

        df = pd.read_csv(files[0], parse_dates=["open_time", "close_time"])
        df = df.set_index("open_time")[["close"]].rename(columns={"close": ticker})
        dfs.append(df)

    data = pd.concat(dfs, axis=1)
    data = data[(data.index >= start) & (data.index <= end)]

    if data.empty:
        raise ValueError(f"No data available for tickers {tickers} in range {start} to {end}")

    return data


def load_pair(x: str, y: str, start: str, end: str, interval: str, data_dir: str = "data") -> PairData:
    """Load data for a single pair and return as PairData."""
    dfs = []
    base_dir = Path().resolve().parent / data_dir

    for ticker in [x, y]:
        ticker_dir = base_dir / ticker
        if not ticker_dir.exists():
            raise FileNotFoundError(f"Directory not found: {ticker_dir}")

        files = list(ticker_dir.glob(f"*_{interval}.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV file with interval '{interval}' found in {ticker_dir}")

        df = pd.read_csv(files[0], parse_dates=["open_time", "close_time"])
        df = df.set_index("open_time")[["close"]].rename(columns={"close": ticker})
        dfs.append(df)

    data = pd.concat(dfs, axis=1)
    data = data[(data.index >= start) & (data.index <= end)]

    if data.empty:
        raise ValueError(f"No data available for tickers {[x, y]} in range {start} to {end}")

    return PairData(x=x, y=y, data=data)


def prepare_pair(pair_data: PairData) -> PairData:
    """Prepare a loaded pair: compute spread, z-score, minmax scaling."""
    df = pair_data.data.copy()
    df["Spread"] = df[pair_data.x] - df[pair_data.y]
    df["Z-Score"] = (df["Spread"] - df["Spread"].mean()) / df["Spread"].std()
    df[f"{pair_data.x}_scaled"] = minmax_scale(df[pair_data.x])
    df[f"{pair_data.y}_scaled"] = minmax_scale(df[pair_data.y])
    return PairData(x=pair_data.x, y=pair_data.y, data=df)


def load_and_prepare_pair(x: str, y: str, start: str, end: str, interval: str, data_dir: str = "data") -> PairData:
    """Load and prepare a single pair."""
    pair_data = load_pair(x, y, start, end, interval, data_dir)
    return prepare_pair(pair_data)


def merge_by_pair(dfs: list[pd.DataFrame], keep_cols: list[list[str]]) -> pd.DataFrame:
    trimmed = []
    for df, cols in zip(dfs, keep_cols):
        trimmed.append(df[['pair'] + cols])

    merged = reduce(lambda left, right: pd.merge(left, right, on='pair', how='outer'), trimmed)
    return merged
