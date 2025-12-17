from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from modules.core.models import Pair
from modules.data_services.data_loaders import load_data


def get_steps(interval: str) -> int:
    """Get steps of the interval."""
    if interval == '1d':
        return 1
    elif interval == '4h':
        return 6
    elif interval == '1h':
        return 24
    elif interval == '30m':
        return 48
    elif interval == '15m':
        return 96
    elif interval == '5m':
        return 288
    elif interval == '3m':
        return 480
    elif interval == '1m':
        return 1440
    else:
        return ValueError(
            f"Wrong interval '{interval}', should be one of: '1d', '4h', '1h', '30m', '15m', '5m', '3m', '1m'.")


def merge_by_pair(dfs: list[pd.DataFrame], keep_cols: list[list[str]]) -> pd.DataFrame:
    """Merge dataframes from statistical tests into one dataframe."""
    trimmed = []
    for df, cols in zip(dfs, keep_cols):
        trimmed.append(df[['pair'] + cols])

    merged = reduce(lambda left, right: pd.merge(left, right, on='pair', how='outer'), trimmed)
    return merged


def add_returns(pair: Pair) -> None:
    """Add return and log return columns into Pair data."""
    data = pair.data.copy()
    col_x = pair.x
    col_y = pair.y

    data[f"{col_x}_returns"] = data[col_x].pct_change()
    data[f"{col_y}_returns"] = data[col_y].pct_change()

    data[f"{col_x}_log_returns"] = np.log(data[col_x] / data[col_x].shift(1))
    data[f"{col_y}_log_returns"] = np.log(data[col_y] / data[col_y].shift(1))

    pair.data = data.dropna()


def cumulative_returns_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prices with cumulative returns to start from 1."""
    df = df.copy()
    for column in df.columns:
        df[column] = (1 + df[column].pct_change().fillna(0)).cumprod()
    return df


def minmax_scale(pair_data: Pair) -> Pair:
    """Scale prices to range [0, 1]."""
    df = pair_data.data.copy()

    def minmax_scale_series(series: pd.Series) -> pd.Series:
        """Scale a series to range [0, 1]."""
        return (series - series.min()) / (series.max() - series.min())

    df[f"{pair_data.x}_scaled"] = minmax_scale_series(df[pair_data.x])
    df[f"{pair_data.y}_scaled"] = minmax_scale_series(df[pair_data.y])
    pair_data.data = df
    return pair_data


def load_btc_benchmark(test_start: str, test_end: str, interval: str) -> pd.DataFrame:
    btc_data = load_data(
        tickers=['BTCUSDT'],
        start=test_start,
        end=test_end,
        interval=interval,
    )
    btc_data['BTC_return'] = btc_data['BTCUSDT'].pct_change()
    btc_data.loc[btc_data.index[0], 'BTC_return'] = 0.0
    btc_data['BTC_cum_return'] = (1 + btc_data['BTC_return']).cumprod() - 1
    return btc_data


def save_to_parquet(df: pd.DataFrame, file_name: str) -> None:
    PARQUET_DIR = Path.cwd() / "parquets"
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_DIR / f"{file_name}.parquet")


def load_parquet(file_name: str) -> pd.DataFrame:
    PARQUET_DIR = Path.cwd() / "parquets"
    return pd.read_parquet(PARQUET_DIR / f"{file_name}.parquet")
