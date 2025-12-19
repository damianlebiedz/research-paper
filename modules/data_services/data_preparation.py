import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    """Add returns and log returns to Pair data."""
    df[f"{ticker_x}_returns"] = df[ticker_x].pct_change().fillna(0.0)
    df[f"{ticker_y}_returns"] = df[ticker_y].pct_change().fillna(0.0)

    df[f"{ticker_x}_log_returns"] = np.log(df[ticker_x] / df[ticker_x].shift(1))
    df[f"{ticker_x}_log_returns"].fillna(0.0, inplace=True)

    df[f"{ticker_y}_log_returns"] = np.log(df[ticker_y] / df[ticker_y].shift(1))
    df[f"{ticker_y}_log_returns"].fillna(0.0, inplace=True)

    df[f"{ticker_x}_c_returns"] = (1 + df[f"{ticker_x}_returns"]).cumprod()
    df[f"{ticker_y}_c_returns"] = (1 + df[f"{ticker_y}_returns"]).cumprod()

    df[f"{ticker_x}_c_log_returns"] = np.exp(df[f"{ticker_x}_log_returns"].cumsum())
    df[f"{ticker_y}_c_log_returns"] = np.exp(df[f"{ticker_y}_log_returns"].cumsum())


def add_log_prices(df: pd.DataFrame, ticker_x: str, ticker_y: str) -> None:
    """Add log prices to Pair data."""
    df[f"{ticker_x}_log"] = np.log(df[ticker_x])
    df[f"{ticker_y}_log"] = np.log(df[ticker_y])
