import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Utility class for loading and preparing market data for strategies from CSV files.
    """

    def __init__(self, interval="1d"):
        self.interval = interval

    def load(self, tickers, start=None, end=None, data_dir="data"):
        """
        Load and prepare data for a given pair of tickers from CSV files.
        """

        dfs = []
        for ticker in tickers:
            ticker_dir = Path(data_dir) / ticker
            if not ticker_dir.exists():
                raise FileNotFoundError(f"Directory not found: {ticker_dir}")

            files = list(ticker_dir.glob(f"*_{self.interval}.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV file with interval '{self.interval}' found in {ticker_dir}")

            file_path = files[0]

            df = pd.read_csv(file_path, parse_dates=["open_time", "close_time"])
            df = df.set_index("close_time")
            df = df[["close"]].rename(columns={"close": ticker})

            dfs.append(df)

        data = pd.concat(dfs, axis=1)

        if data.empty:
            raise ValueError(f"No data available for tickers {tickers} in the selected date range: {start} to {end}")

        if start and data.index.min().date() > pd.to_datetime(start).date():
            raise ValueError(f"Data for {tickers} does not start at requested start date {start}. "
                             f"First available date: {data.index.min().date()}")
        if end and data.index.max().date() < pd.to_datetime(end).date():
            raise ValueError(f"Data for {tickers} does not reach requested end date {end}. "
                             f"Last available date: {data.index.max().date()}")

        data = data[(data.index >= '2024-01-01') & (data.index <= '2024-03-01')]

        data["Spread"] = data[tickers[0]] - data[tickers[1]]
        data["Z-Score"] = (data["Spread"] - data["Spread"].mean()) / data["Spread"].std()

        return data
