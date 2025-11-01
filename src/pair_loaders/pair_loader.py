import pandas as pd
from pathlib import Path

from src.models.pair_data_model import PairData


def minmax_scale(series):
    """Scale a series to range [0, 1]."""
    return (series - series.min()) / (series.max() - series.min())


class PairLoader:
    """Utility class for loading and preparing market data for a single pair from CSV files."""

    def __init__(self, x: str, y: str, data_dir: str = "data"):
        self.x = x
        self.y = y
        self.data_dir = Path().resolve().parent / data_dir

    def load(self, start: str, end: str, interval: str) -> PairData:
        """Load data for a given pair from CSV files."""
        dfs = []
        for ticker in [self.x, self.y]:
            ticker_dir = self.data_dir / ticker
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
            raise ValueError(f"No data available for tickers {[self.x, self.y]} in range {start} to {end}")
        return PairData(x=self.x, y=self.y, data=data)

    def prepare(self, pair_data: PairData) -> PairData:
        """Prepare data for a loaded pair."""
        data = pair_data.data.copy()
        data["Spread"] = data[self.x] - data[self.y]
        data["Z-Score"] = (data["Spread"] - data["Spread"].mean()) / data["Spread"].std()
        data[f"{self.x}_scaled"] = minmax_scale(data[self.x])
        data[f"{self.y}_scaled"] = minmax_scale(data[self.y])
        return PairData(x=self.x, y=self.y, data=data)

    def load_and_prepare(self, start: str, end: str, interval: str) -> PairData:
        """Load and prepare data for a given pair from CSV files."""
        pair_data = self.load(start=start, end=end, interval=interval)
        prepared = self.prepare(pair_data)
        return prepared
