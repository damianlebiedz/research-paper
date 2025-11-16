import pandas as pd

from modules.data_services.data_models import PairData


def cumulative_returns_index(pair_data: PairData) -> PairData:
    """Normalize prices with cumulative returns to start from 1."""
    df = pair_data.data.copy()
    df[f"{pair_data.x}"] = (1 + df[f"{pair_data.x}"].pct_change().fillna(0)).cumprod()
    df[f"{pair_data.y}"] = (1 + df[f"{pair_data.y}"].pct_change().fillna(0)).cumprod()
    pair_data.data = df
    return pair_data


def minmax_scale(pair_data: PairData) -> PairData:
    """Scale prices to range [0, 1]."""
    df = pair_data.data.copy()

    def minmax_scale_series(series: pd.Series) -> pd.Series:
        """Scale a series to range [0, 1]."""
        return (series - series.min()) / (series.max() - series.min())

    df[f"{pair_data.x}_scaled"] = minmax_scale_series(df[pair_data.x])
    df[f"{pair_data.y}_scaled"] = minmax_scale_series(df[pair_data.y])
    pair_data.data = df
    return pair_data
