import pandas as pd


def cumulative_returns_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prices with cumulative returns to start from 1."""
    df = df.copy()
    for column in df.columns:
        df[column] = (1 + df[column].pct_change().fillna(0)).cumprod()
    return df


# def minmax_scale(pair_data: Pair) -> Pair:
#     """Scale prices to range [0, 1]."""
#     df = pair_data.data.copy()
#
#     def minmax_scale_series(series: pd.Series) -> pd.Series:
#         """Scale a series to range [0, 1]."""
#         return (series - series.min()) / (series.max() - series.min())
#
#     df[f"{pair_data.x}_scaled"] = minmax_scale_series(df[pair_data.x])
#     df[f"{pair_data.y}_scaled"] = minmax_scale_series(df[pair_data.y])
#     pair_data.data = df
#     return pair_data
