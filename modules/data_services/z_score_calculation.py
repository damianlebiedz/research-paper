import numpy as np
import statsmodels.api as sm

from modules.data_services.data_models import PairData


def calculate_rolling_zscore(pair_data: PairData, rolling_window: int, source: str = "price") -> PairData:
    """
    Calculate rolling z-score for a pair based on a selected source.
    Automatically finds the first index with enough historical data for pre-training.
    Raises ValueError if not enough data for pre-training.
    """
    df = pair_data.data.copy()

    if source == "price":
        df['spread'] = df[pair_data.x] - df[pair_data.y]
    elif source == "return":
        df['spread'] = df[pair_data.x].pct_change() - df[pair_data.y].pct_change()
    elif source == "log_return":
        df['spread'] = np.log(df[pair_data.x]) - np.log(df[pair_data.y])
    else:
        raise ValueError("source must be one of ['price', 'return', 'log_return']")

    spreads = df['spread'].values
    z_scores = [None] * len(df)

    for i in range(len(df)):
        if i < rolling_window:
            continue
        window = spreads[i - rolling_window:i]
        if len(window) < rolling_window:
            raise ValueError(f"Not enough data for pre-training at index {df.index[i]}")
        mean = np.mean(window)
        std = np.std(window)
        z_scores[i] = (spreads[i] - mean) / std if std != 0 else None

    df['z_score'] = z_scores
    pair_data.data = df
    return pair_data


def calculate_rolling_zscore_with_rolling_beta(pair_data: PairData, rolling_window: int):
    """
    Calculate rolling z-score from raw prices using rolling beta.
    Automatically finds the first index with enough historical data for pre-training.
    Raises ValueError if not enough data for pre-training.
    """
    df = pair_data.data.copy()

    spreads = []
    z_scores = []
    betas = []

    start_index = rolling_window
    if start_index > len(df):
        raise ValueError(f"Not enough data for pre-training: need at least {rolling_window} historical points.")

    first_valid_index = None
    for i in range(start_index, len(df)):
        if i - rolling_window >= 0:
            first_valid_index = i
            break
    if first_valid_index is None:
        raise ValueError("Not enough data for pre-training. Please provide at least rolling_window historical points.")

    for _ in range(first_valid_index):
        spreads.append(None)
        z_scores.append(None)
        betas.append(None)

    for i in range(first_valid_index, len(df)):
        window_df = df.iloc[i - rolling_window:i]

        X = sm.add_constant(window_df[pair_data.y])
        y = window_df[pair_data.x]
        model = sm.OLS(y, X).fit()
        beta = model.params[pair_data.y]

        spread_t = df[pair_data.x].iloc[i] - beta * df[pair_data.y].iloc[i]

        mean_t = (window_df[pair_data.x] - beta * window_df[pair_data.y]).mean()
        std_t = (window_df[pair_data.x] - beta * window_df[pair_data.y]).std()

        z_t = (spread_t - mean_t) / std_t if std_t != 0 else None

        spreads.append(spread_t)
        z_scores.append(z_t)
        betas.append(beta)

    df['beta'] = betas
    df['spread'] = spreads
    df['z_score'] = z_scores

    pair_data.data = df
    return pair_data
