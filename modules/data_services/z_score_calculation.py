import numpy as np
import statsmodels.api as sm

from modules.data_services.data_models import Pair
from modules.data_services.normalization import cumulative_returns_index


def calculate_rolling_zscore(pair_data: Pair, rolling_window: int, source: str = "prices") -> Pair:
    """
    Calculate rolling z-score for a pair based on a selected source.
    Automatically finds the first index with enough historical data for pre-training.
    Raises ValueError if not enough data for pre-training.
    TODO
    """
    df = pair_data.data.copy()
    n = len(df)

    if n < rolling_window:
        raise ValueError(f"Not enough data: need at least {rolling_window} historical points.")

    if source == "prices":
        df['spread'] = df[pair_data.x] - df[pair_data.y]
    elif source == "returns":
        df['spread'] = df[pair_data.x].pct_change() - df[pair_data.y].pct_change()
    elif source == "log_returns":
        df['spread'] = np.log(df[pair_data.x]) - np.log(df[pair_data.y])
    elif source == "cum_returns":
        df_tmp = cumulative_returns_index(df[[pair_data.x, pair_data.y]])
        df['spread'] = df_tmp[pair_data.x] - df_tmp[pair_data.y]
    else:
        raise ValueError("source must be one of ['prices', 'returns', 'log_returns', 'cum_returns]")

    spreads = df['spread'].values
    z_scores = [None] * n

    for i in range(len(df)):
        if i < rolling_window:
            continue
        window = spreads[i - rolling_window:i]
        mean = np.mean(window)
        std = np.std(window)
        z_scores[i] = (spreads[i] - mean) / std if std != 0 else None

    df['z_score'] = z_scores
    pair_data.data = df
    return pair_data


def calculate_rolling_zscore_with_rolling_beta(pair_data: Pair, rolling_window: int) -> Pair:
    """
    Calculate rolling z-score from raw prices using rolling beta.
    Automatically finds the first index with enough historical data for pre-training.
    Raises ValueError if not enough data for pre-training.
    TODO
    """
    df = pair_data.data.copy()
    n = len(df)

    if n < rolling_window:
        raise ValueError(f"Not enough data: need at least {rolling_window} historical points.")

    # 1. Initialize lists for new columns
    betas = [None] * n
    # spreads = [None] * n
    z_scores = [None] * n
    means = [None] * n
    stds = [None] * n

    start_index = rolling_window
    for i in range(start_index, n):
        window_df = df.iloc[i - rolling_window: i]

        X = sm.add_constant(window_df[pair_data.y])
        y = window_df[pair_data.x]

        # Perform OLS
        model = sm.OLS(y, X, missing='drop').fit()
        if pair_data.y not in model.params:
            continue

        beta = model.params[pair_data.y]
        betas[i] = beta

        # Calculate current spread
        spread_t = df[pair_data.x].iloc[i] - beta * df[pair_data.y].iloc[i]
        # spreads[i] = spread_t

        # Calculate stats on the window
        window_spread = window_df[pair_data.x] - beta * window_df[pair_data.y]
        mean_t = window_spread.mean()
        std_t = window_spread.std()

        # 2. Store the calculated mean and std in the lists
        means[i] = mean_t
        stds[i] = std_t

        # Calculate Z-Score
        z_t = (spread_t - mean_t) / std_t if std_t != 0 else None
        z_scores[i] = z_t

    # 3. Save lists to DataFrame columns
    df['beta'] = betas
    # df['spread'] = spreads
    df['mean'] = means
    df['std'] = stds
    df['z_score'] = z_scores

    pair_data.data = df
    return pair_data
