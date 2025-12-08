import pandas as pd
import statsmodels.api as sm

from modules.core.models import Pair


def generate_signal(entry_threshold: float, z_score: float) -> Pair:
    """
    Generate signals for trades depends on current Z-Score.

    Signal = 1:     Long X, Short Y
    Signal = -1:    Short X, Long Y
    Signal = 0:     do nothing
    """
    signal = 0
    if z_score is not None:
        if z_score <= -entry_threshold:
            signal = 1
        elif z_score >= entry_threshold:
            signal = -1
    return signal


def get_spread(x: str, y: str, position: float) -> tuple[float, float]:  # TODO
    """Get spread for two assets depending on position."""
    if position == 0:
        # SPREAD FOR POSITION CLOSING
        ...
    elif position > 0:
        # SPREAD FOR POSITIVE POSITION OPENING
        ...
    else:
        # SPREAD FOR NEGATIVE POSITION OPENING
        ...
    return 1, 1


def calculate_beta_returns(x_returns: str, y_returns: str, df: pd.DataFrame) -> float:
    """Calculate beta from OLS with returns for a pair."""
    X = sm.add_constant(df[y_returns])
    y = df[x_returns]
    model = sm.OLS(y, X, missing="drop").fit()

    beta = model.params[y_returns]
    return beta


def calculate_zscore_prices(x_price: str, y_price: str, beta: float, df: pd.DataFrame) -> float:
    """Calculate Z-Score with beta and spread, mean, std from prices."""
    spread_series = df[x_price] - (beta * df[y_price])
    mean = spread_series.mean()
    std = spread_series.std()
    spread = spread_series.iloc[-1]
    if std == 0:
        return None

    z_score = (spread - mean) / std
    return z_score
