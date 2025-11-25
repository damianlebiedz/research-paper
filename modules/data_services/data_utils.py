import math
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from modules.data_services.data_models import Portfolio, Pair


def get_steps(interval: str) -> int:
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


def pre_training_start(start: str, interval: str, rolling_window_steps: float) -> str:
    steps_per_day = get_steps(interval)
    days = math.ceil(rolling_window_steps / steps_per_day)
    date_object = datetime.strptime(start, "%Y-%m-%d")
    new_date_object = date_object - timedelta(days=days)
    return new_date_object.strftime("%Y-%m-%d")


def calc_portfolio_summary(portfolio: Portfolio) -> pd.DataFrame:
    rows = []
    row_labels = []

    for pair_data in portfolio.pairs_data:
        if pair_data.stats is None:
            continue

        pair_name = f"{pair_data.x}-{pair_data.y}"
        df = pair_data.stats.T
        rows.append(df)
        row_labels.append(pair_name)

    if portfolio.stats is not None:
        df_all = portfolio.stats.T
        rows.append(df_all)
        row_labels.append("Summary")

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, keys=row_labels, axis=0)
    final = combined.unstack(level=1)

    # Opcjonalnie: Zamiana kolejności poziomów kolumn, aby grupować po Fee
    # final = final.swaplevel(0, 1, axis=1).sort_index(axis=1)

    final.columns.names = ["Metric", "Fee Scenario"]
    final.index.name = "Pair"

    return final


def get_pair_data(portfolio_data: Portfolio, x_asset: str, y_asset: str) -> Optional[Pair]:
    matching_pairs = [pair for pair in portfolio_data.pairs_data if pair.x == x_asset and pair.y == y_asset]
    if matching_pairs:
        return matching_pairs[0]
    else:
        return None
