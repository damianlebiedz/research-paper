import numpy as np
import pandas as pd

from modules.data_services.data_models import PairData


def generate_action_space(pair_data: PairData, entry_threshold: float = None,
                          exit_threshold: float = None, position_size: float | None = None) -> PairData:
    """
    Generate action space for strategy based on entry/exit thresholds and position size.

    Action = 1: long x short y (long leg)
    action = -1: short x long y (short leg)
    action = 0: exit position
    """
    df = pair_data.data

    if 'z_score' not in df.columns:
        raise ValueError("DataFrame must contain a 'z_score' column.")

    if 'entry_threshold' not in df.columns:
        if entry_threshold is None:
            raise ValueError("Missing 'entry_threshold' — provide argument or column in the DataFrame.")
        df['entry_threshold'] = entry_threshold

    if 'exit_threshold' not in df.columns:
        if exit_threshold is None:
            raise ValueError("Missing 'exit_threshold' — provide argument or column in the DataFrame.")
        df['exit_threshold'] = exit_threshold

    if 'position_size' in df.columns:
        pass
    elif position_size is not None:
        ps = pd.Series(position_size, index=df.index)
        df['position_size'] = ps
    else:
        raise ValueError("Missing position size: provide 'position_size' column or function argument.")

    zscore = df['z_score']
    entry_thr = df['entry_threshold']
    exit_thr = df['exit_threshold']

    action = pd.Series(0, index=zscore.index)
    for i in range(1, len(zscore)):
        zscore_i = zscore.iloc[i]
        entry_thr_i = entry_thr.iloc[i]
        exit_thr_i = exit_thr.iloc[i]
        if action.iloc[i - 1] == 0:
            if zscore_i < -entry_thr_i:
                action.iloc[i] = 1
            elif zscore_i > entry_thr_i:
                action.iloc[i] = -1
        elif action.iloc[i - 1] == 1:
            if zscore_i >= -exit_thr_i:
                action.iloc[i] = 0
            else:
                action.iloc[i] = 1
        elif action.iloc[i - 1] == -1:
            if zscore_i <= exit_thr_i:
                action.iloc[i] = 0
            else:
                action.iloc[i] = -1
        else:
            action[i] = action[i - 1]

    df['action'] = action
    pair_data.data = df

    return pair_data


def benchmark_strategy(pair_data: PairData, initial_cash: float, fee_rate: float | None = None) -> PairData:
    """
    Simulate trading for a pair using 'position' and 'position_size' columns.
    Fee tracking is separate from the cash in the portfolio.
    """
    df = pair_data.data.copy()
    x_col, y_col = pair_data.x, pair_data.y

    for col in [x_col, y_col, 'position']:
        if col not in df.columns:
            if col == 'position':
                df['position'] = df['action']
            else:
                raise ValueError(f"DataFrame must contain column '{col}'.")

    cash = initial_cash
    total_fees = 0.0
    realized_pnl = 0.0
    prev_pos = 0
    q_x = q_y = 0.0
    entry_price_x = entry_price_y = None
    cash_curve, fee_curve, pnl_curve = [], [], []

    for i in range(len(df)):
        price_x = df[x_col].iloc[i]
        price_y = df[y_col].iloc[i]
        pos = df['position'].iloc[i]
        pos_size = df['position_size'].iloc[i]
        w_x = w_y = pos_size / 2

        if prev_pos == 0 and pos != 0:
            invested = initial_cash * pos_size
            cash = initial_cash - invested
            if pos == 1:
                q_x = (invested * w_x / pos_size) / price_x
                q_y = -(invested * w_y / pos_size) / price_y
            elif pos == -1:
                q_x = -(invested * w_x / pos_size) / price_x
                q_y = (invested * w_y / pos_size) / price_y
            entry_price_x, entry_price_y = price_x, price_y
            entry_val = abs(q_x * price_x) + abs(q_y * price_y)
            if fee_rate:
                fee = entry_val * fee_rate
                total_fees += fee
            prev_pos = pos

        elif prev_pos != 0 and pos == 0:
            pnl = (q_x * (price_x - entry_price_x)) + (q_y * (price_y - entry_price_y))
            realized_pnl += pnl
            cash = initial_cash - (initial_cash * df['position_size'].iloc[i]) + realized_pnl
            q_x = q_y = 0.0
            prev_pos = 0

        if prev_pos != 0:
            current_pnl = (q_x * (price_x - entry_price_x)) + (q_y * (price_y - entry_price_y))
            total_pnl = realized_pnl + current_pnl
        else:
            total_pnl = realized_pnl

        cash_curve.append(cash)
        fee_curve.append(total_fees)
        pnl_curve.append(total_pnl)

    df['fees_paid'] = fee_curve
    df['realized_pnl'] = pnl_curve
    df['cash'] = cash_curve
    df['pnl_pct'] = 1 + df['realized_pnl'] / initial_cash
    df['net_pnl'] = df['realized_pnl'] - df['fees_paid']
    df['net_pnl_pct'] = 1 + df['net_pnl'] / initial_cash

    pair_data.data = df
    pair_data.fee_rate = fee_rate

    return pair_data


def calculate_stats(pair_data: PairData) -> pd.DataFrame:
    """Calculate benchmark statistics (Total PnL, Total Return, Annualized Volatility, Sharpe Ratio, Maximum Drawdown)."""
    df = pair_data.data.copy()
    fee_rate = pair_data.fee_rate

    def compute_stats(pnl_series):
        start_cash = df['cash'].iloc[0]
        pnl_series_total = pnl_series + start_cash

        start_value = pnl_series_total.iloc[0] if pnl_series_total.iloc[0] != 0 else 1.0
        total_return = (pnl_series_total.iloc[-1] / start_value - 1) * 100
        total_pnl = pnl_series_total.iloc[-1] - start_cash

        returns = pnl_series_total.pct_change().fillna(0)

        if len(df) > 1:
            delta_seconds = (df.index[1] - df.index[0]).total_seconds()
            periods_per_year = 365 * 24 * 60 * 60 / delta_seconds
        else:
            periods_per_year = 1

        annualized_vol = returns.std() * np.sqrt(periods_per_year)
        annualized_sharpe = 0.0 if annualized_vol == 0 else returns.mean() / returns.std() * np.sqrt(periods_per_year)

        cumulative_max = pnl_series_total.cummax()
        drawdown = (pnl_series_total - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100

        return {
            "total_pnl": total_pnl,
            "total_return": total_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": annualized_sharpe,
            "max_drawdown": max_drawdown
        }

    brutto_stats = compute_stats(df["realized_pnl"])
    netto_stats = compute_stats(df["net_pnl"])

    stats_df = pd.DataFrame({
        "metric": ["total_pnl", "total_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"],
        "0% fee": list(brutto_stats.values()),
        f"{fee_rate * 100}% fee": list(netto_stats.values())
    }).set_index("metric")

    return stats_df
