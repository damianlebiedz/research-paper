import numpy as np
import pandas as pd

from src.data_services.data_models import PairData


def generate_trades_and_signals(data: pd.DataFrame, entry_threshold: float = None,
                                exit_threshold: float = None) -> None:
    df = data.data

    if 'Z-Score' not in df.columns:
        raise ValueError("DataFrame must contain a 'Z-Score' column.")

    missing_entry = 'entry_threshold' not in df.columns and entry_threshold is None
    missing_exit = 'exit_threshold' not in df.columns and exit_threshold is None

    if missing_entry and missing_exit:
        raise ValueError(
            "Missing both 'entry_threshold' and 'exit_threshold' — provide arguments or columns in the DataFrame.")
    elif missing_entry:
        raise ValueError("Missing 'entry_threshold' — provide an argument or a column in the DataFrame.")
    elif missing_exit:
        raise ValueError("Missing 'exit_threshold' — provide an argument or a column in the DataFrame.")

    if 'entry_threshold' not in df.columns:
        df['entry_threshold'] = entry_threshold
    if 'exit_threshold' not in df.columns:
        df['exit_threshold'] = exit_threshold

    zscore = df['Z-Score']
    entry_thr = df['entry_threshold']
    exit_thr = df['exit_threshold']

    long_sig = zscore < -entry_thr
    short_sig = zscore > entry_thr
    exit_sig = (zscore.shift(1) * zscore < exit_thr)
    exit_sig.iloc[0] = False

    position = np.zeros(len(df))

    for t in range(1, len(df)):
        if long_sig.iloc[t]:
            position[t] = 1
        elif short_sig.iloc[t]:
            position[t] = -1
        elif exit_sig.iloc[t]:
            position[t] = 0
        else:
            position[t] = position[t - 1]

    df['position'] = position
    df['signal'] = 0
    df.loc[long_sig, 'signal'] = 1
    df.loc[short_sig, 'signal'] = -1
    df.loc[exit_sig, 'signal'] = 0
    data.data = df


def benchmark_static_strategy(data: PairData, initial_cash: float, fee_rate: float | None = None,
                              weight_x: float | None = None, weight_y: float | None = None) -> None:
    x_col = data.x
    y_col = data.y
    df = data.data

    for col in [x_col, y_col, 'position']:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")

    if 'weight_x' in df.columns and 'weight_y' in df.columns:
        wx = df['weight_x']
        wy = df['weight_y']
    elif weight_x is not None and weight_y is not None:
        wx = pd.Series(weight_x, index=df.index)
        wy = pd.Series(weight_y, index=df.index)
        df['weight_x'] = wx
        df['weight_y'] = wy
    else:
        raise ValueError("Missing weights: provide 'weight_x'/'weight_y' columns or function arguments.")

    cash = initial_cash
    prev_pos = 0
    total_fees = 0.0
    realized_pnl = 0.0
    entry_price_x = entry_price_y = None
    q_x = q_y = 0.0
    cash_curve, fee_curve, pnl_curve = [], [], []

    for i in range(len(df)):
        price_x = df[x_col].iloc[i]
        price_y = df[y_col].iloc[i]
        pos = df['position'].iloc[i]
        w_x = wx.iloc[i]
        w_y = wy.iloc[i]

        if prev_pos == 0 and pos != 0:
            q_x = (cash * w_x) / price_x * pos
            q_y = -(cash * w_y) / price_y * pos
            entry_price_x, entry_price_y = price_x, price_y
            entry_val = abs(q_x * price_x) + abs(q_y * price_y)
            cash = 0
            if fee_rate:
                fee = entry_val * fee_rate
                total_fees += fee
                cash -= fee
            prev_pos = pos

        elif prev_pos != 0 and pos == 0:
            pnl = (q_x * (price_x - entry_price_x)) + (q_y * (price_y - entry_price_y))
            realized_pnl += pnl
            cash = initial_cash + realized_pnl
            q_x = q_y = 0.0
            prev_pos = 0

        if prev_pos != 0:
            current_pnl = (q_x * (price_x - entry_price_x)) + (q_y * (price_y - entry_price_y))
            total_pnl = realized_pnl + current_pnl
        else:
            total_pnl = realized_pnl

        pnl_curve.append(total_pnl)
        cash_curve.append(cash)
        fee_curve.append(total_fees)

    df['fees_paid'] = pd.Series(fee_curve, index=df.index)
    df['realized_pnl'] = pd.Series(pnl_curve, index=df.index)
    df['cash'] = pd.Series(cash_curve, index=df.index)
    df['pnl_pct'] = 1 + df['realized_pnl'] / initial_cash
    df['net_pnl'] = df['realized_pnl'] - df['fees_paid']
    df['net_pnl_pct'] = 1 + df['net_pnl'] / initial_cash
    data.data = df


def calculate_stats(data: PairData) -> str:
    df = data.data

    def compute_stats(pnl_series):
        start_cash = df['cash'].iloc[0]
        pnl_series_total = pnl_series + start_cash

        start_value = pnl_series_total.iloc[0] if pnl_series_total.iloc[0] != 0 else 1.0
        total_return = (pnl_series_total.iloc[-1] / start_value - 1) * 100

        daily_values = pnl_series_total.resample('1D').last()
        daily_returns = daily_values.pct_change().fillna(0)
        if daily_returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = np.sqrt(365) * daily_returns.mean() / daily_returns.std()
        return total_return, sharpe

    total_return_brutto, sharpe_brutto = compute_stats(df["realized_pnl"])
    total_return_netto, sharpe_netto = compute_stats(df["net_pnl"])

    report = (
        f"Backtest Statistics:\n"
        f"{'Total Return % (brutto)':20}: {total_return_brutto:6.2f}%\n"
        f"{'Total Return % (netto)':20}: {total_return_netto:6.2f}%\n"
        f"{'Sharpe Approx (brutto)':20}: {sharpe_brutto:6.2f}\n"
        f"{'Sharpe Approx (netto)':20}: {sharpe_netto:6.2f}"
    )
    return report
