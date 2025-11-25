import numpy as np
import pandas as pd

from modules.data_services.data_models import Pair
from modules.data_services.data_utils import get_steps
from modules.data_services.normalization import cumulative_returns_index


def generate_signal(entry_threshold: float, spread: float, std: float) -> Pair:
    signal = 0
    if spread is not None:
        if spread <= -(entry_threshold * std):
            signal = 1  # long x short y
        elif spread >= entry_threshold * std:
            signal = -1  # short x long y
    return signal


def generate_trade(position: float, prev_position: float, q_x: float, q_y: float, w_x: float, w_y: float,
                   exit_threshold: float, stop_loss: float, spread: float, price_x: float,
                   price_y: float, initial_cash: float, fee_rate: float, total_fees: float, entry_val: float,
                   stop_loss_threshold: float):
    def open_position():
        # Weights (dollar neutral)
        wx = 0.5
        wy = 0.5

        # Position
        if position > 0:
            qx = initial_cash * wx / price_x
            qy = -(initial_cash * wy) / price_y
        else:
            qx = -(initial_cash * wx) / price_x
            qy = initial_cash * wy / price_y

        entry_value = abs(qx) * price_x + abs(qy) * price_y
        pos_fees = entry_value * fee_rate
        stop_loss_thr = abs(spread * stop_loss)
        return qx, qy, wx, wy, entry_value, pos_fees, stop_loss_thr

    def close_position():
        exit_value = abs(q_x) * price_x + abs(q_y) * price_y
        pos_fees = exit_value * fee_rate
        if prev_position > 0:
            pos_pnl = exit_value - entry_val
        else:
            pos_pnl = entry_val - exit_value
        return pos_pnl, pos_fees

    if position != 0:
        # IN POSITION
        if position < 0:
            if spread <= exit_threshold or (
                    stop_loss_threshold is not None and spread >= stop_loss_threshold):
                # CLOSE POSITION
                pnl, close_fees = close_position()
                total_fees += close_fees
                return 0, pnl, total_fees, spread, 0, 0, None, None, 0, None
        else:
            if spread >= -exit_threshold or (
                    stop_loss_threshold is not None and spread <= -stop_loss_threshold):
                # CLOSE POSITION
                pnl, close_fees = close_position()
                total_fees += close_fees
                return 0, pnl, total_fees, spread, 0, 0, None, None, 0, None
        if prev_position == position:
            # STAY IN POSITION
            exit_val = abs(q_x) * price_x + abs(q_y) * price_y
            if position > 0:
                pnl = exit_val - entry_val
            else:
                pnl = entry_val - exit_val
            return prev_position, pnl, total_fees, spread, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold
        else:
            # REVERSE (CLOSE OLD, OPEN NEW)
            pnl, open_fees = close_position()
            q_x, q_y, w_x, w_y, entry_val, close_fees, stop_loss_threshold = open_position()
            total_fees += open_fees + close_fees
            return position, pnl, total_fees, spread, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold
    else:
        # OUT OF POSITION
        if position != 0:
            # OPEN POSITION
            q_x, q_y, w_x, w_y, entry_val, open_fees, stop_loss_threshold = open_position()
            total_fees += open_fees
            return position, 0, total_fees, spread, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold
        else:
            # STAY OUT OF POSITION
            return 0, 0, total_fees, spread, 0, 0, None, None, 0, None


def calculate_std(col_x, col_y, df: pd.DataFrame):
    spread_col = df[col_x] - df[col_y]
    std = spread_col.std()
    return std


def run_strategy(pair_data: Pair, entry_threshold: float, exit_threshold: float, stop_loss: float,
                 pos_size: float) -> Pair:
    df = cumulative_returns_index(pair_data.data.copy())
    x_col, y_col = pair_data.x, pair_data.y
    fee_rate = pair_data.fee_rate
    initial_cash = pair_data.initial_cash

    total_fees = 0.0
    total_pnl = 0.0
    prev_pnl = 0.0
    prev_pos = 0
    q_x = 0
    q_y = 0
    w_x = 0
    w_y = 0
    entry_val = 0

    stop_loss_threshold = None

    spread_df = df.iloc[:4368]  # 182 days - training period, 6 months 2024-01-01 - 2024-07-01 in 1h interval
    std = calculate_std(x_col, y_col, spread_df)

    for i in range(4368, len(df)):
        if total_pnl == -initial_cash:
            # BANKRUPT
            df = df.iloc[:i].copy()
            break
        else:
            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            # Spread = |CenaNorm_A - CenaNorm_B|
            spread = df[x_col].iloc[i] - df[y_col].iloc[i]

            signal = generate_signal(entry_threshold, spread, std)  # {-1,0,1}

            if prev_pos == 0:
                position = signal * pos_size
            else:
                position = prev_pos

            prev_pos, pnl, total_fees, spread, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold = generate_trade(
                position, prev_pos, q_x, q_y, w_x, w_y, exit_threshold, stop_loss, spread, price_x, price_y,
                initial_cash, fee_rate, total_fees, entry_val, stop_loss_threshold)

            if pnl != 0:
                total_pnl = pnl + prev_pnl
            else:
                prev_pnl = total_pnl

        if total_pnl <= -initial_cash:
            total_pnl = -initial_cash

        idx = df.index[i]
        df.at[idx, 'spread'] = spread
        df.at[idx, 'std'] = std
        df.at[idx, 'stop_loss_threshold'] = stop_loss_threshold
        df.at[idx, 'weight_x'] = w_x
        df.at[idx, 'weight_y'] = w_y
        df.at[idx, 'q_x'] = q_x
        df.at[idx, 'q_y'] = q_y
        df.at[idx, 'cash'] = initial_cash - entry_val
        df.at[idx, 'signal'] = signal
        df.at[idx, 'position'] = prev_pos
        df.at[idx, 'total_pnl'] = total_pnl
        df.at[idx, 'total_fees'] = total_fees
        df.at[idx, 'net_pnl'] = total_pnl - total_fees

    df['total_pnl_pct'] = df['total_pnl'] / initial_cash
    df['net_pnl_pct'] = df['net_pnl'] / initial_cash

    pair_data.data = df
    return pair_data


def calculate_stats(pair_data: Pair) -> pd.DataFrame:
    df = pair_data.data.copy()
    fee_rate = pair_data.fee_rate
    initial_cash = pair_data.initial_cash

    steps_per_day = get_steps(pair_data.interval)
    periods_per_year = steps_per_day * 365

    def calc_trade_stats(pnl_series, position_series, initial_cash: float) -> pd.DataFrame:
        trade_count, total_wins, total_losses = 0, 0, 0
        max_win, max_lose = 0, 0
        max_win_pct, max_lose_pct = 0, 0
        total_trade_pnl_sum = 0
        prev = 0
        open_idx = None

        for i in range(len(df)):
            pos = position_series.iloc[i]

            if prev == 0 and pos != 0:
                # OPEN POSITION
                open_idx = i
            elif (prev < 0 <= pos) or (prev > 0 >= pos) or (prev != 0 and i == len(position_series) - 1):
                # CLOSE POSITION
                trade_count += 1
                if open_idx is not None:
                    pnl = pnl_series.iloc[i] - pnl_series.iloc[open_idx]
                    total_trade_pnl_sum += pnl

                    # Win / lose count
                    if pnl > 0:
                        total_wins += 1
                    elif pnl < 0:
                        total_losses += 1

                    # Max win / lose
                    max_win = max(max_win, pnl)
                    max_lose = min(max_lose, pnl)

                    # Max win / lose [%]
                    max_win_pct = max_win / initial_cash
                    max_lose_pct = max_lose / initial_cash

                open_idx = i if pos != 0 else None
            prev = pos
        avg_trade_return = total_trade_pnl_sum / trade_count if trade_count > 0 else 0
        avg_trade_ret_pct = avg_trade_return / initial_cash

        return total_wins, total_losses, max_win_pct, max_lose_pct, avg_trade_ret_pct

    def compute_stats(pnl_series):
        pnl_series_limited = pnl_series.clip(lower=-initial_cash)
        equity_curve = pnl_series_limited + initial_cash
        returns = equity_curve.pct_change().dropna()

        total_pnl = pnl_series_limited.iloc[-1]
        total_return = total_pnl / initial_cash

        total_wins, total_losses, max_win_pct, max_lose_pct, avg_trade_ret_pct = calc_trade_stats(pnl_series,
                                                                                                  df['position'],
                                                                                                  initial_cash)
        total_trades = total_wins + total_losses
        win_rate = total_wins / total_trades

        period_volatility = returns.std()

        if pd.isna(period_volatility):
            period_volatility = 0.0

        annualized_volatility = period_volatility * np.sqrt(periods_per_year)

        if period_volatility == 0:
            sharpe_ratio, sharpe_ratio_annual = None, None
        else:
            sharpe_ratio = returns.mean() / period_volatility
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(periods_per_year)

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        if pd.isna(downside_std) or downside_std == 0:
            sortino_ratio, sortino_ratio_annual = None, None
        else:
            sortino_ratio = returns.mean() / downside_std
            sortino_ratio_annual = sortino_ratio * np.sqrt(periods_per_year)

        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max.replace(0, 1)
        max_drawdown = drawdown.min()

        years = len(df) / periods_per_year if len(df) > 0 else 0
        if years > 0 and equity_curve.iloc[-1] > 0 and equity_curve.iloc[0] > 0:
            cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            cagr = 0.0

        if max_drawdown == 0:
            calmar_ratio, calmar_ratio_annual = None, None
        else:
            calmar_ratio = total_return / abs(max_drawdown)
            calmar_ratio_annual = cagr / abs(max_drawdown)

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": period_volatility,
            "volatility_annual": annualized_volatility,
            "max_drawdown": max_drawdown,
            "win_count": total_wins,
            "lose_count": total_losses,
            "win_rate": win_rate,
            "max_win": max_win_pct,
            "max_lose": max_lose_pct,
            "avg_trade_return": avg_trade_ret_pct,
            "sharpe_ratio": sharpe_ratio,
            "sharpe_ratio_annual": sharpe_ratio_annual,
            "sortino_ratio": sortino_ratio,
            "sortino_ratio_annual": sortino_ratio_annual,
            "calmar_ratio": calmar_ratio,
            "calmar_ratio_annual": calmar_ratio_annual
        }

    brutto_stats = compute_stats(df["total_pnl"])
    netto_stats = compute_stats(df["net_pnl"])

    metrics_order = [
        "total_return", "cagr", "volatility", "volatility_annual", "max_drawdown", "win_count", "lose_count",
        "win_rate", "max_win", "max_lose", "avg_trade_return", "sharpe_ratio", "sharpe_ratio_annual", "sortino_ratio",
        "sortino_ratio_annual", "calmar_ratio", "calmar_ratio_annual"
    ]

    stats_df = pd.DataFrame({
        "metric": metrics_order,
        "0% fee": [brutto_stats[m] for m in metrics_order],
        f"{fee_rate * 100}% fee": [netto_stats[m] for m in metrics_order]
    }).set_index("metric")

    return stats_df
