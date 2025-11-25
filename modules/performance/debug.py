import numpy as np
import pandas as pd
import statsmodels.api as sm

from modules.data_services.data_models import Pair
from modules.data_services.data_utils import get_steps


def generate_signal(entry_threshold: float, z_score: float) -> Pair:
    signal = 0
    if z_score is not None:
        if z_score <= -entry_threshold:
            signal = 1  # long x short y
        elif z_score >= entry_threshold:
            signal = -1  # short x long y
    return signal


def generate_trade(position: float, prev_position: float, q_x: float, q_y: float, w_x: float, w_y: float,
                   exit_threshold: float, stop_loss: float, beta: float, beta_pos: float,
                   alpha_pos: float, z_score: float, spread: float, price_x: float, price_y: float, mean_pos: float,
                   std_pos: float, initial_cash: float, fee_rate: float, total_fees: float,
                   entry_val: float, stop_loss_threshold: float, log_prices: bool):
    def generate_virtual_z_score():
        if log_prices:
            s_virt = np.log(price_x) - (alpha_pos + beta_pos * np.log(price_y))
        else:
            s_virt = price_x - (alpha_pos + beta_pos * price_y)

        if std_pos != 0:
            z_virt = (s_virt - mean_pos) / std_pos
        else:
            z_virt = None

        return z_virt, s_virt

    def open_position():
        # Weights
        hedged_price_y = beta * price_y
        wx = price_x / (price_x + hedged_price_y)
        wy = hedged_price_y / (price_x + hedged_price_y)

        # Position
        if position > 0:
            qx = initial_cash * wx / price_x
            qy = -(initial_cash * wy) / price_y
        else:
            qx = -(initial_cash * wx) / price_x
            qy = initial_cash * wy / price_y

        entry_value = abs(qx) * price_x + abs(qy) * price_y
        pos_fees = entry_value * fee_rate
        stop_loss_thr = abs(z_score * stop_loss)
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
        z_score_virt, spread_virt = generate_virtual_z_score()
        if beta < 0:
            return 0, None, None, None, None, 0, total_fees, None, None, 0, 0, None, None, 0, None
        elif position < 0:
            if z_score_virt <= exit_threshold or (
                    stop_loss_threshold is not None and z_score_virt >= stop_loss_threshold):
                # CLOSE POSITION
                pnl, close_fees = close_position()
                total_fees += close_fees
                return 0, beta_pos, alpha_pos, mean_pos, std_pos, pnl, total_fees, z_score_virt, spread_virt, 0, 0, None, None, 0, None
        else:
            if z_score_virt >= -exit_threshold or (
                    stop_loss_threshold is not None and z_score_virt <= -stop_loss_threshold):
                # CLOSE POSITION
                pnl, close_fees = close_position()
                total_fees += close_fees
                return 0, beta_pos, alpha_pos, mean_pos, std_pos, pnl, total_fees, z_score_virt, spread_virt, 0, 0, None, None, 0, None
        if prev_position == position:
            # STAY IN POSITION
            exit_val = abs(q_x) * price_x + abs(q_y) * price_y
            if position > 0:
                pnl = exit_val - entry_val
            else:
                pnl = entry_val - exit_val
            return prev_position, beta_pos, alpha_pos, mean_pos, std_pos, pnl, total_fees, z_score_virt, spread_virt, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold
        else:
            # REVERSE (CLOSE OLD, OPEN NEW)
            pnl, open_fees = close_position()
            q_x, q_y, w_x, w_y, entry_val, close_fees, stop_loss_threshold = open_position()
            total_fees += open_fees + close_fees
            return position, beta_pos, alpha_pos, mean_pos, std_pos, pnl, total_fees, z_score, spread, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold
    else:
        # OUT OF POSITION
        if position != 0:
            # TODO: umożliwić skalowanie istniejącej pozycji (zmniejszanie/zwiększanie)
            # OPEN POSITION
            q_x, q_y, w_x, w_y, entry_val, open_fees, stop_loss_threshold = open_position()
            total_fees += open_fees
            return position, beta_pos, alpha_pos, mean_pos, std_pos, 0, total_fees, z_score, spread, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold
        else:
            # STAY OUT OF POSITION
            return 0, None, None, None, None, 0, total_fees, None, None, 0, 0, None, None, 0, None


def calculate_rolling_zscore(col_x, col_y, df: pd.DataFrame, log_prices: bool):
    if log_prices:
        df = df.copy()
        df[col_x] = np.log(df[col_x])
        df[col_y] = np.log(df[col_y])

    X = sm.add_constant(df[col_y])
    y = df[col_x]

    # Run OLS
    model = sm.OLS(y, X, missing='drop').fit()

    beta = model.params[col_y]
    alpha = model.params['const']

    # Spread = P_A - (alpha + beta * P_B)
    spread_col = df[col_x] - (alpha + beta * df[col_y])
    mean = spread_col.mean()
    std = spread_col.std()
    spread = (spread_col.iloc[-1])

    if std == 0:
        return None, spread, alpha, mean, std, beta

    z_score = (spread - mean) / std

    return z_score, spread, alpha, mean, std, beta


def run_strategy(pair_data: Pair, entry_threshold: float, exit_threshold: float, stop_loss: float,
                 pos_size: float, rolling_window: int) -> Pair:
    df = pair_data.data.copy()
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

    beta_pos = None
    alpha_pos = None
    mean_pos = None
    std_pos = None
    stop_loss_threshold = None

    # log_prices = True
    log_prices = False

    for i in range(len(df)):
        if total_pnl == -initial_cash:
            # BANKRUPT
            df = df.iloc[:i].copy()
            break
        else:
            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            if i < rolling_window:
                z_score, spread, alpha, mean, std, beta = None, None, None, None, None, None
            else:
                z_score_df = df.iloc[i - rolling_window:i]
                z_score, spread, alpha, mean, std, beta = calculate_rolling_zscore(x_col, y_col, z_score_df, log_prices)

            beta_pos = beta if beta_pos is None else beta_pos
            alpha_pos = alpha if alpha_pos is None else alpha_pos
            mean_pos = mean if mean_pos is None else mean_pos
            std_pos = std if std_pos is None else std_pos

            # TODO: Docelowo będzie to generować agent
            # entry_threshold = ... [-inf,+inf]
            # ---

            signal = generate_signal(entry_threshold, z_score)  # {-1,0,1}

            # TODO: Docelowo będzie to generować agent
            # exit_threshold = ... [-inf,+inf]
            # position = ... [-1,1]
            if prev_pos == 0:
                position = signal * pos_size
            else:
                position = prev_pos
            # ---

            prev_pos, beta_pos, alpha_pos, mean_pos, std_pos, pnl, total_fees, z_score_virt, spread_virt, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold = generate_trade(
                position, prev_pos, q_x, q_y, w_x, w_y, exit_threshold, stop_loss, beta, beta_pos, alpha_pos, z_score,
                spread, price_x, price_y, mean_pos, std_pos, initial_cash, fee_rate, total_fees, entry_val,
                stop_loss_threshold, log_prices)

            if pnl != 0:
                total_pnl = pnl + prev_pnl
            else:
                prev_pnl = total_pnl

        if total_pnl <= -initial_cash:
            total_pnl = -initial_cash

        idx = df.index[i]
        df.at[idx, 'z_score'] = z_score
        df.at[idx, 'spread'] = spread
        df.at[idx, 'mean'] = mean
        df.at[idx, 'std'] = std
        df.at[idx, 'beta'] = beta
        df.at[idx, 'z_score_virtual'] = z_score_virt
        df.at[idx, 'spread_virt'] = spread_virt
        df.at[idx, 'mean_pos'] = mean_pos
        df.at[idx, 'std_pos'] = std_pos
        df.at[idx, 'beta_pos'] = beta_pos
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
    """Calculate comprehensive benchmark statistics."""
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

# def run_strategy(pairs: list[str], trading_start: str, trading_end: str, interval: str,
#                  window_in_steps: int, z_score_method: str, entry_threshold: float = None, exit_threshold: float = None,
#                  position_size: float = None, initial_cash: float = 1000000, fee_rate: float = 0) -> Portfolio:
#     """Run pair trading strategy on a list of pairs."""
#     n = len(pairs)
#     portfolio = Portfolio(start=trading_start, end=trading_end, interval=interval, fee_rate=fee_rate)
#     pt_start = pre_training_start(start=trading_start, interval=interval, rolling_window_steps=window_in_steps)
#
#     agg_cols = ['fees_paid', 'realized_pnl', 'net_pnl']
#
#     for pair in pairs:
#         x, y = pair.split('-')
#         data = load_pair(x=x, y=y, start=pt_start, end=trading_end, interval=interval)
#
#         if z_score_method == 'rolling_beta':
#             calculate_rolling_zscore_with_rolling_beta(data, rolling_window=window_in_steps)
#         elif z_score_method == 'prices':
#             calculate_rolling_zscore(data, rolling_window=window_in_steps, source='prices')
#         elif z_score_method == 'returns':
#             calculate_rolling_zscore(data, rolling_window=window_in_steps, source='returns')
#         elif z_score_method == 'log_returns':
#             calculate_rolling_zscore(data, rolling_window=window_in_steps, source='log_returns')
#         elif z_score_method == 'cum_returns':
#             calculate_rolling_zscore(data, rolling_window=window_in_steps, source='cum_returns')
#         else:
#             raise ValueError(f"Unknown z_score_method: {z_score_method}")
#
#         data.start = trading_start
#
#         generate_action_space(pair_data=data, entry_threshold=entry_threshold, exit_threshold=exit_threshold,
#                               position_size=position_size)
#         generate_position_space(pair_data=data, position_size=position_size)
#         generate_trades(pair_data=data, initial_cash=initial_cash, fee_rate=fee_rate)
#
#         data.stats = calculate_stats(data, initial_cash)
#         portfolio.pairs_data.append(data)
#
#         if portfolio.data is None:
#             portfolio.data = data.data[agg_cols].copy()
#         else:
#             portfolio.data[agg_cols] += data.data[agg_cols]
#
#     if portfolio.data is not None:
#         total_initial_cash = initial_cash * n
#
#         portfolio.data['pnl_pct'] = portfolio.data['realized_pnl'] / total_initial_cash
#         portfolio.data['net_pnl_pct'] = portfolio.data['net_pnl'] / total_initial_cash
#
#         portfolio_stats = calculate_stats(portfolio, initial_cash)
#
#         for col in portfolio_stats.columns:
#             pair_metrics = [p.stats[col] for p in portfolio.pairs_data]
#
#             total_wins = sum(m['win_count'] for m in pair_metrics)
#             total_loses = sum(m['lose_count'] for m in pair_metrics)
#             total_trades = total_wins + total_loses
#
#             new_win_rate = total_wins / total_trades if total_trades > 0 else 0.0
#
#             max_win_pos = max((m['max_win_pos'] for m in pair_metrics), default=0.0) / n
#             max_lose_pos = min((m['max_lose_pos'] for m in pair_metrics), default=0.0) / n
#
#             total_ret = portfolio_stats.loc['total_return', col]
#             avg_trade_return = total_ret / total_trades if total_trades > 0 else 0.0
#
#             portfolio_stats.at['win_count', col] = total_wins
#             portfolio_stats.at['lose_count', col] = total_loses
#             portfolio_stats.at['win_rate', col] = new_win_rate
#             portfolio_stats.at['max_win_pos', col] = max_win_pos
#             portfolio_stats.at['max_lose_pos', col] = max_lose_pos
#             portfolio_stats.at['avg_trade_return', col] = avg_trade_return
#
#         portfolio.stats = portfolio_stats
#         portfolio.summary = calc_portfolio_summary(portfolio)
#
#     return portfolio
