from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress

from modules.data_services.data_models import Pair
from modules.data_services.data_utils import get_steps
from modules.performance.data_models import PositionState, StrategyParams


def generate_signal(entry_threshold: float, z_score: float) -> Pair:
    signal = 0
    if z_score is not None:
        if z_score <= -entry_threshold:
            signal = 1  # Long X Short Y
        elif z_score >= entry_threshold:
            signal = -1  # Short X Long Y
    return signal


def get_spread(x: str, y: str, position: float) -> tuple[float, float]:  # TODO
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


def generate_trade(x: str, y: str, position_state: PositionState, strategy_params: StrategyParams, price_x: float,
                   price_y: float, total_fees: float, is_spread: bool) -> tuple[float, float]:
    z_score = position_state.z_score
    alpha = position_state.alpha
    beta = position_state.beta
    mean = position_state.mean
    std = position_state.std
    prev_position = position_state.prev_position
    q_x = position_state.q_x
    q_y = position_state.q_y

    exit_threshold = strategy_params.exit_threshold
    stop_loss = strategy_params.stop_loss
    fee_rate = strategy_params.fee_rate
    initial_cash = strategy_params.initial_cash

    def generate_virtual_z_score() -> tuple[Optional[float], float]:
        s_virt = price_x - (alpha + beta * price_y)
        if position_state.std != 0:
            return (s_virt - mean) / std, s_virt
        return None, s_virt

    def open_position() -> tuple[float, float, float, float, float, float, float]:
        wx = 1 / (beta + 1)
        wy = 1 - wx
        position_cash = abs(position_state.position) * initial_cash
        x_spread, y_spread = get_spread(x, y, position_state.position) if is_spread else 1, 1

        if position_state.position > 0:
            qx = position_cash * wx / (price_x * x_spread)
            qy = -(position_cash * wy) / (price_y * y_spread)
        elif position_state.position < 0:
            qx = -(position_cash * wx) / (price_x * x_spread)
            qy = position_cash * wy / (price_y * y_spread)
        else:
            raise ValueError("Position cannot be 0 while opening")

        entry_value = abs(qx) * price_x + abs(qy) * price_y
        pos_fees = entry_value * fee_rate
        t_fees = total_fees + pos_fees
        stop_loss_thr = abs(z_score * stop_loss)
        return qx, qy, wx, wy, entry_value, stop_loss_thr, t_fees

    def close_position() -> tuple[float, float]:
        x_spread, y_spread = get_spread(x, y, 0) if is_spread else 1, 1
        exit_value = abs(q_x) * (price_x * x_spread) + abs(q_y) * (price_y * y_spread)
        pos_fees = exit_value * fee_rate
        if position_state.prev_position > 0:
            pos_pnl = exit_value - position_state.entry_val
        elif position_state.prev_position < 0:
            pos_pnl = position_state.entry_val - exit_value
        else:
            raise ValueError("Position cannot be 0 while closing")
        t_fees = total_fees + pos_fees
        return pos_pnl, t_fees

    # IN POSITION
    if prev_position != 0:
        z_score_virt, spread_virt = generate_virtual_z_score()
        position_state.z_score = z_score_virt
        position_state.spread = spread_virt

        # CLOSE POSITION (STOP LOSS OR TAKE PROFIT)
        if (
                prev_position < 0 and (
                z_score_virt <= exit_threshold or (
                position_state.stop_loss_threshold is not None and z_score_virt >= position_state.stop_loss_threshold))) or (
                prev_position > 0 and (
                z_score_virt >= -exit_threshold or (
                position_state.stop_loss_threshold is not None and z_score_virt <= -position_state.stop_loss_threshold))
        ):
            pnl, total_fees = close_position()
            position_state.clear_position()

        # STAY IN POSITION
        else:
            exit_val = abs(q_x) * price_x + abs(q_y) * price_y
            if prev_position > 0:
                pnl = exit_val - position_state.entry_val
            else:
                pnl = position_state.entry_val - exit_val
            position_state.position = prev_position

    # OUT OF POSITION
    else:

        # OPEN POSITION
        if position_state.position != 0:
            q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold, total_fees = open_position()
            position_state.update_position(position_state.position, prev_position, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold)
            pnl = 0

        # STAY OUT OF POSITION
        else:
            pnl = 0

    return pnl, total_fees


def calculate_rolling_zscore(col_x, col_y, df: pd.DataFrame):
    X = sm.add_constant(df[col_y])
    y = df[col_x]

    model = sm.OLS(y, X, missing='drop').fit()
    alpha = model.params['const']
    beta = model.params[col_y]

    spread_col = df[col_x] - (alpha + beta * df[col_y])
    mean = spread_col.mean()
    std = spread_col.std()
    spread = (spread_col.iloc[-1])

    if std == 0:
        return None, spread, alpha, beta, mean, std

    z_score = (spread - mean) / std
    return z_score, spread, alpha, beta, mean, std


def run_strategy(pair: Pair, rolling_window: int, entry_threshold: float = None, exit_threshold: float = None,
                 stop_loss: float = None, pos_size: float = None, is_spread: bool = False,
                 static_hedge: bool = True) -> Pair:
    df = pair.data.copy()
    x_col, y_col = pair.x, pair.y
    initial_cash = pair.initial_cash

    total_fees = 0.0
    total_pnl = 0.0
    prev_pnl = 0.0

    position_state = PositionState()
    strategy_params = StrategyParams

    strategy_params.fee_rate = pair.fee_rate
    strategy_params.initial_cash = initial_cash

    for i in range(rolling_window - 1, len(df)):
        prev_pos = position_state.prev_position

        if total_pnl == -initial_cash:
            # BANKRUPT
            df = df.iloc[:i].copy()
            break
        else:
            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            if all(x is None for x in [entry_threshold, exit_threshold, stop_loss, pos_size, rolling_window]):
                # TODO: Agent
                entry_threshold = ...  # [-inf,+inf]
                exit_threshold = ...  # [-inf,+inf]
                stop_loss = ...  # > entry_threshold

            z_score, spread, alpha, beta, mean, std = calculate_rolling_zscore(
                x_col, y_col, df.iloc[i - rolling_window + 1:i + 1])

            signal = generate_signal(entry_threshold, z_score)

            if pos_size is None:
                if prev_pos == 0 and signal != 0:
                    # TODO: Agent
                    pos_size = ...  # [-1,1]

            if static_hedge:
                if prev_pos == 0 and signal != 0 and beta >= 0:
                    position_state.position = signal * pos_size
                    position_state.update_hedge_if_none(z_score, spread, alpha, beta, mean, std)
            else:
                position_state.position = signal * pos_size
                position_state.update_hedge_if_none(z_score, spread, alpha, beta, mean, std)

            strategy_params.exit_threshold = exit_threshold
            strategy_params.stop_loss = stop_loss
            strategy_params.pos_size = pos_size

            pnl, total_fees = generate_trade(
                x_col, y_col, position_state, strategy_params, price_x, price_y, total_fees, is_spread)

            if pnl != 0:
                total_pnl = pnl + prev_pnl
            else:
                prev_pnl = total_pnl

        if total_pnl <= -initial_cash:
            total_pnl = -initial_cash

        idx = df.index[i]
        df.at[idx, 'z_score'] = z_score
        # df.at[idx, 'spread'] = spread
        df.at[idx, 'alpha'] = alpha
        df.at[idx, 'beta'] = beta
        df.at[idx, 'mean'] = mean
        df.at[idx, 'std'] = std
        if static_hedge:
            df.at[idx, 'z_score_virtual'] = position_state.z_score
            # df.at[idx, 'spread_virt'] = position_state.spread
            df.at[idx, 'alpha_pos'] = position_state.alpha
            df.at[idx, 'beta_pos'] = position_state.beta
            df.at[idx, 'mean_pos'] = position_state.mean
            df.at[idx, 'std_pos'] = position_state.std
        df.at[idx, 'stop_loss_threshold'] = position_state.stop_loss_threshold
        df.at[idx, 'weight_x'] = position_state.w_x
        df.at[idx, 'weight_y'] = position_state.w_y
        df.at[idx, 'q_x'] = position_state.q_x
        df.at[idx, 'q_y'] = position_state.q_y
        df.at[idx, 'cash'] = initial_cash - position_state.entry_val
        df.at[idx, 'signal'] = signal
        df.at[idx, 'position'] = position_state.position
        # df.at[idx, 'prev_position'] = position_state.prev_position
        df.at[idx, 'total_pnl'] = total_pnl
        df.at[idx, 'total_fees'] = total_fees
        df.at[idx, 'net_pnl'] = total_pnl - total_fees

        if static_hedge:
            if position_state.position == 0 and prev_pos != 0:
                position_state.clear_hedge()
        else:
            position_state.clear_hedge()
        position_state.prev_position = position_state.position

    df['total_pnl_pct'] = df['total_pnl'] / initial_cash
    df['net_pnl_pct'] = df['net_pnl'] / initial_cash

    pair.data = df[rolling_window - 1:]
    return pair


def calculate_stats(pair: Pair) -> pd.DataFrame:
    df = pair.data.copy()
    fee_rate = pair.fee_rate
    initial_cash = pair.initial_cash

    steps_per_day = get_steps(pair.interval)
    periods_per_year = steps_per_day * 365

    def calc_trade_array(pnl_series: pd.Series, position_series: pd.Series) -> np.array:
        prev = 0
        open_idx = None
        trade_pnl = []

        for i in range(len(pnl_series)):
            pos = position_series.iloc[i]

            if prev == 0 and pos != 0:
                open_idx = i
            elif (prev < 0 <= pos) or (prev > 0 >= pos) or (prev != 0 and i == len(position_series) - 1):
                if open_idx is not None:
                    pnl = pnl_series.iloc[i] - pnl_series.iloc[open_idx]
                    trade_pnl.append(pnl)
                open_idx = i if pos != 0 else None
            prev = pos

        return np.array(trade_pnl)

    def compute_stats(pnl_series: pd.Series) -> dict:
        equity_curve = pnl_series + initial_cash
        returns = equity_curve.pct_change().dropna()

        total_pnl = pnl_series.iloc[-1]
        total_return = total_pnl / initial_cash

        trade_pnl = calc_trade_array(pnl_series, df['position'].dropna())

        # Total wins / Total losses
        total_wins = np.sum(trade_pnl > 0)
        total_losses = np.sum(trade_pnl < 0)

        # Win rate
        total_trades = total_wins + total_losses
        win_rate = total_wins / total_trades if total_trades > 0 else None

        # Max win / Max lose
        winning_trades = trade_pnl[trade_pnl > 0]
        losing_trades = trade_pnl[trade_pnl < 0]
        max_win_pct = winning_trades.max() / initial_cash if len(winning_trades) > 0 else None
        max_lose_pct = losing_trades.min() / initial_cash if len(losing_trades) > 0 else None

        # Avg win / Avg lose / Avg trade return
        avg_win_trade_pct = winning_trades.mean() / initial_cash if total_wins > 0 else 0
        avg_lose_trade_pct = losing_trades.mean() / initial_cash if total_losses > 0 else 0
        avg_trade_ret_pct = np.mean(trade_pnl) / initial_cash if total_trades > 0 else 0

        # Volatility
        period_volatility = returns.std() or 0.0
        annualized_volatility = period_volatility * np.sqrt(periods_per_year)

        # Sharpe ratio
        sharpe_ratio = returns.mean() / period_volatility if period_volatility != 0 else None
        sharpe_ratio_annual = sharpe_ratio * np.sqrt(periods_per_year) if sharpe_ratio is not None else None

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std if downside_std and not pd.isna(downside_std) else None
        sortino_ratio_annual = sortino_ratio * np.sqrt(periods_per_year) if sortino_ratio is not None else None

        # Maximum drawdown
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # CAGR (Compound Annual Growth Rate)
        years = len(df) / periods_per_year if len(df) > 0 else 0
        cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) if years > 0 and equity_curve.iloc[
            0] > 0 else 0.0

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else None
        calmar_ratio_annual = cagr / abs(max_drawdown) if max_drawdown != 0 else None

        # K-Ratio (Lars Kestner)
        x = np.arange(len(equity_curve))
        slope, intercept, r_value, p_value, std_err = linregress(x, equity_curve)
        k_ratio = slope / std_err if std_err != 0 else None

        # SQN (System Quality Number, Van Tharp)
        if len(trade_pnl) > 1:
            sqn = np.sqrt(len(trade_pnl)) * trade_pnl.mean() / trade_pnl.std()
        else:
            sqn = None

        # Downside SQN
        losing_trades = trade_pnl[trade_pnl < 0]
        downside_sqn = np.sqrt(len(trade_pnl)) * trade_pnl.mean() / losing_trades.std() if len(
            losing_trades) > 1 else None

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": period_volatility,
            "volatility_annual": annualized_volatility,
            "max_drawdown": max_drawdown,
            "win_count": int(total_wins),
            "lose_count": int(total_losses),
            "win_rate": win_rate,
            "max_win": max_win_pct,
            "max_lose": max_lose_pct,
            "avg_win_return": avg_win_trade_pct,
            "avg_lose_return": avg_lose_trade_pct,
            "avg_trade_return": avg_trade_ret_pct,
            "sharpe_ratio": sharpe_ratio,
            "sharpe_ratio_annual": sharpe_ratio_annual,
            "sortino_ratio": sortino_ratio,
            "sortino_ratio_annual": sortino_ratio_annual,
            "calmar_ratio": calmar_ratio,
            "calmar_ratio_annual": calmar_ratio_annual,
            "k_ratio": k_ratio,
            "sqn": sqn,
            "downside_sqn": downside_sqn,
        }

    brutto_stats = compute_stats(df["total_pnl"])
    netto_stats = compute_stats(df["net_pnl"])

    metrics_order = [
        "total_return", "cagr", "volatility", "volatility_annual", "max_drawdown", "win_count", "lose_count",
        "win_rate", "max_win", "max_lose", "avg_win_return", "avg_lose_return", "avg_trade_return", "sharpe_ratio",
        "sharpe_ratio_annual", "sortino_ratio", "sortino_ratio_annual", "calmar_ratio", "calmar_ratio_annual",
        "k_ratio", "sqn", "downside_sqn"
    ]

    stats_df = pd.DataFrame({
        "metric": metrics_order,
        "0% fee": [brutto_stats[m] for m in metrics_order],
        f"{fee_rate * 100}% fee": [netto_stats[m] for m in metrics_order]
    }).set_index("metric")

    stats_df = stats_df.round(4)
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
