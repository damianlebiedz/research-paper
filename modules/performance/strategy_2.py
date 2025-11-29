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
            signal = 1  # long x short y
        elif z_score >= entry_threshold:
            signal = -1  # short x long y
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
                   price_y: float, total_fees: float, is_spread: bool):
    z_score = position_state.z_score
    alpha = position_state.alpha
    beta = position_state.beta
    mean = position_state.mean
    std = position_state.std
    position = position_state.position
    prev_position = position_state.prev_position
    q_x = position_state.q_x
    q_y = position_state.q_y

    exit_threshold = strategy_params.exit_threshold
    stop_loss = strategy_params.stop_loss
    fee_rate = strategy_params.fee_rate
    initial_cash = strategy_params.initial_cash

    def generate_virtual_z_score():
        s_virt = price_x - (alpha + beta * price_y)
        if position_state.std != 0:
            return (s_virt - mean) / std, s_virt
        return None, s_virt

    def open_position():
        wx = 1 / (beta + 1)
        wy = 1 - wx
        position_cash = abs(position) * initial_cash
        x_spread, y_spread = get_spread(x, y, position) if is_spread else 1, 1

        if position > 0:
            qx = position_cash * wx / (price_x * x_spread)
            qy = -(position_cash * wy) / (price_y * y_spread)
        elif position < 0:
            qx = -(initial_cash * wx) / (price_x * x_spread)
            qy = initial_cash * wy / (price_y * y_spread)
        else:
            raise ValueError("Position cannot be 0 while opening")

        entry_value = abs(qx) * price_x + abs(qy) * price_y
        pos_fees = entry_value * fee_rate
        t_fees = total_fees + pos_fees
        stop_loss_thr = abs(z_score * stop_loss)
        return qx, qy, wx, wy, entry_value, stop_loss_thr, t_fees

    def close_position():
        x_spread, y_spread = get_spread(x, y, 0) if is_spread else 1, 1
        exit_value = abs(q_x) * (price_x * x_spread) + abs(q_y) * (price_y * y_spread)
        pos_fees = exit_value * fee_rate
        if position > 0:
            pos_pnl = exit_value - position_state.entry_val
        elif position < 0:
            pos_pnl = position_state.entry_val - exit_value
        else:
            raise ValueError("Position cannot be 0 while closing")
        t_fees = total_fees + pos_fees
        return pos_pnl, t_fees

    if prev_position != 0:
        # IN POSITION
        z_score_virt, spread_virt = generate_virtual_z_score()
        position_state.z_score = z_score_virt
        position_state.spread = spread_virt
        if position < 0:
            if z_score_virt <= exit_threshold or (
                    position_state.stop_loss_threshold is not None and z_score_virt >= position_state.stop_loss_threshold):
                # CLOSE POSITION
                pnl, total_fees = close_position()
                position_state.clear_position()
        else:
            if z_score_virt >= -exit_threshold or (
                    position_state.stop_loss_threshold is not None and z_score_virt <= -position_state.stop_loss_threshold):
                # CLOSE POSITION
                pnl, total_fees = close_position()
                position_state.clear_position()
        if prev_position == position:
            # STAY IN POSITION
            exit_val = abs(q_x) * price_x + abs(q_y) * price_y
            if position > 0:
                pnl = exit_val - position_state.entry_val
            else:
                pnl = position_state.entry_val - exit_val
            position_state.z_score = z_score_virt
            position_state.spread = spread_virt
            position_state.position = prev_position
        else:
            # REVERSE (CLOSE OLD, OPEN NEW)
            pnl, total_fees = close_position()
            q_x, q_y, w_x, w_y, entry_val, total_fees, stop_loss_threshold = open_position()
            position_state.update_position(position, prev_position, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold)
    else:
        if position != 0:
            # OPEN POSITION
            q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold, total_fees = open_position()
            position_state.update_position(position, prev_position, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold)
            pnl = 0
        else:
            # STAY OUT OF POSITION
            position_state.clear_hedge()
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
                 stop_loss: float = None, pos_size: float = None, is_spread: bool = False) -> Pair:
    df = pair.data.copy()
    x_col, y_col = pair.x, pair.y
    initial_cash = pair.initial_cash

    total_fees = 0.0
    total_pnl = 0.0
    prev_pnl = 0.0

    position_state = PositionState()
    strategy_params = StrategyParams

    prev_pos = position_state.prev_position
    strategy_params.fee_rate = pair.fee_rate
    strategy_params.initial_cash = initial_cash

    for i in range(rolling_window - 1, len(df)):
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
                if prev_pos == 0:
                    pos_size = ...  # [-1,1]

            z_score, spread, alpha, beta, mean, std = calculate_rolling_zscore(
                x_col, y_col, df.iloc[i - rolling_window + 1:i + 1])

            signal = generate_signal(entry_threshold, z_score)

            if prev_pos == 0 and signal != 0 and beta >= 0:
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

        position_state.prev_position = position_state.position
        prev_pos = position_state.prev_position

        idx = df.index[i]
        df.at[idx, 'z_score'] = z_score
        df.at[idx, 'spread'] = spread
        df.at[idx, 'alpha'] = alpha
        df.at[idx, 'beta'] = beta
        df.at[idx, 'mean'] = mean
        df.at[idx, 'std'] = std
        df.at[idx, 'z_score_virtual'] = None if (
                position_state.prev_position is None and position_state.position is not None) else position_state.z_score
        df.at[idx, 'spread_virt'] = None if (
                position_state.prev_position is None and position_state.position is not None) else position_state.spread
        df.at[idx, 'alpha_pos'] = None if (
                position_state.prev_position is None and position_state.position is not None) else position_state.alpha
        df.at[idx, 'beta_pos'] = None if (
                position_state.prev_position is None and position_state.position is not None) else position_state.beta
        df.at[idx, 'mean_pos'] = None if (
                position_state.prev_position is None and position_state.position is not None) else position_state.mean
        df.at[idx, 'std_pos'] = None if (
                position_state.prev_position is None and position_state.position is not None) else position_state.std
        df.at[idx, 'stop_loss_threshold'] = position_state.stop_loss_threshold
        df.at[idx, 'weight_x'] = position_state.w_x
        df.at[idx, 'weight_y'] = position_state.w_y
        df.at[idx, 'q_x'] = position_state.q_x
        df.at[idx, 'q_y'] = position_state.q_y
        df.at[idx, 'cash'] = initial_cash - position_state.entry_val
        df.at[idx, 'signal'] = signal
        df.at[idx, 'position'] = position_state.prev_position
        df.at[idx, 'total_pnl'] = total_pnl
        df.at[idx, 'total_fees'] = total_fees
        df.at[idx, 'net_pnl'] = total_pnl - total_fees

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

    def calc_trade_array(pnl_series, position_series):
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

    def compute_stats(pnl_series):
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
        max_win_pct = max(winning_trades) / initial_cash
        max_lose_pct = min(losing_trades) / initial_cash

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
