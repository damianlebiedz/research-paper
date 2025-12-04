from functools import partial
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import RISK_FREE_ANNUAL
from modules.data_services.data_loaders import load_pair
from modules.data_services.data_models import Pair
from modules.data_services.data_utils import get_steps, add_returns
from modules.performance.param_optimization import bayesian_optimization
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


def generate_trade(x: str, y: str, z_score: float, beta: float, pair: Pair, position_state: PositionState,
                   strategy_params: StrategyParams, price_x: float, price_y: float, total_fees: float,
                   is_spread: bool) -> tuple[float, float]:
    prev_position = position_state.prev_position
    q_x = position_state.q_x
    q_y = position_state.q_y

    exit_threshold = strategy_params.exit_threshold
    stop_loss = strategy_params.stop_loss
    fee_rate = pair.fee_rate
    initial_cash = pair.initial_cash

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
        # CLOSE POSITION (STOP LOSS OR TAKE PROFIT)
        if (
                prev_position < 0 and (
                z_score <= exit_threshold or (
                position_state.stop_loss_threshold is not None and z_score >= position_state.stop_loss_threshold))) or (
                prev_position > 0 and (
                z_score >= -exit_threshold or (
                position_state.stop_loss_threshold is not None and z_score <= -position_state.stop_loss_threshold))
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
            position_state.update_position(position_state.position, prev_position, q_x, q_y, w_x, w_y, entry_val,
                                           stop_loss_threshold)
            pnl = 0
        # STAY OUT OF POSITION
        else:
            pnl = 0
    return pnl, total_fees


def calculate_beta_returns(x_returns: str, y_returns: str, df: pd.DataFrame) -> float:
    X = sm.add_constant(df[y_returns])
    y = df[x_returns]
    model = sm.OLS(y, X, missing="drop").fit()

    beta = model.params[y_returns]
    return beta


def calculate_zscore_prices(x_price: str, y_price: str, beta: float, df: pd.DataFrame) -> float:
    spread_series = df[x_price] - (beta * df[y_price])
    mean = spread_series.mean()
    std = spread_series.std()
    spread = spread_series.iloc[-1]
    if std == 0:
        return None

    z_score = (spread - mean) / std
    return z_score


def run_strategy(pair: Pair, rolling_window: int, entry_threshold: float = None, exit_threshold: float = None,
                 stop_loss: float = None, pos_size: float = None, beta_hedge: bool = False,
                 is_spread: bool = False) -> Pair:
    df = pair.data.copy()
    x_col, y_col = pair.x, pair.y
    initial_cash = pair.initial_cash

    total_fees = 0.0
    total_pnl = 0.0
    prev_pnl = 0.0

    position_state = PositionState()
    strategy_params = StrategyParams

    if pair.test_start is not None:
        start_pos = df.index.get_loc(pd.to_datetime(pair.test_start))
    else:
        raise ValueError("Test start must be set to run the strategy")
    if start_pos - rolling_window + 1 < 0:
        raise ValueError("Rolling window cannot be bigger than pre-training period")
    df = df.iloc[start_pos - rolling_window + 1:]
    first_pos = rolling_window - 1
    last_pos = df.index.get_loc(pd.to_datetime(pair.end))

    for i in range(first_pos, last_pos + 1):
        if total_pnl == -initial_cash:
            # BANKRUPT
            df = df.iloc[:i].copy()
            break
        else:
            prev_pos = position_state.prev_position

            price_x = df[x_col].iloc[i]
            price_y = df[y_col].iloc[i]

            if all(x is None for x in [entry_threshold, exit_threshold, stop_loss, pos_size, rolling_window]):
                # TODO: Agent
                entry_threshold = ...  # [-inf,+inf]
                exit_threshold = ...  # [-inf,+inf]
                stop_loss = ...  # > entry_threshold

            strategy_params.entry_threshold = entry_threshold
            strategy_params.exit_threshold = exit_threshold
            strategy_params.stop_loss = stop_loss

            if beta_hedge:
                beta = calculate_beta_returns(
                    f"{x_col}_returns", f"{y_col}_returns", df.iloc[i - rolling_window + 1:i + 1]
                )
            else:
                beta = 1
            z_score = calculate_zscore_prices(
                x_col, y_col, beta, df.iloc[i - rolling_window + 1:i + 1]
            )

            signal = generate_signal(entry_threshold, z_score)

            if pos_size is None:
                if prev_pos == 0 and signal != 0:
                    # TODO: Agent
                    pos_size = ...  # [-1,1]

            if beta is not None and beta >= 0:
                position_state.position = signal * pos_size

            strategy_params.pos_size = pos_size

            pnl, total_fees = generate_trade(
                x_col, y_col, z_score, beta, pair, position_state, strategy_params, price_x, price_y, total_fees,
                is_spread
            )

            if pnl != 0:
                total_pnl = pnl + prev_pnl
            else:
                prev_pnl = total_pnl

        if total_pnl <= -initial_cash:
            total_pnl = -initial_cash

        idx = df.index[i]
        df.at[idx, 'z_score'] = z_score
        df.at[idx, 'beta'] = beta
        df.at[idx, 'entry_thr'] = strategy_params.entry_threshold
        df.at[idx, 'exit_thr'] = strategy_params.exit_threshold
        df.at[idx, 'sl_thr'] = position_state.stop_loss_threshold
        df.at[idx, 'w_x'] = position_state.w_x
        df.at[idx, 'w_y'] = position_state.w_y
        df.at[idx, 'q_x'] = position_state.q_x
        df.at[idx, 'q_y'] = position_state.q_y
        # df.at[idx, 'cash'] = initial_cash - position_state.entry_val
        # df.at[idx, 'signal'] = signal
        # df.at[idx, 'prev_position'] = position_state.prev_position
        df.at[idx, 'position'] = position_state.position
        df.at[idx, 'total_return'] = total_pnl
        df.at[idx, 'total_fees'] = total_fees
        df.at[idx, 'net_return'] = total_pnl - total_fees

        position_state.prev_position = position_state.position

    df['total_return_pct'] = df['total_return'] / initial_cash
    df['net_return_pct'] = df['net_return'] / initial_cash

    pair.data = df[rolling_window - 1:].drop(
        columns=[f'{x_col}_returns', f'{y_col}_returns', f'{x_col}_log_returns', f'{y_col}_log_returns']).round(4)
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

        # CAGR (Compound Annual Growth Rate)
        years = len(df) / periods_per_year if len(df) > 0 else 0
        cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) if years > 0 and equity_curve.iloc[
            0] > 0 else 0.0

        # Sharpe ratio
        period_rf = (1 + RISK_FREE_ANNUAL) ** (1 / periods_per_year) - 1
        sharpe_ratio = (returns.mean() - period_rf) / period_volatility if period_volatility != 0 else None
        sharpe_ratio_annual = (cagr - RISK_FREE_ANNUAL) / annualized_volatility if annualized_volatility != 0 else None

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std if downside_std and not pd.isna(downside_std) else None
        sortino_ratio_annual = sortino_ratio * np.sqrt(periods_per_year) if sortino_ratio is not None else None

        # Maximum drawdown
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else None
        calmar_ratio_annual = cagr / abs(max_drawdown) if max_drawdown != 0 else None

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
            "avg_win_return": avg_win_trade_pct,
            "avg_lose_return": avg_lose_trade_pct,
            "avg_trade_return": avg_trade_ret_pct,
            "sharpe_ratio": sharpe_ratio,
            "sharpe_ratio_annual": sharpe_ratio_annual,
            "sortino_ratio": sortino_ratio,
            "sortino_ratio_annual": sortino_ratio_annual,
            "calmar_ratio": calmar_ratio,
            "calmar_ratio_annual": calmar_ratio_annual,
        }

    brutto_stats = compute_stats(df["total_return"])
    netto_stats = compute_stats(df["net_return"])

    metrics_order = [
        "total_return", "cagr", "volatility", "volatility_annual", "max_drawdown", "win_count", "lose_count",
        "win_rate", "max_win", "max_lose", "avg_win_return", "avg_lose_return", "avg_trade_return", "sharpe_ratio",
        "sharpe_ratio_annual", "sortino_ratio", "sortino_ratio_annual", "calmar_ratio", "calmar_ratio_annual"
    ]

    stats_df = pd.DataFrame({
        "metric": metrics_order,
        "0% fee": [brutto_stats[m] for m in metrics_order],
        f"{fee_rate * 100}% fee": [netto_stats[m] for m in metrics_order]
    }).set_index("metric")

    stats_df = stats_df.round(4)
    return stats_df


def strategy_wrapper(rolling_window: int, entry_threshold: float, exit_threshold: float, stop_loss: float,
                     ticker_x: str, ticker_y: str, fee_rate: float, initial_cash: float, position_size: float,
                     pre_trading_start: str, trading_start: str, trading_end: str, interval: str, metric: tuple,
                     beta_hedge: bool, is_spread: bool) -> float:
    try:
        pair = load_pair(x=ticker_x, y=ticker_y, start=pre_trading_start, end=trading_end, interval=interval)
        add_returns(pair)

        pair.test_start = trading_start
        pair.fee_rate = fee_rate
        pair.initial_cash = initial_cash

        run_strategy(
            pair, rolling_window, entry_threshold, exit_threshold, stop_loss, position_size, beta_hedge, is_spread
        )

        pair.stats = calculate_stats(pair)
        score = pair.stats.loc[metric]

        if isinstance(score, pd.Series):
            score = score.iloc[0]
        if pd.isna(score):
            return 0.0
        return score

    except Exception as e:
        print("Error in strategy run:", e)
        return -1e9


def calc_bayesian_params(ticker_x: str, ticker_y: str, fee_rate: float, initial_cash: float, position_size: float,
                         pre_training_start: str, training_start: str, training_end: str, interval: str, beta_hedge: bool,
                         is_spread: bool, param_space: list, metric: tuple = ("sortino_ratio_annual", "0.05% fee"),
                         minimize: bool = False) -> tuple[dict, float]:
    static_params = {
        "ticker_x": ticker_x,
        "ticker_y": ticker_y,
        "fee_rate": fee_rate,
        "initial_cash": initial_cash,
        "position_size": position_size,
        "pre_trading_start": pre_training_start,
        "trading_start": training_start,
        "trading_end": training_end,
        "interval": interval,
    }

    wrapped_strategy = partial(
        strategy_wrapper,
        beta_hedge=beta_hedge,
        is_spread=is_spread,
    )

    best_params, best_score, results = bayesian_optimization(
        strategy_func=wrapped_strategy,
        param_space=param_space,
        static_params=static_params,
        metric=metric,
        minimize=minimize,
    )
    return best_params, best_score
