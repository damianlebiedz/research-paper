from functools import partial
import pandas as pd

from modules.core.execution import TradeExecutor
from modules.core.indicators import calculate_beta_returns, calculate_zscore_prices, generate_signal
from modules.data_services.data_loaders import load_pair
from modules.core.models import Pair
from modules.data_services.data_utils import add_returns
from modules.performance.optimization import random_search
from modules.core.models import PositionState, StrategyParams
from modules.performance.stats import calculate_stats


def single_pair_strategy(pair: Pair, rolling_window: int, entry_threshold: float = None, exit_threshold: float = None,
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

            if beta is not None and beta >= 0: # Execute trade only if beta exists and is >= 0
                position_state.position = signal * pos_size

            strategy_params.pos_size = pos_size

            pnl, total_fees = TradeExecutor.execute(
                pair, position_state, strategy_params, price_x, price_y, z_score, beta, total_fees, is_spread
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


def run_single_pair_strategy(rolling_window: int, entry_threshold: float, exit_threshold: float, stop_loss: float,
                             ticker_x: str, ticker_y: str, fee_rate: float, initial_cash: float, position_size: float,
                             pre_trading_start: str, trading_start: str, trading_end: str, interval: str,
                             beta_hedge: bool, is_spread: bool, risk_free_rate_annual: float) -> Pair:
    pair = load_pair(x=ticker_x, y=ticker_y, start=pre_trading_start, end=trading_end, interval=interval)
    add_returns(pair)
    pair.test_start = trading_start
    pair.fee_rate = fee_rate
    pair.initial_cash = initial_cash
    single_pair_strategy(
        pair, rolling_window, entry_threshold, exit_threshold, stop_loss, position_size, beta_hedge, is_spread
    )
    pair.stats = calculate_stats(pair, risk_free_rate_annual)
    return pair


def strategy_wrapper(rolling_window: int, entry_threshold: float, exit_threshold: float, stop_loss: float,
                     ticker_x: str, ticker_y: str, fee_rate: float, initial_cash: float, position_size: float,
                     pre_trading_start: str, trading_start: str, trading_end: str, interval: str, metric: tuple,
                     beta_hedge: bool, is_spread: bool, risk_free_rate_annual: float) -> float:
    try:
        pair = run_single_pair_strategy(rolling_window, entry_threshold, exit_threshold, stop_loss, ticker_x, ticker_y,
                                        fee_rate, initial_cash, position_size, pre_trading_start, trading_start,
                                        trading_end, interval, beta_hedge, is_spread, risk_free_rate_annual)
        score = pair.stats.loc[metric]

        if isinstance(score, pd.Series):
            score = score.iloc[0]
        if pd.isna(score):
            return 0.0
        return score

    except Exception as e:
        print("Error in strategy run:", e)
        return -1e9


def optimize_params(ticker_x: str, ticker_y: str, fee_rate: float, initial_cash: float, position_size: float,
                    pre_training_start: str, training_start: str, training_end: str, interval: str,
                    beta_hedge: bool, is_spread: bool, risk_free_rate_annual: float, param_space: list,
                    metric: tuple = ("sortino_ratio_annual", "0.05% fee")) -> tuple[dict, float]:
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
        risk_free_rate_annual=risk_free_rate_annual,
    )

    best_params, best_score = random_search(
        strategy_func=wrapped_strategy,
        param_space=param_space,
        static_params=static_params,
        metric=metric,
    )
    return best_params, best_score
