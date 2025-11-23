import numpy as np
import pandas as pd

from modules.data_services.data_models import PairData, PortfolioData
from modules.data_services.data_pipeline import load_pair
from modules.data_services.data_utils import calc_portfolio_summary, pre_training_start, get_steps
from modules.data_services.z_score_calculation import calculate_rolling_zscore_with_rolling_beta, \
    calculate_rolling_zscore


def generate_action_space(pair_data: PairData, entry_threshold: float = None,
                          exit_threshold: float = None, position_size: float | None = None) -> PairData:
    """Generate action space with NaN protection."""
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

        if pd.isna(zscore_i):
            action.iloc[i] = 0
            continue

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
            action.iloc[i] = action.iloc[i - 1]

    df['action'] = action
    pair_data.data = df

    return pair_data


def generate_position_space(pair_data, position_size: float | None = None) -> object:
    df = pair_data.data.copy()

    if 'position' not in df.columns:
        if 'action' in df.columns:
            df['position'] = df['action']
        else:
            df['position'] = 0

    if 'position_size' in df.columns:
        pass
    elif position_size is not None:
        df['position_size'] = position_size
    else:
        df['position_size'] = 1.0

    has_coint_metrics = all(col in df.columns for col in ['beta', 'mean', 'std'])

    virtual_z_curve = []
    weight_x_curve = []
    weight_y_curve = []

    prev_pos = 0
    entry_beta = 0.0
    entry_mean = 0.0
    entry_std = 0.0

    for i in range(len(df)):
        pos = df['position'].iloc[i]
        pos_size = df['position_size'].iloc[i]

        raw_z = df['z_score'].iloc[i] if 'z_score' in df.columns else 0.0
        virt_z = raw_z

        w_x = pos_size / 2
        w_y = pos_size / 2

        if has_coint_metrics:
            curr_beta = df['beta'].iloc[i]
            curr_mean = df['mean'].iloc[i]
            curr_std = df['std'].iloc[i]

            is_new_entry = (prev_pos == 0 and pos != 0)
            is_reversal = (prev_pos != 0 and pos != 0 and pos != prev_pos)

            # SNAPSHOT
            if is_new_entry or is_reversal:
                entry_beta = curr_beta
                entry_mean = curr_mean
                entry_std = curr_std

            price_x = df[pair_data.x].iloc[i]
            price_y = df[pair_data.y].iloc[i]

            # Virtual Z-Score
            if pos != 0:
                virtual_spread = price_x - (entry_beta * price_y)
                if entry_std != 0:
                    virt_z = (virtual_spread - entry_mean) / entry_std
                else:
                    virt_z = 0.0

            # Static Beta-Hedge
            beta_for_sizing = entry_beta if pos != 0 else curr_beta

            # Weights
            if beta_for_sizing < 0:
                w_x = w_y = 0
            else:
                hedged_price_y = beta_for_sizing * price_y
                w_x = price_x / (price_x + hedged_price_y)
                w_y = hedged_price_y / (price_x + hedged_price_y)

        virtual_z_curve.append(virt_z)
        weight_x_curve.append(w_x)
        weight_y_curve.append(w_y)

        prev_pos = pos

    df['z_score_virtual'] = virtual_z_curve
    df['weight_x'] = weight_x_curve
    df['weight_y'] = weight_y_curve

    pair_data.data = df
    return pair_data


def generate_trades(pair_data: PairData, initial_cash: float, fee_rate: float = 0) -> PairData:
    """
    Simulate trading for a pair using 'position' and 'position_size' columns.
    Calculates fees on both ENTRY and EXIT.
    Fixed capital model: always trading the same amount.
    Fee and PnL tracking is separate from the cash in the portfolio.
    """
    df = pair_data.data.copy()
    x_col, y_col = pair_data.x, pair_data.y

    required = [x_col, y_col, 'position', 'weight_x', 'weight_y']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Run generate_position_space first.")

    realized_pnl = 0.0
    total_fees = 0.0
    prev_pos = 0
    q_x = q_y = 0.0
    entry_price_x = entry_price_y = None

    pnl_curve = []
    fee_curve = []
    equity_curve = []

    trading_stopped = False

    for i in range(len(df)):
        price_x = df[x_col].iloc[i]
        price_y = df[y_col].iloc[i]

        pos = df['position'].iloc[i]
        w_x = df['weight_x'].iloc[i] * df['position_size'].iloc[i]
        w_y = df['weight_y'].iloc[i] * df['position_size'].iloc[i]

        # END OF TRADING (Stop Out)
        if trading_stopped:
            pos = 0

        # IN POSITION
        if prev_pos != 0:
            current_pnl = (q_x * (price_x - entry_price_x)) + (q_y * (price_y - entry_price_y))
            equity = realized_pnl + current_pnl - total_fees
            virt_z = df['z_score_virtual'].iloc[i]
            if abs(virt_z) < 0.1:
                pos = 0
        else:
            equity = realized_pnl - total_fees

        # REVERSE (CLOSE OLD, OPEN NEW)
        if prev_pos != 0 and pos != 0 and pos != prev_pos:
            # Close Old
            exit_val = abs(q_x * price_x) + abs(q_y * price_y)
            total_fees += exit_val * fee_rate
            realized_pnl += q_x * (price_x - entry_price_x) + q_y * (price_y - entry_price_y)

            # Open New
            invested = initial_cash
            if pos == 1:  # Long Spread (Long X, Short Y)
                q_x = invested * w_x / price_x
                q_y = -(invested * w_y) / price_y
            else:  # Short Spread (Short X, Long Y)
                q_x = -(invested * w_x) / price_x
                q_y = invested * w_y / price_y

            entry_price_x = price_x
            entry_price_y = price_y

            entry_val = abs(q_x * price_x) + abs(q_y * price_y)
            total_fees += entry_val * fee_rate
            prev_pos = pos

        # OPEN POSITION
        elif prev_pos == 0 and pos != 0:
            invested = initial_cash
            if pos > 0:
                q_x = invested * w_x / price_x
                q_y = -(invested * w_y) / price_y
            else:
                q_x = -(invested * w_x) / price_x
                q_y = invested * w_y / price_y

            entry_price_x = price_x
            entry_price_y = price_y

            entry_val = abs(q_x * price_x) + abs(q_y * price_y)
            total_fees += entry_val * fee_rate
            prev_pos = pos

        # CLOSE POSITION
        elif prev_pos != 0 and pos == 0:
            exit_val = abs(q_x * price_x) + abs(q_y * price_y)
            total_fees += exit_val * fee_rate
            realized_pnl += (q_x * (price_x - entry_price_x)) + (q_y * (price_y - entry_price_y))

            q_x = q_y = 0.0
            prev_pos = 0

        pnl_curve.append(realized_pnl)
        fee_curve.append(total_fees)
        equity_curve.append(equity)

        if equity <= -initial_cash and not trading_stopped:
            trading_stopped = True

    df['fees_paid'] = fee_curve
    df['equity'] = equity_curve
    df['realized_pnl'] = pnl_curve
    df['net_pnl'] = df['realized_pnl'] - df['fees_paid']
    df['realized_pnl_pct'] = 1 + df['realized_pnl'] / initial_cash
    df['net_pnl_pct'] = 1 + df['net_pnl'] / initial_cash

    pair_data.data = df
    pair_data.fee_rate = fee_rate

    return pair_data


def calculate_stats(pair_data: PairData, initial_cash: float) -> pd.DataFrame:
    """Calculate comprehensive benchmark statistics."""
    df = pair_data.data.copy()
    fee_rate = pair_data.fee_rate

    steps_per_day = get_steps(pair_data.interval)
    periods_per_year = steps_per_day * 365

    def compute_stats(pnl_series):
        pnl_series_limited = pnl_series.clip(lower=-initial_cash)
        equity_curve = pnl_series_limited + initial_cash
        returns = equity_curve.pct_change().fillna(0)

        trades_pnl = pnl_series_limited.diff().fillna(0)
        trades_pnl = trades_pnl[trades_pnl != 0]

        total_pnl = pnl_series_limited.iloc[-1]
        total_return = total_pnl / initial_cash

        total_trades = len(trades_pnl)
        winning_trades = trades_pnl[trades_pnl > 0]
        losing_trades = trades_pnl[trades_pnl < 0]

        win_count = len(winning_trades)
        lose_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        max_win_pos = (winning_trades.max() / initial_cash) if not winning_trades.empty else 0.0
        max_lose_pos = (losing_trades.min() / initial_cash) if not losing_trades.empty else 0.0

        avg_trade_return = (trades_pnl.mean() / initial_cash) if total_trades > 0 else 0.0

        period_volatility = returns.std()

        if pd.isna(period_volatility):
            period_volatility = 0.0

        annualized_volatility = period_volatility * np.sqrt(periods_per_year)

        if period_volatility == 0:
            annualized_sharpe = None
        else:
            annualized_sharpe = returns.mean() / period_volatility * np.sqrt(periods_per_year)

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        if pd.isna(downside_std) or downside_std == 0:
            sortino_ratio = None
        else:
            sortino_ratio = returns.mean() / downside_std * np.sqrt(periods_per_year)

        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max.replace(0, 1)
        max_drawdown = drawdown.min()

        years = len(df) / periods_per_year if len(df) > 0 else 0
        if years > 0 and equity_curve.iloc[-1] > 0 and equity_curve.iloc[0] > 0:
            cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
        else:
            cagr = 0.0

        if max_drawdown == 0:
            calmar_ratio = None
        else:
            calmar_ratio = (cagr / abs(max_drawdown))

        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": period_volatility,
            "annual_volatility": annualized_volatility,
            "max_drawdown": max_drawdown,
            "win_count": win_count,
            "lose_count": lose_count,
            "win_rate": win_rate,
            "max_win_pos": max_win_pos,
            "max_lose_pos": max_lose_pos,
            "avg_trade_return": avg_trade_return,
            "sharpe_ratio": annualized_sharpe,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }

    brutto_stats = compute_stats(df["realized_pnl"])
    netto_stats = compute_stats(df["net_pnl"])

    metrics_order = [
        "total_return", "cagr", "volatility", "annual_volatility", "max_drawdown", "win_count", "lose_count",
        "win_rate", "max_win_pos", "max_lose_pos", "avg_trade_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio"
    ]

    stats_df = pd.DataFrame({
        "metric": metrics_order,
        "0% fee": [brutto_stats[m] for m in metrics_order],
        f"{fee_rate * 100}% fee": [netto_stats[m] for m in metrics_order]
    }).set_index("metric")

    return stats_df


def run_strategy(pairs: list[str], trading_start: str, trading_end: str, interval: str,
                 window_in_steps: int, z_score_method: str, entry_threshold: float = None, exit_threshold: float = None,
                 position_size: float = None, initial_cash: float = 1000000, fee_rate: float = 0) -> PortfolioData:
    """Run pair trading strategy on a list of pairs."""
    n = len(pairs)
    portfolio = PortfolioData(start=trading_start, end=trading_end, interval=interval, fee_rate=fee_rate)
    pt_start = pre_training_start(start=trading_start, interval=interval, rolling_window_steps=window_in_steps)

    agg_cols = ['fees_paid', 'realized_pnl', 'net_pnl']

    for pair in pairs:
        x, y = pair.split('-')
        data = load_pair(x=x, y=y, start=pt_start, end=trading_end, interval=interval)

        if z_score_method == 'rolling_beta':
            calculate_rolling_zscore_with_rolling_beta(data, rolling_window=window_in_steps)
        elif z_score_method == 'prices':
            calculate_rolling_zscore(data, rolling_window=window_in_steps, source='prices')
        elif z_score_method == 'returns':
            calculate_rolling_zscore(data, rolling_window=window_in_steps, source='returns')
        elif z_score_method == 'log_returns':
            calculate_rolling_zscore(data, rolling_window=window_in_steps, source='log_returns')
        elif z_score_method == 'cum_returns':
            calculate_rolling_zscore(data, rolling_window=window_in_steps, source='cum_returns')
        else:
            raise ValueError(f"Unknown z_score_method: {z_score_method}")

        data.start = trading_start

        generate_action_space(pair_data=data, entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                              position_size=position_size)
        generate_position_space(pair_data=data, position_size=position_size)
        generate_trades(pair_data=data, initial_cash=initial_cash, fee_rate=fee_rate)

        data.stats = calculate_stats(data, initial_cash)
        portfolio.pairs_data.append(data)

        if portfolio.data is None:
            portfolio.data = data.data[agg_cols].copy()
        else:
            portfolio.data[agg_cols] += data.data[agg_cols]

    if portfolio.data is not None:
        total_initial_cash = initial_cash * n

        portfolio.data['pnl_pct'] = portfolio.data['realized_pnl'] / total_initial_cash
        portfolio.data['net_pnl_pct'] = portfolio.data['net_pnl'] / total_initial_cash

        portfolio_stats = calculate_stats(portfolio, initial_cash)

        for col in portfolio_stats.columns:
            pair_metrics = [p.stats[col] for p in portfolio.pairs_data]

            total_wins = sum(m['win_count'] for m in pair_metrics)
            total_loses = sum(m['lose_count'] for m in pair_metrics)
            total_trades = total_wins + total_loses

            new_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

            max_win_pos = max((m['max_win_pos'] for m in pair_metrics), default=0.0) / n
            max_lose_pos = min((m['max_lose_pos'] for m in pair_metrics), default=0.0) / n

            total_ret = portfolio_stats.loc['total_return', col]
            avg_trade_return = total_ret / total_trades if total_trades > 0 else 0.0

            portfolio_stats.at['win_count', col] = total_wins
            portfolio_stats.at['lose_count', col] = total_loses
            portfolio_stats.at['win_rate', col] = new_win_rate
            portfolio_stats.at['max_win_pos', col] = max_win_pos
            portfolio_stats.at['max_lose_pos', col] = max_lose_pos
            portfolio_stats.at['avg_trade_return', col] = avg_trade_return

        portfolio.stats = portfolio_stats
        portfolio.summary = calc_portfolio_summary(portfolio)

    return portfolio
