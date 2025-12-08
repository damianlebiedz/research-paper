"""Calculate stats for the Pair."""
import numpy as np
import pandas as pd

from modules.core.models import Pair
from modules.data_services.data_utils import get_steps


def calculate_stats(pair: Pair, risk_free_rate_annual: float) -> pd.DataFrame:
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
        period_rf = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
        sharpe_ratio = (returns.mean() - period_rf) / period_volatility if period_volatility != 0 else None
        sharpe_ratio_annual = (cagr - risk_free_rate_annual) / annualized_volatility if annualized_volatility != 0 else None

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
