"""Generate plots for the Pair."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from modules.core.models import Pair


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def _resolve_results_dir(directory: str | None) -> Path:
    base = get_project_root() / "results"
    if directory:
        base = base / directory
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_zscore(pair_data: Pair, directory: str | None = None, save: bool = False, show: bool = True) -> None:
    x, y = pair_data.x, pair_data.y
    start, end = pair_data.start, pair_data.end
    interval = pair_data.interval
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df["z_score"], color="grey")

    plt.plot(df.index, df["entry_thr"].astype(float), color="red", label="entry_thr")
    plt.plot(df.index, -df["entry_thr"].astype(float), color="red")
    plt.plot(df.index, df["exit_thr"].astype(float), color="green", label="exit_thr")
    plt.plot(df.index, -df["exit_thr"].astype(float), color="green")

    plt.title(f"Z-Score: {x}/{y}")
    plt.ylabel("Z-Score")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.xlim(df.index.min(), df.index.max())
    plt.legend(loc="lower right", fontsize="small")

    if save:
        filename = f"{x}_{y}_zscore_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_positions(pair_data: Pair, directory: str | None = None, save: bool = True, show: bool = False) -> None:
    x, y, start, end, interval = pair_data.x, pair_data.y, pair_data.start, pair_data.end, pair_data.interval
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df['position'], color='black', linewidth=1.6)
    ax.set_ylabel('Position', color='black')
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Date')
    ax.set_title(f"Position Over Time: {x}/{y}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df.index.min(), df.index.max())
    plt.xticks(rotation=45, ha='right')

    if save:
        filename = f"{x}_{y}_positions_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_pnl(pair_data: Pair, btc_data: pd.DataFrame, directory: str | None = None, save: bool = True,
             show: bool = False) -> None:
    x, y, start, end, interval = pair_data.x, pair_data.y, pair_data.start, pair_data.end, pair_data.interval
    fee_rate = pair_data.fee_rate
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['total_return_pct'], label='Total Return [%] (Gross)', color='red', linewidth=1.6, zorder=3)
    ax1.plot(df.index, df['net_return_pct'], label=f'Total Return [%] (Net, fee: {fee_rate * 100}%)',
             linewidth=1.2, linestyle='--', color='red', zorder=3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Return [%]', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    ax1.plot(btc_data.index, btc_data['BTC_cum_return'], label=f'BTCUSDT total return',
             linewidth=1, linestyle='--', color='grey', zorder=1)

    plt.xlim(df.index.min(), df.index.max())
    ax1.legend(loc='lower right', fontsize="small")
    ax1.set_title(f'Total Return [%]: {x}/{y}')

    if save:
        filename = f"{x}_{y}_return_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
