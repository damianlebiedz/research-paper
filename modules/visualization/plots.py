import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from modules.core.models import StrategyResult


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def _resolve_results_dir(directory: str | None) -> Path:
    base = get_project_root() / "results"
    if directory:
        base = base / directory
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_zscore(
    result: StrategyResult,
    directory: str | None = None,
    save: bool = False,
    show: bool = True,
) -> None:
    x, y = result.ticker_x, result.ticker_y
    start, end = result.start, result.end
    interval = result.interval
    df = result.data
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
    plt.xticks(rotation=45, ha="right")
    plt.xlim(df.index.min(), df.index.max())
    plt.legend(loc="lower right", fontsize="small")

    if save:
        filename = f"{x}_{y}_zscore_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_positions(
    result: StrategyResult,
    directory: str | None = None,
    save: bool = False,
    show: bool = True,
) -> None:
    x, y, start, end, interval = (
        result.ticker_x,
        result.ticker_y,
        result.start,
        result.end,
        result.interval,
    )
    df = result.data
    results_dir = _resolve_results_dir(directory)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["position"], color="white", linewidth=1.6)
    ax.set_ylabel("Position", color="white")
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="y", labelcolor="white")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Date")
    ax.set_title(f"Position Over Time: {x}/{y}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df.index.min(), df.index.max())
    plt.xticks(rotation=45, ha="right")

    if save:
        filename = f"{x}_{y}_positions_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_pnl(
    result: StrategyResult,
    btc_data: pd.DataFrame | None = None,
    directory: str | None = None,
    save: bool = False,
    show: bool = True,
) -> None:
    x, y, start, end, interval = (
        result.ticker_x,
        result.ticker_y,
        result.start,
        result.end,
        result.interval,
    )
    fee_rate = result.fee_rate
    df = result.data
    results_dir = _resolve_results_dir(directory)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(
        df.index,
        df["total_return_pct"],
        label="Total Return (Gross)",
        color="red",
        linewidth=1.6,
        zorder=3,
    )
    ax1.plot(
        df.index,
        df["net_return_pct"],
        label=f"Total Return (Net, fee: {fee_rate * 100}%)",
        linewidth=1.2,
        linestyle="--",
        color="red",
        zorder=3,
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Return", color="white")
    ax1.tick_params(axis="y", labelcolor="white")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    if btc_data is not None:
        ax1.plot(
            btc_data.index,
            btc_data["BTC_cum_return"],
            label="BTCUSDT total return",
            linewidth=1,
            linestyle="--",
            color="grey",
            zorder=1,
        )

    plt.xlim(df.index.min(), df.index.max())
    ax1.legend(loc="lower right", fontsize="small")
    ax1.set_title(f"Total Return: {x}/{y}")

    if save:
        filename = f"{x}_{y}_return_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
