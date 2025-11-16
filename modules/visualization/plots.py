import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from modules.data_services.data_models import PairData
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_results_dir(directory: str | None) -> Path:
    base = Path().resolve().parent / "results"
    if directory:
        base = base / directory
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_prices(pair_data: PairData, directory: str | None = None, label: str = "prices") -> None:
    x, y = pair_data.x, pair_data.y
    start = pair_data.start
    end = pair_data.end
    interval = pair_data.interval
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df[[f"{x}", f"{y}"]])
    plt.title(f"{label.capitalize()}: {x} vs {y}")
    plt.ylabel("Value")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{x}_{y}_{label}_{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved plot: {save_path}")
    plt.show()


def plot_zscore(pair_data: PairData, directory: str | None = None, thresholds: bool = False) -> None:
    x, y = pair_data.x, pair_data.y
    start, end = pair_data.start, pair_data.end
    interval = pair_data.interval
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df.index, y=df["z_score"], color="grey")
    label = ""
    if thresholds:
        label = "with_thr_"
        plt.plot(df.index, df['entry_threshold'], color="red", linestyle="--", label="short leg")
        plt.plot(df.index, -df['entry_threshold'], color="green", linestyle="--", label="long leg")
        plt.plot(df.index, df['exit_threshold'], color="black", linestyle="--", label="take profit")
        plt.plot(df.index, -df['exit_threshold'], color="black", linestyle="--")
    plt.title(f"Z-Score — {x}/{y}")
    plt.ylabel("Z-Score")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()

    filename = f"{x}_{y}_zscore_{label}{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved plot: {save_path}")
    plt.show()


def plot_pnl(pair_data: PairData, directory: str | None = None) -> None:
    x, y, start, end, interval = pair_data.x, pair_data.y, pair_data.start, pair_data.end, pair_data.interval
    fee_rate = pair_data.fee_rate
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df['pnl_pct'], label='Total Return [%]', linewidth=1.8)
    ax1.plot(df.index, df['net_pnl_pct'], label=f'Total Return [%] with {fee_rate*100}% fee', linewidth=1.6, linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PnL%', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()
    plt.title('Total Return [%]')

    filename = f"{x}_{y}_pnl_{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved plot: {save_path}")
    plt.show()


def plot_positions(pair_data: PairData, directory: str | None = None) -> None:
    x, y, start, end, interval = pair_data.x, pair_data.y, pair_data.start, pair_data.end, pair_data.interval
    df = pair_data.data
    results_dir = _resolve_results_dir(directory)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df.index, df['position'], color='tab:red', linewidth=1.6)
    ax.set_ylabel('Position', color='tab:red')
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Time')
    plt.title('Position Over Time')
    fig.tight_layout()

    filename = f"{x}_{y}_positions_{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved plot: {save_path}")
    plt.show()
