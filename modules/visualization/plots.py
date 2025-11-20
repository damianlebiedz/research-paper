import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from modules.data_services.data_models import PairData, PortfolioData
from modules.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_results_dir(directory: str | None) -> Path:
    base = Path().resolve().parent / "results"
    if directory:
        base = base / directory
    base.mkdir(parents=True, exist_ok=True)
    return base


def plot_prices(pair_data: PairData, directory: str | None = None,
                label: str = "prices", save: bool = True, show: bool = False) -> None:
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

    if save:
        filename = f"{x}_{y}_{label}_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
        logger.debug(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_zscore(pair_data: PairData, directory: str | None = None,
                thresholds: bool = False, save: bool = True, show: bool = False) -> None:
    x, y = pair_data.x, pair_data.y
    start, end = pair_data.start, pair_data.end
    interval = pair_data.interval
    df = pair_data.data.dropna()
    results_dir = _resolve_results_dir(directory)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y=df["z_score"], color="grey")
    label = ""
    if thresholds:
        label = "with_thr_"
        plt.plot(df.index, df['entry_threshold'], color="red", linestyle="--", label="short leg")
        plt.plot(df.index, -df['entry_threshold'], color="green", linestyle="--", label="long leg")
        plt.plot(df.index, df['exit_threshold'], color="black", linestyle="--", label="take profit")
        plt.plot(df.index, -df['exit_threshold'], color="black", linestyle="--")

    plt.title(f"Z-Score: {x}/{y}")
    plt.ylabel("Z-Score")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.xlim(df.index.min(), df.index.max())
    if thresholds:
        plt.legend(loc="lower right", fontsize="small")

    if save:
        filename = f"{x}_{y}_zscore_{label}{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
        logger.debug(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_positions(pair_data: PairData, directory: str | None = None,
                   save: bool = True, show: bool = False) -> None:
    x, y, start, end, interval = pair_data.x, pair_data.y, pair_data.start, pair_data.end, pair_data.interval
    df = pair_data.data.dropna()
    results_dir = _resolve_results_dir(directory)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df['position'], color='yellow', linewidth=1.6)
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
        logger.debug(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_pnl(pair_data: PairData, directory: str | None = None,
             save: bool = True, show: bool = False) -> None:
    x, y, start, end, interval = pair_data.x, pair_data.y, pair_data.start, pair_data.end, pair_data.interval
    fee_rate = pair_data.fee_rate
    df = pair_data.data.dropna()
    results_dir = _resolve_results_dir(directory)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['pnl_pct'], label='Total Return [%] (Gross)', linewidth=1.6)
    ax1.plot(df.index, df['net_pnl_pct'], label=f'Total Return [%] (Net, fee: {fee_rate * 100}%)',
             linewidth=1.6, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PnL%', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.xlim(df.index.min(), df.index.max())
    ax1.legend(loc='lower right', fontsize="small")
    ax1.set_title(f'Total Return [%]: {x}/{y}')

    if save:
        filename = f"{x}_{y}_pnl_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
        logger.debug(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_zscore_with_positions(pair_data: PairData, directory: str | None = None,
                               thresholds: bool = False, save: bool = True, show: bool = False) -> None:
    x, y = pair_data.x, pair_data.y
    start, end = pair_data.start, pair_data.end
    interval = pair_data.interval
    df = pair_data.data.dropna()
    results_dir = _resolve_results_dir(directory)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05}
    )
    sns.lineplot(x=df.index, y=df["z_score"], color="grey", ax=ax1)
    label = ""
    if thresholds:
        label = "with_thr_"
        ax1.plot(df.index, df['entry_threshold'], color="red", linestyle="--", label="short leg")
        ax1.plot(df.index, -df['entry_threshold'], color="green", linestyle="--", label="long leg")
        ax1.plot(df.index, df['exit_threshold'], color="black", linestyle="--", label="take profit")
        ax1.plot(df.index, -df['exit_threshold'], color="black", linestyle="--")

    ax1.set_title(f"Z-Score with positions: {x}/{y}")
    ax1.set_ylabel("Z-Score")
    ax1.grid(True, alpha=0.3)
    if thresholds:
        ax1.legend(loc='lower right', fontsize="small")
    ax1.set_xlabel("")

    ax2.plot(df.index, df['position'], color='yellow', linewidth=1.6)
    ax2.set_ylabel('Position')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df.index.min(), df.index.max())
    plt.xticks(rotation=45, ha='right')

    if save:
        filename = f"{x}_{y}_zscore_with_positions_{label}{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
        logger.debug(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_summary_pnl(portfolio_data: PortfolioData, directory: str | None = None,
                     save: bool = True, show: bool = False) -> None:
    start, end, interval = portfolio_data.start, portfolio_data.end, portfolio_data.interval
    fee_rate = portfolio_data.fee_rate
    df = portfolio_data.data.dropna()
    results_dir = _resolve_results_dir(directory)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['pnl_pct'], label='Total Return [%] (Gross)', linewidth=1.6)
    ax1.plot(df.index, df['net_pnl_pct'], label=f'Total Return [%] (Net, fee: {fee_rate * 100}%)',
             linewidth=1.6, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PnL%', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.xlim(df.index.min(), df.index.max())
    ax1.legend(loc='lower right', fontsize="small")
    ax1.set_title('Total Return [%] of portfolio')

    if save:
        filename = f"portfolio_pnl_{start}_{end}_{interval}.png".replace(":", "-")
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=150)
        logger.debug(f"Saved plot: {save_path}")
    if show:
        plt.show()
    plt.close()
