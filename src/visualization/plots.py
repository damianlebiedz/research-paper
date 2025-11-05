import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_prices(pair_data, start: str, end: str, interval: str):
    x, y = pair_data.x, pair_data.y
    df = pair_data.data.copy()

    results_dir = Path().resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df[[f"{x}", f"{y}"]])
    plt.title(f"Prices: {x} vs {y}")
    plt.ylabel("Value")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{x}_{y}_prices_{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved prices plot: {save_path}")
    plt.show()


def plot_scaled_prices(pair_data, start: str, end: str, interval: str):
    x, y = pair_data.x, pair_data.y
    df = pair_data.data.copy()

    results_dir = Path().resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df[[f"{x}_scaled", f"{y}_scaled"]])
    plt.title(f"Scaled Prices: {x} vs {y}")
    plt.ylabel("Scaled Value")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{x}_{y}_scaled_prices_{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved scaled prices plot: {save_path}")
    plt.show()


def plot_zscore_with_thresholds(pair_data, start: str, end: str, interval: str, open_threshold: float = 2.0,
                                close_threshold: float = 0):
    x, y = pair_data.x, pair_data.y
    df = pair_data.data.copy()

    results_dir = Path().resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df.index, y=df["Z-Score"], color="black")

    plt.axhline(0, color="gray", linestyle="--", lw=1)
    plt.axhline(open_threshold, color="red", linestyle="--", lw=1, label=f"Open +{open_threshold}")
    plt.axhline(-open_threshold, color="red", linestyle="--", lw=1, label=f"Open -{open_threshold}")
    plt.axhline(close_threshold, color="green", linestyle="--", lw=1, label=f"Close +{close_threshold}")
    plt.axhline(-close_threshold, color="green", linestyle="--", lw=1, label=f"Close -{close_threshold}")

    plt.title(f"Z-Score with Thresholds (±{open_threshold}, ±{close_threshold}) — {x}/{y}")
    plt.ylabel("Z-Score")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()

    filename = f"{x}_{y}_zscore_{start}_{end}_{interval}.png".replace(":", "-")
    save_path = results_dir / filename
    plt.savefig(save_path, dpi=150)
    logger.debug(f"Saved Z-Score plot: {save_path}")
    plt.show()
