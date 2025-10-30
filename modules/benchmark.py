import itertools
import pandas as pd
import matplotlib.pyplot as plt
from modules.backtest import BacktestEngine
from modules.data_loader import DataLoader


class BenchmarkEngine:
    """
    Class for running bulk backtests across multiple data configurations
    and strategy parameter sets. Responsible only for orchestration,
    results aggregation, and visualization.
    """

    def __init__(self, strategy_class):
        """
        :param strategy_class: strategy class (must implement generate_trades and calculate_returns)
        """
        self.strategy_class = strategy_class
        self.results_df = None
        self.engines = []

    def run(self, data_configs, strategy_params_list):
        """
        Run backtests for all combinations of data configs and strategy parameters.

        :param data_configs: dict or list of dicts with keys: tickers, start, end, interval
        :param strategy_params_list: list of dicts with keys: entry_threshold, exit_threshold, size
        :return: DataFrame with all aggregated results
        """
        results_list = []
        self.engines = []

        # Ensure data_configs is always a list
        if isinstance(data_configs, dict):
            data_configs = [data_configs]

        for data_cfg, strat_cfg in itertools.product(data_configs, strategy_params_list):
            loader = DataLoader(interval=data_cfg.get("interval", "1d"))
            data = loader.load(
                tickers=data_cfg["tickers"],
                start=data_cfg["start"],
                end=data_cfg["end"],
            )

            strategy = self.strategy_class(
                tickers=data_cfg["tickers"],
                entry_threshold=strat_cfg["entry_threshold"],
                exit_threshold=strat_cfg["exit_threshold"],
                size=strat_cfg.get("size")
            )
            engine = BacktestEngine(data=data, strategy=strategy)
            _, stats = engine.run()

            stats.update({
                "tickers": ",".join(data_cfg["tickers"]),
                "start": data_cfg["start"],
                "end": data_cfg["end"],
                "interval": data_cfg.get("interval"),
                "entry_threshold": strat_cfg["entry_threshold"],
                "exit_threshold": strat_cfg["exit_threshold"],
                "size": strat_cfg.get("size")
            })

            results_list.append(stats)
            self.engines.append(engine)

        self.results_df = pd.DataFrame(results_list)
        return self.results_df

    def plot_all_equities(self):
        """
        Plot equity curves for all tested combinations, sorted by final equity.
        """
        if not self.engines:
            raise ValueError("Run benchmark first before plotting.")

        sorted_engines = sorted(
            self.engines,
            key=lambda e: e.results["Equity"].iloc[-1],
            reverse=True
        )

        plt.figure(figsize=(12, 6))
        for engine in sorted_engines:
            label = (
                f"{engine.strategy.tickers}, "
                f"E={engine.strategy.entry_threshold}, "
                f"X={engine.strategy.exit_threshold}, "
                f"Sz={engine.strategy.size}"
            )
            plt.plot(engine.results["Equity"], label=label)

        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Equity Curves – All Configurations")
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
