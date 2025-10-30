import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BacktestEngine:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
        self.results = None
        self.stats = None

    def run(self):
        positions, trades = self.strategy.generate_trades(self.data)
        pnl, equity = self.strategy.calculate_returns(self.data)
        self.results = pd.DataFrame({
            "Position": positions,
            "PnL": pnl,
            "Equity": equity
        }, index=self.data.index)
        self.stats = self.calculate_stats(pnl, trades)
        return self.results, self.stats

    def calculate_stats(self, pnl, trades):
        stats = {
            "Total Return %": (self.results["Equity"].iloc[-1] - 1) * 100,
            "Trades": len([t for t in trades if t[0] in ("open_long", "open_short")]),
            "Sharpe Approx": np.sqrt(252) * pnl.mean() / (pnl.std() + 1e-9)
        }

        report = (
            f"Backtest Statistics:\n"
            f"{'Total Return %':20}: {stats['Total Return %']:6.2f}%\n"
            f"{'Trades':20}: {int(stats['Trades']):6d}\n"
            f"{'Sharpe Approx':20}: {stats['Sharpe Approx']:6.2f}"
        )

        return report

    def plot_results(self, data):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        tickers = self.strategy.tickers
        trades = self.strategy.trades
        results = self.results

        ax1.plot(data.index, data[tickers[0]], label=tickers[0], color="blue")
        ax1.plot(data.index, data[tickers[1]], label=tickers[1], color="orange")

        for action, date in trades:
            if action == "open_long":
                ax1.axvline(date, color="green", linestyle="--", alpha=0.7)
            elif action == "open_short":
                ax1.axvline(date, color="red", linestyle="--", alpha=0.7)
            elif action == "close":
                ax1.axvline(date, color="black", linestyle=":", alpha=0.5)

        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(results.index, results["Equity"], label="Equity", color="black")
        ax2.set_ylabel("Equity")
        ax2.legend(loc="upper right")

        plt.show()
