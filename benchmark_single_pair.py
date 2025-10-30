import pandas as pd

from modules.backtest import BacktestEngine
from modules.data_loader import DataLoader
from strategies.pairs_trading import PairsTradingStrategy

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def benchmark_single_pair():
    tickers = ['DOTUSDT', 'LINKUSDT']

    loader = DataLoader(interval="1h")
    data = loader.load(
        tickers=tickers,
        start='2024-01-01',
        end='2024-03-01',
        data_dir='data'
    )

    def minmax_scale(series):
        """Scale a series to range [0, 1]."""
        return (series - series.min()) / (series.max() - series.min())

    # Scale data
    data_scaled = pd.DataFrame()
    data_scaled[tickers[0]] = minmax_scale(data[tickers[0]])
    data_scaled[tickers[1]] = minmax_scale(data[tickers[1]])

    # Initialize strategy
    strategy = PairsTradingStrategy(
        tickers=tickers,
        entry_threshold=1.0,
        exit_threshold=0.0,
        size=1.0
    )

    # Initialize and run backtest engine
    engine = BacktestEngine(data=data, strategy=strategy)
    results, stats = engine.run()
    print(f"Backtest statistics: {stats}")

    # Plot results
    engine.plot_results(data_scaled)


if __name__ == '__main__':
    benchmark_single_pair()
