import pandas as pd

from src.pair_loaders.multi_pair_loader import MultiPairLoader
# from src.modules.backtest_engine import BacktestEngine
from src.pair_loaders.pair_loader import PairLoader
from src.visualization.plot_pair_data import plot_scaled_prices, plot_zscore_with_thresholds

# from src.strategies.pair_trading_strategy import PairsTradingStrategy

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def run():
    start = "2024-01-01"
    end = "2024-03-01"
    interval = "1h"

    # 1. Load and prepare a single pair
    loader = PairLoader(x="DOTUSDT", y="LINKUSDT")
    single_result = loader.load_and_prepare(start=start, end=end, interval=interval)
    print(single_result.data)
    plot_scaled_prices(single_result, start, end, interval)
    plot_zscore_with_thresholds(single_result, start, end, interval)

    # 2. Load and prepare multiple pairs
    pairs = [
        ("AVAXUSDT", "DOGEUSDT"),
        ("ETHUSDT", "BTCUSDT"),
    ]
    multi_loader = MultiPairLoader(pairs)
    multiple_results = multi_loader.load_and_prepare(start=start, end=end, interval=interval)

    for pair in multiple_results:
        print(pair.data)
        plot_scaled_prices(pair, start, end, interval)
        plot_zscore_with_thresholds(pair, start, end, interval)

    # # Scale data
    # ...
    #
    # # Initialize strategy
    # strategy = PairsTradingStrategy(
    #     tickers=tickers,
    #     entry_threshold=1.0,
    #     exit_threshold=0.0,
    #     size=1.0
    # )
    #
    # # Initialize and run backtest engine
    # engine = BacktestEngine(data=data, strategy=strategy)
    # results, stats = engine.run()
    # print(f"Backtest statistics: {stats}")
    #
    # # Plot results
    # engine.plot_results(data_scaled)


if __name__ == '__main__':
    run()
