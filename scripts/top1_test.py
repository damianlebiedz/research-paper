from skopt.space import Integer, Real

from modules.data_services.data_loaders import load_data, load_pair
from modules.data_services.data_utils import add_returns
from modules.pair_selection.statistical_tests import engle_granger_cointegration
from modules.performance.strategy import single_pair_strategy, calculate_stats, calc_bayesian_params
from modules.visualization.plots import plot_positions, plot_zscore, plot_pnl


if __name__ == "__main__":
    tickers = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "TRXUSDT",
               "DOTUSDT", "LINKUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT"]
    interval = "1h"

    fee_rate = 0.0005  # 0.05%
    initial_cash = 100000
    position_size = 1  # always 100% of portfolio

    beta_hedge = True
    is_spread = False

    ### === 1. Training ===

    pair_selection_start = "2024-01-01"
    pair_selection_end = "2024-03-01"

    ### === 1.1 Pair Selection ===

    # Load data
    df = load_data(
        tickers=tickers,
        start=pair_selection_start,
        end=pair_selection_end,
        interval=interval
    )

    # Pair Selection
    eg_df = engle_granger_cointegration(df)
    print(eg_df.head(5))

    # Select a TOP1 pair
    tickers = eg_df.iloc[0:1, 0].tolist()
    ticker_x = tickers[0].split('-')[0]
    ticker_y = tickers[0].split('-')[1]

    # === 1.2 Parameter optimization ===

    pre_training_start = "2024-01-01"
    training_start = "2024-02-01"
    training_end = "2024-03-01"

    # Perform Bayesian Optimization
    param_space = [
        Integer(2, 600, name="rolling_window"),
        Real(1.0, 5.0, name="entry_threshold"),
        Real(0.0, 3.0, name="exit_threshold"),
        Real(1.0, 3.0, name="stop_loss"),
    ]
    metric = ("sortino_ratio_annual", "0.05% fee")
    minimize = False  # Maximize metric

    best_params, best_score = calc_bayesian_params(ticker_x, ticker_y, fee_rate, initial_cash, position_size,
                                                   pre_training_start, training_start, training_end, interval,
                                                   beta_hedge, is_spread, param_space, metric, minimize)

    print(best_params)
    print(best_score)

    # === 2. Test ===
    pre_test_start = "2024-02-01"
    test_start = "2024-03-01"
    test_end = "2024-04-01"

    # Load pair and calculate returns
    pair = load_pair(x=ticker_x, y=ticker_y, start=pre_test_start, end=test_end, interval=interval)
    add_returns(pair)

    entry_threshold = best_params["entry_threshold"]
    exit_threshold = best_params["exit_threshold"]
    stop_loss = best_params["stop_loss"]
    rolling_window = best_params["rolling_window"]

    # Run strategy
    pair.test_start = test_start
    pair.fee_rate = fee_rate
    pair.initial_cash = initial_cash

    single_pair_strategy(pair, rolling_window, entry_threshold, exit_threshold, stop_loss, position_size, beta_hedge, is_spread)
    pair.data.drop(columns=['total_return', 'total_fees', 'net_return'])

    # Calculate statistics
    pair.stats = calculate_stats(pair)
    print(pair.stats)

    # Visualization
    plot_positions(pair, 'strategy', True, True)

    btc_data = load_data(
        tickers=['BTCUSDT'],
        start=test_start,
        end=test_end,
        interval=interval,
    )
    btc_data['BTC_return'] = btc_data['BTCUSDT'].pct_change()
    btc_data.loc[btc_data.index[0], 'BTC_return'] = 0.0
    btc_data['BTC_cum_return'] = (1 + btc_data['BTC_return']).cumprod() - 1

    plot_pnl(pair, btc_data, 'strategy', True, True)
    plot_zscore(pair)
