import logging
import hydra
from omegaconf import DictConfig
from skopt.space import Integer, Real

from modules.data_services.data_loaders import load_data, load_pair
from modules.core.models import Pair
from modules.data_services.data_utils import add_returns
from modules.data_services.statistical_tests import engle_granger_cointegration
from modules.performance.strategy import optimize_params, calculate_stats, single_pair_strategy
from modules.visualization.plots import plot_positions, plot_pnl, plot_zscore

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Load configuration
    tickers = cfg.market.tickers
    interval = cfg.market.interval

    fee_rate = cfg.market.fee_rate
    initial_cash = cfg.market.initial_cash
    risk_free_rate = cfg.market.risk_free_rate_annual

    position_size = cfg.strategy.pos_size
    beta_hedge = cfg.strategy.beta_hedge
    is_spread = cfg.strategy.is_spread

    ### === 1. Training ===

    pair_selection_start = cfg.pair_selection.start
    pair_selection_end = cfg.pair_selection.end

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
    logger.info(eg_df.head(5))

    # Select a TOP5 pairs
    tickers = eg_df.iloc[0:5, 0].tolist()

    # === 1.2 Parameter optimization ===

    pre_training_start = cfg.training.pre_start
    training_start = cfg.training.start
    training_end = cfg.training.end

    pre_test_start = cfg.strategy.pre_start
    test_start = cfg.strategy.start
    test_end = cfg.strategy.end

    # Perform Parameter Optimization
    param_space = [
        Integer(10, 30, name="rolling_window"),
        Real(2.0, 2.5, name="entry_threshold"),
        Real(0.5, 1.0, name="exit_threshold"),
        Real(2.0, 3.0, name="stop_loss"),
    ]
    metric = ("sortino_ratio_annual", "0.05% fee") # Objective function

    pairs = []
    for ticker in tickers:
        ticker_x = ticker.split('-')[0]
        ticker_y = ticker.split('-')[1]
        logger.info(f"Pair: {ticker_x}/{ticker_y}")

        best_params, best_score = optimize_params(ticker_x, ticker_y, fee_rate, initial_cash, position_size,
                                                  pre_training_start, training_start, training_end, interval,
                                                  beta_hedge, is_spread, risk_free_rate, param_space, metric)
        logger.info(best_params)
        logger.info(best_score)

        # === 2. Test ===

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

        single_pair_strategy(pair, rolling_window, entry_threshold, exit_threshold, stop_loss, position_size,
                             beta_hedge, is_spread)
        logger.info(pair.data.drop(columns=['total_return', 'total_fees', 'net_return']))

        # Calculate statistics
        pair.stats = calculate_stats(pair=pair, risk_free_rate_annual=risk_free_rate)
        logger.info(pair.stats)

        pairs.append(pair)

        # Visualization
        plot_positions(pair, directory='multi_pair_strategy', show=True, save=True)

        btc_data = load_data(
            tickers=['BTCUSDT'],
            start=test_start,
            end=test_end,
            interval=interval,
        )
        btc_data['BTC_return'] = btc_data['BTCUSDT'].pct_change()
        btc_data.loc[btc_data.index[0], 'BTC_return'] = 0.0
        btc_data['BTC_cum_return'] = (1 + btc_data['BTC_return']).cumprod() - 1

        plot_pnl(pair, btc_data, directory='multi_pair_strategy', show=True, save=True)
        plot_zscore(pair, directory='multi_pair_strategy', show=True, save=True)

    cols_to_sum = ['position', 'total_return', 'total_fees', 'net_return']
    summary_data = pairs[0].data[cols_to_sum].copy()
    for p in pairs[1:]:
        summary_data += p.data[cols_to_sum]
    summary = Pair(data=summary_data, start=pre_test_start, test_start=test_start, end=test_end,
                   interval=interval, fee_rate=fee_rate, initial_cash=initial_cash * 5)
    summary.data['total_return_pct'] = summary.data['total_return'] / (initial_cash * 5)
    summary.data['net_return_pct'] = summary.data['net_return'] / (initial_cash * 5)
    summary.data['position'] = summary.data['position'] / 5
    summary.stats = calculate_stats(pair=summary, risk_free_rate_annual=cfg.risk_free_rate_annual)

    # Show statistics
    logger.info(summary.stats)

    # Visualization
    plot_positions(summary, directory='multi_pair_strategy', show=True, save=True)

    btc_data = load_data(
        tickers=['BTCUSDT'],
        start=test_start,
        end=test_end,
        interval=interval,
    )
    btc_data['BTC_return'] = btc_data['BTCUSDT'].pct_change()
    btc_data.loc[btc_data.index[0], 'BTC_return'] = 0.0
    btc_data['BTC_cum_return'] = (1 + btc_data['BTC_return']).cumprod() - 1

    plot_pnl(summary, btc_data, directory='multi_pair_strategy', show=True, save=True)


if __name__ == "__main__":
    main()
