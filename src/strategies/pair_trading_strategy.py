import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy


class PairsTradingStrategy(BaseStrategy):
    def __init__(self, tickers, entry_threshold, exit_threshold, size=1):
        super().__init__(tickers, size)
        self.tickers = tickers
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.size = size
        self.position = None
        self.trades = None
        self.pnl = None
        self.equity = None

    def generate_trades(self, data):
        zscore = data['Z-Score']

        long_sig = zscore < -self.entry_threshold
        short_sig = zscore > self.entry_threshold

        exit_sig = (zscore.shift(1) * zscore < self.exit_threshold)
        exit_sig.iloc[0] = False

        position = np.zeros(len(data))
        trades = []

        for t in range(1, len(data)):
            if long_sig.iloc[t]:
                if position[t - 1] != 1:
                    trades.append(("open_long", data.index[t]))
                position[t] = 1
            elif short_sig.iloc[t]:
                if position[t - 1] != -1:
                    trades.append(("open_short", data.index[t]))
                position[t] = -1
            elif exit_sig.iloc[t]:
                if position[t - 1] != 0:
                    trades.append(("close", data.index[t]))
                position[t] = 0
            else:
                position[t] = position[t - 1]

        self.position, self.trades = position, trades
        return position, trades

    def calculate_returns(self, data):
        base_ret = data[self.tickers[0]].pct_change().fillna(0)
        hedge_ret = data[self.tickers[1]].pct_change().fillna(0)

        pnl = np.zeros(len(data))
        for t in range(1, len(data)):
            if self.position[t - 1] == 1:
                pnl[t] = self.size * (base_ret.iloc[t] - hedge_ret.iloc[t])
            elif self.position[t - 1] == -1:
                pnl[t] = self.size * (-base_ret.iloc[t] + hedge_ret.iloc[t])

        pnl = pd.Series(pnl, index=data.index)
        equity = (1 + pnl).cumprod()

        self.pnl, self.equity = pnl, equity
        return pnl, equity
