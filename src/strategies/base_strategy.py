import numpy as np
import pandas as pd


class BaseStrategy:
    """
    Base class for trading strategies.
    All strategies should inherit from this class and implement the generate_signals and calculate_returns methods.
    """

    def __init__(self, tickers, size) -> (list, float):
        """
        Initialize the strategy.
        """
        self.tickers = tickers
        self.size = size
        self.position = None
        self.trades = None
        self.pnl = None
        self.equity = None

    def generate_signals(self, data: pd.DataFrame) -> (np.ndarray, list):
        """
        Calculate entry/exit signals and positions.
        """
        raise NotImplementedError("Each strategy must implement generate_signals()")

    def calculate_returns(self, data: pd.DataFrame) -> (pd.Series, pd.Series):
        """
        Calculate PnL and equity curve based on the generated positions.
        """
        raise NotImplementedError("Each strategy must implement calculate_returns()")
