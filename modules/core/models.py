from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Pair:
    data: pd.DataFrame
    x: str = None
    y: str = None
    interval: str = None
    start: str = None
    test_start: str = None
    end: str = None

    stats: Optional[pd.DataFrame] = None
    fee_rate: float = 0
    initial_cash: float = 100000

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class PositionState:
    position: float = 0
    prev_position: float = 0
    q_x: float = 0
    q_y: float = 0
    w_x: float = None
    w_y: float = None
    entry_val: float = 0
    stop_loss_threshold: float = None

    def update_position(self, position, prev_position, q_x, q_y, w_x, w_y, entry_val, stop_loss_threshold):
        self.position = position
        self.prev_position = prev_position
        self.q_x = q_x
        self.q_y = q_y
        self.w_x = w_x
        self.w_y = w_y
        self.entry_val = entry_val
        self.stop_loss_threshold = stop_loss_threshold

    def clear_position(self):
        self.position = 0
        self.q_x = 0
        self.q_y = 0
        self.w_x = None
        self.w_y = None
        self.entry_val = 0
        self.stop_loss_threshold = None


@dataclass
class StrategyParams:
    entry_threshold: float
    exit_threshold: float
    stop_loss: float
