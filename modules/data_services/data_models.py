from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Position:
    start: str
    end: str
    beta: float
    alpha: float
    mean: float
    std: float
    w_x: float
    w_y: float


@dataclass
class Pair:
    x: str
    y: str
    start: str
    end: str
    interval: str
    data: pd.DataFrame
    stats: Optional[pd.DataFrame] = None
    fee_rate: float = 0 # TODO
    initial_cash: float = 100000 # TODO
    # positions: list[Position] = field(default_factory=list) # TODO

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class Portfolio:
    start: str
    end: str
    interval: str
    fee_rate: float
    initial_cash: float
    data: Optional[pd.DataFrame] = None
    stats: Optional[pd.DataFrame] = None
    summary: Optional[pd.DataFrame] = None
    pairs_data: list[Pair] = field(default_factory=list)
