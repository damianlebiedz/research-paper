from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Pair:
    x: str
    y: str
    start: str
    end: str
    interval: str
    data: pd.DataFrame
    stats: Optional[pd.DataFrame] = None
    fee_rate: float = 0
    initial_cash: float = 100000

    def __getitem__(self, key):
        return getattr(self, key)
