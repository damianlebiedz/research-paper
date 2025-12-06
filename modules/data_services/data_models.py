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
