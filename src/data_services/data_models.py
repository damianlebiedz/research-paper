from dataclasses import dataclass

import pandas as pd


@dataclass
class PairData:
    """Holds data for one pair."""
    x: str
    y: str
    data: pd.DataFrame

