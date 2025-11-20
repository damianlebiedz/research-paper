from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class PairData:
    """Holds data for one pair."""
    x: str
    y: str
    start: str
    end: str
    interval: str
    data: pd.DataFrame
    stats: Optional[pd.DataFrame] = None
    fee_rate: float = 0


@dataclass
class PortfolioData:
    """Holds data for one portfolio."""
    start: str
    end: str
    interval: str
    data: Optional[pd.DataFrame] = None
    stats: Optional[pd.DataFrame] = None
    summary: Optional[pd.DataFrame] = None
    pairs_data: list[PairData] = field(default_factory=list)
    fee_rate: float = 0
