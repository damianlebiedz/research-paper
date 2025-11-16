from dataclasses import dataclass, field
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
    stats: pd.DataFrame
    fee_rate: float = 0


@dataclass
class PortfolioData:
    """Holds data for one portfolio."""
    pairs_data: list[PairData] = field(default_factory=list)
