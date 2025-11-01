from src.models.pair_data_model import PairData
from src.pair_loaders.pair_loader import PairLoader


class MultiPairLoader:
    """Utility class for loading and preparing market data for multiple pairs from CSV files."""

    def __init__(self, pairs: list[tuple[str, str]], data_dir: str = "data"):
        self.pairs = pairs
        self.data_dir = data_dir

    def load(self, start: str, end: str, interval: str) -> list[PairData]:
        """Load data for given pairs from CSV files."""
        results = []
        for x, y in self.pairs:
            loader = PairLoader(x, y, data_dir=self.data_dir)
            pair_data = loader.load(start=start, end=end, interval=interval)
            results.append(pair_data)
        return results

    def prepare(self, loaded_pairs: list[PairData]) -> list[PairData]:
        """Prepare data for loaded pairs."""
        results = []
        for pair_data in loaded_pairs:
            loader = PairLoader(pair_data.x, pair_data.y, data_dir=self.data_dir)
            prepared_data = loader.prepare(pair_data)
            results.append(prepared_data)
        return results

    def load_and_prepare(self, start: str, end: str, interval: str) -> list[PairData]:
        """Load and prepare data for given pairs from CSV files."""
        results = []
        for x, y in self.pairs:
            loader = PairLoader(x, y, data_dir=self.data_dir)
            prepared = loader.load_and_prepare(start=start, end=end, interval=interval)
            results.append(prepared)
        return results
