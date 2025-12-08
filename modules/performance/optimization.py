from typing import Any

from skopt.space import Integer, Real
import numpy as np
from random import uniform, randint
from joblib import Parallel, delayed


def random_search(strategy_func, param_space, static_params, metric, n_iter=500, n_jobs=-1,
                  replicates=1, penalty_bad=-1e2) -> tuple[Any, float]:
    """Perform a random search over the parameter space."""

    def evaluate_point(pd, idx) -> float:
        """Evaluate a point in the parameter space."""
        scores = []
        for _ in range(replicates):
            try:
                val = strategy_func(**{**static_params, **pd}, metric=metric)
                if val is None or np.isnan(val) or val == 0 or np.isinf(val):
                    scores.append(penalty_bad)
                else:
                    scores.append(float(val))
            except Exception as e:
                print(e)
                scores.append(penalty_bad)

        avg_score = float(np.mean(scores))
        print(f"Iteration {idx + 1}/{n_iter} | Score: {avg_score:.4f}")

        return avg_score

    pdicts = []
    for _ in range(n_iter):
        pdict = {}
        for dim in param_space:
            if isinstance(dim, Integer):
                pdict[dim.name] = randint(dim.low, dim.high)
            elif isinstance(dim, Real):
                pdict[dim.name] = uniform(dim.low, dim.high)
        pdicts.append(pdict)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_point)(p, i) for i, p in enumerate(pdicts)
    )
    best_score, best_params = max(results, key=lambda x: x[0])

    return best_params, best_score
