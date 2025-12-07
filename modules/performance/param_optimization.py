import numpy as np
from skopt import gp_minimize


def bayesian_optimization(
        strategy_func,
        param_space,
        static_params: dict,
        metric: tuple,
        n_calls: int = 100,
        n_initial_points: int = 10,
        random_state: int = 42,
        minimize: bool = False,
        replicates: int = 3,
        penalty_bad: float = -1e2,
):
    def evaluate_point(pdict):
        scores = []
        for r in range(replicates):
            try:
                val = strategy_func(**{**static_params, **pdict}, metric=metric)
                if val is None or np.isnan(val) or np.isinf(val):
                    scores.append(penalty_bad)
                else:
                    scores.append(float(val))
            except Exception as e:
                print(f"[eval error] {pdict} rep={r} -> {e}")
                scores.append(penalty_bad)
        return float(np.mean(scores))

    def wrapped(x):
        pdict = {dim.name: v for dim, v in zip(param_space, x)}
        score = evaluate_point(pdict)

        if score == 0 or np.isnan(score):
            return penalty_bad

        return score if minimize else -score

    result = gp_minimize(
        wrapped,
        dimensions=param_space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
        acq_func="EI",
        # verbose=True,
    )

    best_params = {dim.name: v for dim, v in zip(param_space, result.x)}
    best_score = result.fun if minimize else -result.fun

    return best_params, best_score, result
