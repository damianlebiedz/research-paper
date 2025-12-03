import numpy as np
import pandas as pd
from skopt import gp_minimize


def bayesian_optimization(
        strategy_func,
        param_space,
        static_params: dict,
        metric: tuple,
        n_calls: int = 100,
        # n_initial_points: int = 30,
        random_state: int = 42,
        minimize: bool = False,
):
    def objective(**params):
        if (
                params["entry_threshold"] <= params["exit_threshold"]
                or params["stop_loss"] <= params["entry_threshold"]
                or params["rolling_window"] <= 1
        ):
            return -1e9

        full_params = {**static_params, **params}
        score = strategy_func(**full_params, metric=metric)

        if pd.isna(score) or np.isinf(score):
            return -1e9

        return score

    def wrapped(x):
        pdict = {dim.name: v for dim, v in zip(param_space, x)}
        score = objective(**pdict)

        return score if minimize else -score

    result = gp_minimize(
        wrapped,
        param_space,
        n_calls=n_calls,
        # n_initial_points=n_initial_points,
        # acq_func="EI",
        random_state=random_state
    )

    best_params = {dim.name: v for dim, v in zip(param_space, result.x)}
    best_score = result.fun if minimize else -result.fun

    return best_params, best_score, result
