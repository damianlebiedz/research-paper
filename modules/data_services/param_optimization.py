from typing import Any, Callable

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Dimension


def bayesian_optimization(
        strategy_func: Callable,
        param_space: list[Dimension],
        static_params: dict[str, Any],
        n_calls: int,
        random_state: int,
        minimize: bool = False,
        metric_path: tuple = ('sharpe_ratio', '0.05% fee', 'Summary')
) -> tuple[dict[str, Any], float, Any]:
    """Performs Bayesian Optimization over the given parameter space."""
    def objective(**dynamic_params):
        if 'entry_threshold' in dynamic_params and 'exit_threshold' in dynamic_params:
            if dynamic_params['entry_threshold'] <= dynamic_params['exit_threshold']:
                return -1.0

        full_params = {**static_params, **dynamic_params}
        if 'window_in_steps' in full_params:
            full_params['window_in_steps'] = int(full_params['window_in_steps'])

        try:
            p = strategy_func(**full_params)
            score = p.summary
            for key in metric_path:
                score = score[key]

            if pd.isna(score) or np.isinf(score):
                return 0.0
            return score

        except Exception as e:
            print(f"Optimization Error for params {dynamic_params}: {e}")
            return -1.0

    def wrapped_objective(x):
        param_dict = {dim.name: val for dim, val in zip(param_space, x)}
        score = objective(**param_dict)
        return score if minimize else -score

    results = gp_minimize(
        wrapped_objective,
        dimensions=param_space,
        n_calls=n_calls,
        random_state=random_state
    )
    best_params = {dim.name: val for dim, val in zip(param_space, results.x)}
    best_score = results.fun if minimize else -results.fun

    return best_params, best_score, results
