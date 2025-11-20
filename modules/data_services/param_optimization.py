from skopt import gp_minimize


def bayesian_optimization(param_space, objective, n_calls=40, random_state=42, minimize=False):
    def wrapped_objective(x):
        score = objective(x)
        return score if minimize else -score

    res = gp_minimize(
        wrapped_objective,
        dimensions=param_space,
        n_calls=n_calls,
        random_state=random_state
    )
    best_params = {dim.name: val for dim, val in zip(param_space, res.x)}
    best_score = res.fun if minimize else -res.fun
    return best_params, best_score, res
