from itertools import combinations

import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def pearson_correlation(df: pd.DataFrame) -> pd.DataFrame:
    pairs = list(combinations(df.columns, 2))
    results = []

    for x, y in pairs:
        corr = df[[x, y]].corr(method='pearson').iloc[0, 1]
        results.append({'pair': f'{x}-{y}', 'corr': corr})

    return pd.DataFrame(results)


def engle_granger_cointegration(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    pairs = list(combinations(tickers, 2))
    results = []
    for x, y in pairs:
        score, p_value, _ = coint(df[x], df[y])
        results.append({'pair': f'{x}-{y}', 'score': score, 'p_value': p_value})
    results_df = pd.DataFrame(results)
    return results_df


def johansen_cointegration(df: pd.DataFrame, tickers: list[str], det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    pairs = list(combinations(tickers, 2))
    results = []

    for x, y in pairs:
        data = df[[x, y]].dropna()
        result = coint_johansen(data, det_order, k_ar_diff)

        # bierzemy statystykę trace dla r=0 (czyli test na istnienie przynajmniej jednej relacji kointegrującej)
        trace_stat = result.lr1[0]
        crit_95 = result.cvt[0, 1]

        # heurystyczny "pseudo p-value"
        p_est = max(0.0, min(1.0, 1 - trace_stat / crit_95)) if trace_stat < crit_95 else 0.01

        results.append({
            'pair': f'{x}-{y}',
            'trace_stat': trace_stat,
            'crit_95': crit_95,
            'p_est': p_est
        })

    results_df = pd.DataFrame(results)
    return results_df
