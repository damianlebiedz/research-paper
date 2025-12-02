from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from modules.data_services.normalization import cumulative_returns_index


def sum_of_standard_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df_cum_returns = cumulative_returns_index(df)
    results = []

    for x, y in combinations(df_cum_returns.columns, 2):
        ssd = np.sum((df_cum_returns[x] - df_cum_returns[y]) ** 2)
        results.append({'pair': f'{x}-{y}', 'ssd': ssd})

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by='ssd', ascending=True).reset_index(drop=True)


def pearson_correlation(df: pd.DataFrame, source: str = "prices") -> pd.DataFrame:
    df_clean = df[df.columns].dropna()

    if source == "returns":
        df_clean = df_clean.pct_change().dropna()
    elif source == "log_returns":
        df_clean = np.log(df_clean / df_clean.shift(1)).dropna()
    elif source != "prices":
        raise ValueError("source must be one of ['price', 'return', 'log_return']")

    corr_matrix = df_clean.corr(method='pearson')
    corr_df = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_df.columns = ['x', 'y', f'corr_{source}']
    corr_df['pair'] = corr_df['x'] + '-' + corr_df['y']
    corr_df = corr_df[['pair', f'corr_{source}']]
    return corr_df.sort_values(by=f'corr_{source}', ascending=False).reset_index(drop=True)


def engle_granger_cointegration(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df[df.columns].dropna()
    results = []

    for x, y in list(combinations(df.columns, 2)):
        X = df_clean[x].values
        Y = df_clean[y].values
        score, p_value, _ = coint(Y, X)
        results.append({
            'pair': f'{x}-{y}',
            'eg_p_value': p_value,
            'adf_stat': score,
        })
    return pd.DataFrame(results).sort_values(by='eg_p_value', ascending=True).reset_index(drop=True)


def johansen_cointegration(df: pd.DataFrame, det_order: int = 0,
                           k_ar_diff: int = 1) -> pd.DataFrame:
    df_clean = df[df.columns].dropna()
    results = []

    for x, y in combinations(df.columns, 2):
        data = df_clean[[x, y]]
        result = coint_johansen(data, det_order, k_ar_diff)

        trace_stat = result.lr1[0]
        crit_95 = result.cvt[0, 1]
        crit_99 = result.cvt[0, 2]

        results.append({
            'pair': f'{x}-{y}',
            'trace_stat': trace_stat,
            'crit_95': crit_95,
            'crit_99': crit_99,
            'trace_stat - crit_95': trace_stat - crit_95,
            'trace_stat - crit_99': trace_stat - crit_99,
        })
    return pd.DataFrame(results).sort_values(by='trace_stat - crit_95', ascending=False).reset_index(drop=True)


def perform_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    corr_prices_df = pearson_correlation(df, source="prices")
    corr_returns_df = pearson_correlation(df, source="returns")
    corr_log_returns_df = pearson_correlation(df, source="log_returns")
    eg_df = engle_granger_cointegration(df)
    johansen_df = johansen_cointegration(df)

    from modules.data_services.data_pipeline import merge_by_pair
    merged_df = merge_by_pair(
        dfs=[corr_prices_df, corr_returns_df, corr_log_returns_df, eg_df, johansen_df],
        keep_cols=[
            ['corr_prices'],
            ['corr_returns'],
            ['corr_log_returns'],
            ['eg_p_value'],
            ['trace_stat - crit_95']
        ]
    )
    return merged_df
