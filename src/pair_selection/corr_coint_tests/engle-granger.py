import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from tqdm import tqdm


def engle_granger_matrix(df):
    n = df.shape[1]
    result = pd.DataFrame(np.nan, index=df.columns, columns=df.columns)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for i, j in tqdm(pairs, desc="Engle-Granger Pairs"):
        y = df.iloc[:, i]
        x = df.iloc[:, j]
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        resid = model.resid
        adf_pvalue = adfuller(resid)[1]
        result.iloc[i, j] = result.iloc[j, i] = 1 if adf_pvalue < 0.05 else 0

    np.fill_diagonal(result.values, 1)
    return result


def plot_matrix(matrix, title='Matrix'):
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.show()