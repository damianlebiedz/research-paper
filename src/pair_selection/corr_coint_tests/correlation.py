def pearson_matrix(df, rolling_window=None):
    if rolling_window:
        return df.rolling(rolling_window).corr()
    return df.corr()
