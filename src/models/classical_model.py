"""Classical baseline: SARIMA / GBT on lagged features."""


def fit(train_df):
    raise NotImplementedError


def forecast(model, horizon=12):
    raise NotImplementedError
