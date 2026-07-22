import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    df['year_month'] = pd.to_datetime(df['year_month'])
    df = df.sort_values('year_month')


    # =====================================================
    # 1. BDI VOLATILITY
    # =====================================================

    df['BDI_change'] = df['BDI_mean'].diff()

    for window in [3, 6, 12]:

        df[f'BDI_volatility_{window}m'] = (
            df['BDI_mean']
            .rolling(window)
            .std()
        )


    # =====================================================
    # 2. BDI MONTHLY RETURNS
    # =====================================================

    df['BDI_monthly_return'] = (
        df['BDI_mean']
        .pct_change()
    )

    df['BDI_log_return'] = np.log(
        df['BDI_mean'] /
        df['BDI_mean'].shift(1)
    )


    # =====================================================
    # 3. BDI MOMENTUM
    # =====================================================

    for period in [1,3,6,12]:

        df[f'BDI_momentum_{period}m'] = (
            df['BDI_mean'] -
            df['BDI_mean'].shift(period)
        )


    # =====================================================
    # 4. ROLLING AVERAGES
    # =====================================================

    for window in [3,6,12]:

        df[f'BDI_MA_{window}m'] = (
            df['BDI_mean']
            .rolling(window)
            .mean()
        )


    # =====================================================
    # 5. BDI LAGS
    # =====================================================

    for lag in [1,3,6,12]:

        df[f'BDI_lag_{lag}m'] = (
            df['BDI_mean']
            .shift(lag)
        )


    # =====================================================
    # 6. BOTSWANA MACRO FEATURES
    # =====================================================

    botswana_features = [

        'GDP_growth',
        'inflation',
        'USD_BWP',
        'trade_balance',
        'exports',
        'imports',
        'diamond_production',
        'foreign_reserves',
        'unemployment'

    ]


    for feature in botswana_features:

        if feature in df.columns:

            # Monthly change
            df[f'{feature}_change'] = (
                df[feature]
                .pct_change()
            )

            # 3 and 12 month lag

            df[f'{feature}_lag_3m'] = (
                df[feature]
                .shift(3)
            )

            df[f'{feature}_lag_12m'] = (
                df[feature]
                .shift(12)
            )


    # =====================================================
    # 7. BANK OF BOTSWANA POLICY RATE
    # =====================================================

    if 'policy_rate' in df.columns:

        df['policy_rate_change'] = (
            df['policy_rate']
            .diff()
        )


        for lag in [1,3,6,12]:

            df[f'policy_rate_lag_{lag}m'] = (
                df['policy_rate']
                .shift(lag)
            )


    # =====================================================
    # 8. BRENT OIL FEATURES
    # =====================================================

    if 'Brent' in df.columns:

        df['Brent_return'] = (
            df['Brent']
            .pct_change()
        )


        for lag in [1,3,6,12]:

            df[f'Brent_lag_{lag}m'] = (
                df['Brent']
                .shift(lag)
            )


        df['Brent_momentum_3m'] = (
            df['Brent'] -
            df['Brent'].shift(3)
        )


        df['Brent_momentum_12m'] = (
            df['Brent'] -
            df['Brent'].shift(12)
        )


    # =====================================================
    # 9. SOUTH AFRICA LINK FEATURES
    # =====================================================

    sa_features = [

        'ZAR_BWP',
        'SA_inflation',
        'SA_GDP',
        'SA_policy_rate'

    ]


    for feature in sa_features:

        if feature in df.columns:

            df[f'{feature}_lag_3m'] = (
                df[feature]
                .shift(3)
            )

            df[f'{feature}_lag_12m'] = (
                df[feature]
                .shift(12)
            )


    # Remove missing rows created by lags

    df = df.dropna().reset_index(drop=True)

    return df
