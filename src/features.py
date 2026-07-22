"""Daily BDI -> monthly feature engineering (volatility, trend, momentum, extremes)."""
import pandas as pd


def engineer_bdi_features(bdi: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns: Date, BDI_Close, BDI_High, BDI_Low
    Returns one row per year_month with engineered features, e.g.:
      - BDI_mean, BDI_std (volatility)
      - BDI_month_return (last - first) / first (trend)
      - BDI_extreme_days (count |daily % change| > 3%)
      - BDI_3m_momentum (direction of trailing 3-month mean)
    """
    raise NotImplementedError
