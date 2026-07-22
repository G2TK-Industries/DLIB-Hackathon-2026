"""Load, clean, and merge the 5 raw datasets into a monthly panel."""
import pandas as pd

DATA_DIR = "data/raw"
OUT_PATH = "data/processed/merged_monthly.csv"


def load_all():
    bdi = pd.read_csv(f"{DATA_DIR}/01_baltic_dry_index_daily.csv", parse_dates=["Date"])
    brent = pd.read_csv(f"{DATA_DIR}/02_brent_crude_monthly.csv", parse_dates=["Date"])
    pr = pd.read_csv(f"{DATA_DIR}/03_botswana_policy_rate.csv", parse_dates=["Date"])
    fao = pd.read_csv(f"{DATA_DIR}/04_fao_botswana_prices.csv", parse_dates=["Date"])
    hcp = pd.read_csv(f"{DATA_DIR}/05_human_capital_project.csv", parse_dates=["Date"])
    return bdi, brent, pr, fao, hcp


def merge_all():
    # TODO: call features.py for BDI aggregation, then merge all 5 on year_month
    raise NotImplementedError


if __name__ == "__main__":
    merge_all()
