"""
data_pipeline.py

Loads, cleans and merges all five datasets into a single monthly dataset.

Output:
    data/processed/merged_monthly.csv
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


def add_year_month(df):
    """
    Converts a Date column into YYYY-MM format.
    """
    df = df.copy()
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    return df



def load_data():

    print("Loading datasets...")

    bdi = pd.read_csv(
        RAW_DATA / "01_baltic_dry_index_daily.csv",
        parse_dates=["Date"]
    )

    brent = pd.read_csv(
        RAW_DATA / "02_brent_crude_monthly.csv",
        parse_dates=["Date"]
    )

    policy = pd.read_csv(
        RAW_DATA / "03_botswana_policy_rate.csv",
        parse_dates=["Date"]
    )

    fao = pd.read_csv(
        RAW_DATA / "04_fao_botswana_prices.csv",
        parse_dates=["Date"]
    )

    hcp = pd.read_csv(
        RAW_DATA / "05_human_capital_project.csv",
        parse_dates=["Date"]
    )

    return bdi, brent, policy, fao, hcp



def clean_data(bdi, brent, policy, fao, hcp):

    print("Cleaning datasets...")

    datasets = [bdi, brent, policy, fao, hcp]

    for df in datasets:

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Sort chronologically
        df.sort_values("Date", inplace=True)

        # Reset index
        df.reset_index(drop=True, inplace=True)

    return bdi, brent, policy, fao, hcp



def prepare_bdi(bdi):
    """
    Naive monthly aggregation.

    NOTE:
    Advanced BDI feature engineering belongs in features.py.
    """

    bdi = add_year_month(bdi)

    monthly = (
        bdi.groupby("year_month")
        .agg(
            BDI_mean=("BDI_Close", "mean")
        )
        .reset_index()
    )

    return monthly


def prepare_brent(brent):

    brent = add_year_month(brent)

    return brent[["year_month", "Brent_USD_per_barrel"]]


def prepare_policy(policy):

    policy = add_year_month(policy)

    return policy[["year_month", "policy_rate"]]


def prepare_fao(fao):

    fao = add_year_month(fao)

    fao["feature"] = "FAO_" + fao["Item Code"].astype(str)

    fao = (
        fao.pivot_table(
            index="year_month",
            columns="feature",
            values="Value",
            aggfunc="first"
        )
        .reset_index()
    )

    fao.columns.name = None

    return fao


def prepare_hcp(hcp):

    hcp = add_year_month(hcp)

    hcp["feature"] = (
        hcp["REF_AREA"] + "_" + hcp["INDICATOR"]
    )

    hcp = (
        hcp.pivot_table(
            index="year_month",
            columns="feature",
            values="Value",
            aggfunc="first"
        )
        .reset_index()
    )

    hcp.columns.name = None

    return hcp



def merge_data(bdi, brent, policy, fao, hcp):

    print("Merging datasets...")

    merged = bdi.copy()

    for df in [brent, policy, fao, hcp]:

        merged = merged.merge(
            df,
            on="year_month",
            how="outer"
        )

    merged.sort_values("year_month", inplace=True)

    merged.reset_index(drop=True, inplace=True)

    return merged

def remove_missing_rows(df):
    """
    Removes rows containing missing values.
    """
    print("\nChecking for missing values...")

    before = len(df)

    missing_rows = df[df.isnull().any(axis=1)]

    print(f"Rows with missing values: {len(missing_rows)}")

    df = df.dropna()

    after = len(df)

    print(f"Rows removed: {before - after}")
    print(f"Rows remaining: {after}")

    return df

def verify_dataset(df):
    """
    Checks dataset structure, missing values and row information.
    """

    print("\n========== DATASET INFORMATION ==========")

    # Dataset shape
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n========== COLUMN INFORMATION ==========")

    # Column names and data types
    print(df.info())

    print("\n========== MISSING VALUES ==========")

    # Count missing values per column
    missing = df.isnull().sum()

    missing_summary = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing Percentage (%)": 
            (missing.values / len(df)) * 100
    })

    print(missing_summary)

    print("\n========== ROWS WITH MISSING VALUES ==========")

    # Count rows containing missing values
    missing_rows = df[df.isnull().any(axis=1)]

    print(f"Rows containing missing values: {len(missing_rows)}")

    if len(missing_rows) > 0:
        print("\nExample rows with missing values:")
        print(missing_rows.head())

    print("\n========== DUPLICATE ROWS ==========")

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    print("\n========== SAMPLE DATA ==========")

    print(df.head())

    print("\n========== LAST ROWS ==========")

    print(df.tail())

def save_data(df):

    output = PROCESSED_DATA / "merged_monthly.csv"

    df.to_csv(output, index=False)

    print(f"\nSaved merged dataset to:\n{output}")



def main():

    bdi, brent, policy, fao, hcp = load_data()

    bdi, brent, policy, fao, hcp = clean_data(
        bdi,
        brent,
        policy,
        fao,
        hcp
    )

    bdi = prepare_bdi(bdi)
    brent = prepare_brent(brent)
    policy = prepare_policy(policy)
    fao = prepare_fao(fao)
    hcp = prepare_hcp(hcp)

    merged = merge_data(
        bdi,
        brent,
        policy,
        fao,
        hcp
    )

    verify_dataset(merged)

    # Remove incomplete rows
    merged = remove_missing_rows(merged)

    save_data(merged)

    print("\nFinished successfully.")
    print(merged.head())
    print(f"\nDataset shape: {merged.shape}")


if __name__ == "__main__":
    main()