"""Single entry point: loads data, runs chosen model, writes submissions/predictions.csv.

Usage:
    python src/generate_predictions.py --model classical
    python src/generate_predictions.py --model deep
"""
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classical", "deep"], required=True)
    args = parser.parse_args()
    # TODO: load merged data, fit/load chosen model, forecast 12 months,
    # write submissions/predictions.csv with columns [year_month, forecast]
