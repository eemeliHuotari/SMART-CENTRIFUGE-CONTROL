import argparse
from pathlib import Path
from src.modeling.train_models import train_models

def main():
    parser = argparse.ArgumentParser(description="Train and compare multi-target regression models.")
    parser.add_argument("--proc_dir", default="data/processed_examples", help="Directory with X/y CSVs")
    parser.add_argument("--out_models", default="results/models", help="Where to save model + scaler")
    parser.add_argument("--out_metrics", default="results/metrics/model_comparison_results.csv", help="CSV for metrics")
    args = parser.parse_args()

    X_csv = Path(args.proc_dir) / "X_features_for_ml.csv"
    y_csv = Path(args.proc_dir) / "y_targets_for_ml.csv"

    results_df, best = train_models(str(X_csv), str(y_csv), args.out_models, args.out_metrics)
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best} → saved to {args.out_models}")

if __name__ == "__main__":
    main()
