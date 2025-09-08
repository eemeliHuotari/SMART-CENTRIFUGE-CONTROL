import argparse
from pathlib import Path
from src.preprocessing.clean_sensors import load_and_align_sensors, save_cleaned
from src.preprocessing.extract_runs import extract_runs_from_tank, save_runs
from src.preprocessing.match_drying import build_drying_pairs, save_drying
from src.preprocessing.feature_engineering import compute_efficiency_and_features

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline end-to-end.")
    parser.add_argument("--raw_dir", default="data/raw_examples", help="Directory with raw CSVs")
    parser.add_argument("--proc_dir", default="data/processed_examples", help="Output directory for processed artifacts")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    proc = Path(args.proc_dir)
    proc.mkdir(parents=True, exist_ok=True)

    linkous = raw / "Linkous_data_fixed.csv"
    kuivain = raw / "Kuivain.csv"
    pinta   = raw / "pesusailio_pinta.csv"
    syote   = raw / "Syöte ja tuote.csv"

    # 1) Clean sensors
    aligned = load_and_align_sensors(str(linkous))
    clean_csv = proc / "cleaned_centrifuge_data.csv"
    save_cleaned(aligned, str(clean_csv))
    print(f"Saved {clean_csv}")

    # 2) Drying events
    drying_df = build_drying_pairs(str(kuivain))
    drying_csv = proc / "drying_events.csv"
    save_drying(drying_df, str(drying_csv))
    print(f"Saved {drying_csv}")

    # 3) Runs from tank
    runs_df = extract_runs_from_tank(str(pinta))
    runs_csv = proc / "centrifuge_runs_from_pinta.csv"
    save_runs(runs_df, str(runs_csv))
    print(f"Saved {runs_csv}")

    # 4) Feature engineering & ML tables
    eff_csv  = proc / "centrifuge_efficiency_final.csv"
    X_csv    = proc / "X_features_for_ml.csv"
    y_csv    = proc / "y_targets_for_ml.csv"
    full_csv = proc / "ml_ready_full_dataset.csv"

    X, y, full = compute_efficiency_and_features(
        str(drying_csv), str(runs_csv), str(syote), str(clean_csv),
        str(eff_csv), str(X_csv), str(y_csv), str(full_csv)
    )
    print(f"Saved ML tables: {X_csv}, {y_csv}, {full_csv}")

if __name__ == "__main__":
    main()
