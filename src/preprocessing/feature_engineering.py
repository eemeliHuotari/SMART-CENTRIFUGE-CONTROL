import pandas as pd
import numpy as np

TANK_CAPACITY_LITERS = 30000

def compute_efficiency_and_features(
    drying_csv: str,
    runs_csv: str,
    syote_csv: str,
    cleaned_sensors_csv: str,
    out_efficiency_csv: str,
    out_X_csv: str,
    out_y_csv: str,
    out_full_csv: str
):
    drying_df = pd.read_csv(drying_csv, parse_dates=["wet_time", "dry_time", "mid_time"])
    runs_df = pd.read_csv(runs_csv, parse_dates=["start_time", "end_time"])

    # Simple row-wise concat assumes one drying event per run in order
    assert len(runs_df) == len(drying_df), "Mismatch in centrifuge runs vs drying events"
    df = pd.concat([drying_df.reset_index(drop=True), runs_df.reset_index(drop=True)], axis=1)

    # Load kiintoaine syöte time series
    syote = pd.read_csv(syote_csv, encoding="utf-8", sep=";", decimal=",")
    syote_time = pd.to_datetime(syote['Syote_Aika'], dayfirst=True, errors='coerce')
    syote_val = pd.to_numeric(syote['Kiintoaine_syote_g/l'].astype(str).str.replace(",", "."), errors="coerce")
    syote_series = pd.Series(syote_val.values, index=syote_time).dropna()
    syote_series = syote_series.resample("1min").mean().interpolate("time")

    # feed liters from % (recompute to be safe if CSV edited)
    df["feed_liters"] = (df["start_fill_pct"] - df["end_fill_pct"]) / 100 * TANK_CAPACITY_LITERS

    # nearest solids concentration at run start
    def nearest_solids(t):
        idx = syote_series.index.get_indexer([t], method="nearest")[0]
        return syote_series.iloc[idx]

    df["Kiintoaine_syote_g/l"] = df["start_time"].apply(nearest_solids)

    # mass balance
    residual_moisture_fraction = 0.08
    df["wet_weight_g"]  = df["wet_weight"]
    df["dry_weight_g"]  = df["dry_weight"]
    df["solids_in_g"]   = df["feed_liters"] * df["Kiintoaine_syote_g/l"]
    df["water_in_g"]    = df["feed_liters"] * 1000 - df["solids_in_g"]
    df["solids_out_g"]  = df["dry_weight_g"] * (1 - residual_moisture_fraction)
    df["water_out_g"]   = df["wet_weight_g"] - df["solids_out_g"]
    df["moisture_lost_g"] = df["wet_weight_g"] - df["dry_weight_g"]
    df["water_removed_g"] = df["water_in_g"] - df["water_out_g"]
    df["adjusted_water_removed"] = df["water_removed_g"] - df["moisture_lost_g"]

    # Aggregate sensor features per run window
    sensor_df = pd.read_csv(cleaned_sensors_csv, parse_dates=["Unnamed: 0"])
    sensor_df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
    agg_features = {
        "syote_Arvo_m3/h": ["mean", "std"],
        "rumpu_nopeus_Arvo_RPM": ["mean", "std"],
        "Ero_nopeus_Arvo_RPM": ["mean", "std"],
        "Momentti_Arvo_%": ["mean", "std"],
    }
    sensor_features = []
    for _, row in df.iterrows():
        mask = (sensor_df["timestamp"] >= row["start_time"]) & (sensor_df["timestamp"] <= row["end_time"])
        segment = sensor_df.loc[mask]
        if segment.empty:
            continue
        stats = segment.agg(agg_features)
        flat_stats = {f"{k}_{stat}": stats[k][stat] for k in agg_features for stat in agg_features[k]}
        flat_stats["start_time"] = row["start_time"]
        sensor_features.append(flat_stats)

    agg_df = pd.DataFrame(sensor_features)
    df = df.merge(agg_df, on="start_time", how="inner")

    # Targets
    df["water_efficiency"] = (df["water_in_g"] - df["moisture_lost_g"]) / df["feed_liters"]
    df["solids_loss_ratio"] = df["solids_out_g"] / df["solids_in_g"]
    df["torque_mean"] = df["Momentti_Arvo_%_mean"]

    features = [
        "syote_Arvo_m3/h_mean", "syote_Arvo_m3/h_std",
        "rumpu_nopeus_Arvo_RPM_mean", "rumpu_nopeus_Arvo_RPM_std",
        "Ero_nopeus_Arvo_RPM_mean", "Ero_nopeus_Arvo_RPM_std",
        "Kiintoaine_syote_g/l",
        "Momentti_Arvo_%_mean", "Momentti_Arvo_%_std",
    ]
    targets = ["water_efficiency", "solids_loss_ratio", "torque_mean"]

    X = df[features].copy()
    y = df[targets].copy()

    df.to_csv(out_efficiency_csv, index=False)
    X.to_csv(out_X_csv, index=False)
    y.to_csv(out_y_csv, index=False)
    df.to_csv(out_full_csv, index=False)  # full ML-ready table

    return X, y, df
