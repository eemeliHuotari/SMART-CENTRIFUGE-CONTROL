import pandas as pd
import numpy as np
from datetime import timedelta

def load_and_align_sensors(path_linkous: str) -> pd.DataFrame:
    """
    Reads Linkous_data_fixed.csv, parses timestamps & values, aligns all series
    to a common 1‑minute grid, and interpolates.
    Returns a dataframe indexed by a DatetimeIndex with unified sensor columns.
    """
    df = pd.read_csv(path_linkous, encoding="utf-8", sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    # Measurement pairs from the original script
    measurements = [
        ('syote_Aika', 'syote_Arvo_m3/h'),
        ('lampo_nestepuoli_laakeri_Aika', 'lampo_nestepuoli_laakeri_Arvo_°C'),
        ('lampo_kiinteapuoli_laakeri_Aika', 'lampo_kiinteapuoli_laakeri_Arvo_°C'),
        ('tarina_kiinteapuoli_Aika', 'tarina_kiinteapuoli_Arvo_mm/s'),
        ('Momentti_Aika', 'Momentti_Arvo_%'),
        ('Rumpu_nopeus_Aika', 'rumpu_nopeus_Arvo_RPM'),
        ('Ero_nopeus_Aika', 'Ero_nopeus_Arvo_RPM'),
        ('tarina_nestepuoli_Aika', 'tarina_nestepuoli_Arvo_mm/s'),
        ('RUMPU_MOOTTORI_M1_Aika', 'RUMPU_MOOTTORI_M1_Arvo')
    ]

    series_dict = {}
    for time_col, value_col in measurements:
        timestamps = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
        timestamps = timestamps.dt.floor("min")
        values = pd.to_numeric(df[value_col].astype(str).str.replace(",", "."), errors="coerce")
        valid = ~timestamps.isna() & (values >= 0)
        ts = timestamps[valid]
        vs = values[valid]
        s = pd.DataFrame({value_col: vs.values}, index=ts).groupby(level=0).mean()
        series_dict[value_col] = s

    rpm_series = series_dict['rumpu_nopeus_Arvo_RPM']
    start = rpm_series.index.min()
    end = rpm_series.index.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid timestamps in RPM data")

    common_time_index = pd.date_range(start=start, end=end, freq='1min')
    aligned_df = pd.DataFrame(index=common_time_index)
    for col, s in series_dict.items():
        aligned_df[col] = s.reindex(common_time_index).interpolate(method="time")

    key_sensors = [
        'Ero_nopeus_Arvo_RPM',
        'tarina_kiinteapuoli_Arvo_mm/s',
        'tarina_nestepuoli_Arvo_mm/s',
        'Momentti_Arvo_%',
        'RUMPU_MOOTTORI_M1_Arvo'
    ]
    aligned_df.dropna(subset=key_sensors, inplace=True)
    return aligned_df

def save_cleaned(df: pd.DataFrame, out_csv: str):
    df.to_csv(out_csv)
