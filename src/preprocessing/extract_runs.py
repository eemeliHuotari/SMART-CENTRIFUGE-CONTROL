import pandas as pd

TANK_CAPACITY_LITERS = 30000  # 30 m³

def extract_runs_from_tank(path_pinta: str) -> pd.DataFrame:
    """
    Reads pesusailio_pinta.csv, pairs 'aloitus' with the next 'lopetus' and
    computes feed volume from level % difference.
    """
    df = pd.read_csv(path_pinta, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip()

    df['Pesusailio_pinta_Arvo_%'] = df['Pesusailio_pinta_Arvo_%'].astype(str).str.replace(',', '.').astype(float)
    df['pesusailio_Aika'] = pd.to_datetime(df['pesusailio_Aika'], dayfirst=True, errors='coerce')

    starts = df[df['linkous_aloitus_lopetus'].str.lower() == 'aloitus'].copy()
    ends = df[df['linkous_aloitus_lopetus'].str.lower() == 'lopetus'].copy()

    runs = []
    for _, start in starts.iterrows():
        following_ends = ends[ends["pesusailio_Aika"] > start["pesusailio_Aika"]]
        if not following_ends.empty:
            end = following_ends.iloc[0]
            runs.append({
                "start_time": start["pesusailio_Aika"],
                "end_time": end["pesusailio_Aika"],
                "start_fill_pct": start["Pesusailio_pinta_Arvo_%"],
                "end_fill_pct": end["Pesusailio_pinta_Arvo_%"]
            })
            ends = ends.drop(index=end.name)

    centrifuge_df = pd.DataFrame(runs)
    centrifuge_df["feed_liters"] = (
        (centrifuge_df["start_fill_pct"] - centrifuge_df["end_fill_pct"]) / 100.0
    ) * TANK_CAPACITY_LITERS
    return centrifuge_df

def save_runs(df: pd.DataFrame, out_csv: str):
    df.to_csv(out_csv, index=False)