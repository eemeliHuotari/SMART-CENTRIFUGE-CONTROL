import pandas as pd

def build_drying_pairs(path_kuivain: str) -> pd.DataFrame:
    """
    Reads Kuivain.csv and constructs wet→dry pairs with mid_time and moisture_lost.
    """
    kuivain = pd.read_csv(path_kuivain, encoding="utf-8-sig", sep=None, engine="python")
    kuivain.columns = kuivain.columns.str.trim() if hasattr(str, "trim") else kuivain.columns.str.strip()
    kuivain.rename(columns={
        kuivain.columns[0]: "timestamp",
        kuivain.columns[1]: "weight_kg",
        kuivain.columns[2]: "quality",
        kuivain.columns[3]: "drying_marker"
    }, inplace=True)

    kuivain['timestamp'] = pd.to_datetime(kuivain['timestamp'], dayfirst=True, errors='coerce')
    kuivain['weight_kg'] = (
        kuivain['weight_kg'].astype(str).str.replace(',', '.', regex=False).astype(float)
    )

    kuivain['label'] = kuivain['drying_marker'].astype(str).str.lower().fillna("").apply(
        lambda x: 'wet' if ('ennen kuivaus' in x or 'ennen kuivatus' in x or 'ennen kuivausta' in x)
        else ('dry' if 'kuiva' in x else None)
    )

    labeled = kuivain.dropna(subset=['label']).sort_values('timestamp')
    wet_points = labeled[labeled['label'] == 'wet'].reset_index(drop=True)
    dry_points = labeled[labeled['label'] == 'dry'].reset_index(drop=True)

    pairs = []
    for _, wet_row in wet_points.iterrows():
        next_dry = dry_points[dry_points['timestamp'] > wet_row['timestamp']]
        if not next_dry.empty:
            dry_row = next_dry.iloc[0]
            pairs.append({
                'wet_time': wet_row['timestamp'],
                'dry_time': dry_row['timestamp'],
                'wet_weight': wet_row['weight_kg'],
                'dry_weight': dry_row['weight_kg'],
            })

    drying_df = pd.DataFrame(pairs)
    if not drying_df.empty:
        drying_df['mid_time'] = drying_df['wet_time'] + (drying_df['dry_time'] - drying_df['wet_time']) / 2
        drying_df['moisture_lost'] = drying_df['wet_weight'] - drying_df['dry_weight']
        drying_df = drying_df[drying_df['moisture_lost'] > 0]
    return drying_df

def save_drying(df: pd.DataFrame, out_csv: str):
    df.to_csv(out_csv, index=False)
