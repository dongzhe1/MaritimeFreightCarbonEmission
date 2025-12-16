import pandas as pd
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def apply_median_filter(df, window_size=3):
    """
    Applies a median filter to smooth specific columns as mentioned in the paper.
    Paper reference: "Aberrant records were effectively mitigated using a median filter"[cite: 88].
    """
    df_filtered = df.copy()

    df_filtered['LAT'] = df['LAT'].rolling(window=window_size, center=True, min_periods=1).median()
    df_filtered['LON'] = df['LON'].rolling(window=window_size, center=True, min_periods=1).median()
    df_filtered['SOG'] = df['SOG'].rolling(window=window_size, center=True, min_periods=1).median()

    return df_filtered

def process_ais_data(input_file_path, output_base_dir, year):
    """
    Reads raw AIS data, filters by VesselType, and splits into individual MMSI files.
    """
    print(f"Loading data from: {input_file_path}...")

    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    cargo_mask = df['VesselType'].between(70, 79)
    tanker_mask = df['VesselType'].between(80, 89)

    df_cargo = df[cargo_mask].copy()
    df_tanker = df[tanker_mask].copy()

    print(f"Total records: {len(df)}")
    print(f"Cargo records: {len(df_cargo)}")
    print(f"Tanker records: {len(df_tanker)}")

    cargo_out_dir = os.path.join(output_base_dir, str(year), 'cargo_mmsi')
    tanker_out_dir = os.path.join(output_base_dir, str(year), 'tanker_mmsi')

    os.makedirs(cargo_out_dir, exist_ok=True)
    os.makedirs(tanker_out_dir, exist_ok=True)

    print("Processing Cargo ships...")
    process_group(df_cargo, cargo_out_dir)

    print("Processing Tankers...")
    process_group(df_tanker, tanker_out_dir)

    print("Preprocessing complete!")

def process_group(df_subset, output_dir):
    """
    Splits the dataframe by MMSI, sorts by time, applies filter, and saves to CSV.
    """
    if df_subset.empty:
        print("No data found for this category.")
        return

    grouped = df_subset.groupby('MMSI')

    for mmsi, group in tqdm(grouped, total=len(grouped)):
        group = group.sort_values('BaseDateTime')

        if len(group) > 3:
            group = apply_median_filter(group)

        filename = f"{mmsi}.csv"
        file_path = os.path.join(output_dir, filename)

        cols_to_keep = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'VesselType']
        group[cols_to_keep].to_csv(file_path, index=False)

if __name__ == "__main__":
    INPUT_FILE = "AIS_2024_01_01.csv"
    OUTPUT_DIR = "./data_processed"

    YEAR = 2024

    if os.path.exists(INPUT_FILE):
        process_ais_data(INPUT_FILE, OUTPUT_DIR, YEAR)
    else:
        print(f"File not found: {INPUT_FILE}. Please check the path.")