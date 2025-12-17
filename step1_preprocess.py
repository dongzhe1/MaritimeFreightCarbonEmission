import pandas as pd
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def apply_median_filter(df, window_size=3):
    """
    Applies a median filter to smooth specific columns.
    """
    df_filtered = df.copy()
    df_filtered = df_filtered.sort_values('BaseDateTime')

    df_filtered['LAT'] = df_filtered['LAT'].rolling(window=window_size, center=True, min_periods=1).median()
    df_filtered['LON'] = df_filtered['LON'].rolling(window=window_size, center=True, min_periods=1).median()
    df_filtered['SOG'] = df_filtered['SOG'].rolling(window=window_size, center=True, min_periods=1).median()

    return df_filtered

def process_directory(input_dir, output_file, year):
    """
    Reads all CSV files in a directory, filters by vessel type,
    applies median filter per MMSI, and saves to a single combined CSV.
    """
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(all_files)} files. Reading and filtering...")

    chunk_list = []
    for f in tqdm(all_files, desc="Reading Files"):
        try:
            df = pd.read_csv(f)
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
            mask = df['VesselType'].between(70, 89)
            filtered_df = df[mask].copy()

            if not filtered_df.empty:
                keep_cols = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'VesselType', 'Status', 'IMO']
                cols = [c for c in keep_cols if c in filtered_df.columns]
                chunk_list.append(filtered_df[cols])

        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not chunk_list:
        print("No valid data found after filtering.")
        return

    print("Concatenating data...")
    full_df = pd.concat(chunk_list, ignore_index=True)

    print(f"Total records: {len(full_df)}. Grouping by MMSI for smoothing...")

    processed_chunks = []
    grouped = full_df.groupby('MMSI')

    for mmsi, group in tqdm(grouped, desc="Smoothing Ships"):
        if len(group) > 3:
            processed_group = apply_median_filter(group)
            processed_chunks.append(processed_group)
        else:
            processed_chunks.append(group)

    if not processed_chunks:
        return

    final_df = pd.concat(processed_chunks, ignore_index=True)

    print(f"Saving to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Step 1 Complete.")

if __name__ == "__main__":
    YEAR = 2024
    RAW_DATA_DIR = "E:\\AIS\\2024"

    OUTPUT_DIR = os.path.join("E:\\AIS\\data_processed", str(YEAR))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "step1_combined.csv")

    process_directory(RAW_DATA_DIR, OUTPUT_FILE, YEAR)