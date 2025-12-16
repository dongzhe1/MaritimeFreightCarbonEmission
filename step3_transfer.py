import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def transfer_process_route(file_path, year):
    """
    Processes a single ship file to extract:
    1. Route details (for File 1)
    2. Transfer flow tuples (Source, Dest, Amount) (for File 2)
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return [], []

    df['Label'] = df['Label'].astype(str)

    if 'PortName' not in df.columns or 'State' not in df.columns:
        return [], []

    is_at_port = (df['Label'] == '1') & (df['PortName'].notna())
    start_of_stay_mask = is_at_port & (~is_at_port.shift(1, fill_value=False))
    stop_start_indices = df.index[start_of_stay_mask].tolist()

    if len(stop_start_indices) < 2:
        return [], []

    routes_data = []
    transfer_flows = [] 

    for i in range(len(stop_start_indices) - 1):
        idx_start = stop_start_indices[i]
        idx_end = stop_start_indices[i+1]

        
        mmsi = df.loc[idx_start, 'MMSI']
        start_port = df.loc[idx_start, 'PortName']
        start_state = df.loc[idx_start, 'State']
        end_port = df.loc[idx_end, 'PortName']
        end_state = df.loc[idx_end, 'State']

        
        if pd.isna(start_state) or pd.isna(end_state):
            continue

        
        segment_df = df.loc[idx_start : idx_end]
        total_co2 = segment_df['CO2'].sum()

        
        route_entry = {
            'Year': year,
            'MMSI': mmsi,
            'Start Port': start_port,
            'Start State': start_state,
            'End Port': end_port,
            'End State': end_state,
            'Total Emissions': total_co2
        }

        
        states_visited = segment_df['State'].dropna().unique()

        for state in states_visited:
            state_emissions = segment_df.loc[segment_df['State'] == state, 'CO2'].sum()

            
            
            if state_emissions > 0:
                route_entry[state] = state_emissions

            
            
            
            transfer_amount = state_emissions / 2.0

            if transfer_amount > 0:
                
                transfer_flows.append((start_state, state, transfer_amount))
                
                transfer_flows.append((end_state, state, transfer_amount))

        routes_data.append(route_entry)

    return routes_data, transfer_flows

def main_transfer():
    YEAR = 2024 
    BASE_DIR = "./data_processed"

    
    CARGO_DIR = os.path.join(BASE_DIR, str(YEAR), "cargo_stop")
    TANKER_DIR = os.path.join(BASE_DIR, str(YEAR), "tanker_stop")

    all_files = glob.glob(os.path.join(CARGO_DIR, "*.csv")) + \
                glob.glob(os.path.join(TANKER_DIR, "*.csv"))

    print(f"Found {len(all_files)} files. Generating standard outputs...")

    all_routes = []
    all_transfers = [] 

    
    for f in tqdm(all_files, desc="Processing"):
        routes, transfers = transfer_process_route(f, YEAR)
        all_routes.extend(routes)
        all_transfers.extend(transfers)

    if not all_routes:
        print("No routes found.")
        return

    
    print("Generating File 1: Routes Table...")
    df_routes = pd.DataFrame(all_routes)

    
    fixed_cols = ['Year', 'MMSI', 'Start Port', 'Start State', 'End Port', 'End State', 'Total Emissions']
    state_cols = sorted([c for c in df_routes.columns if c not in fixed_cols])

    
    df_routes[state_cols] = df_routes[state_cols].fillna(0)

    
    final_cols = fixed_cols + state_cols
    df_routes = df_routes[final_cols]

    output_1 = os.path.join(BASE_DIR, str(YEAR), "Ship_Routes.csv")
    df_routes.to_csv(output_1, index=False)
    print(f"✅ Saved File 1: {output_1}")

    
    print("Generating File 2: Transfer Matrix...")

    
    df_flows = pd.DataFrame(all_transfers, columns=['Source', 'Dest', 'Amount'])

    
    df_matrix = df_flows.groupby(['Source', 'Dest'])['Amount'].sum().reset_index()

    
    pivot_matrix = df_matrix.pivot(index='Source', columns='Dest', values='Amount').fillna(0)

    
    pivot_matrix.reset_index(inplace=True)
    pivot_matrix.rename(columns={'Source': 'States with carbon transfer out'}, inplace=True)
    pivot_matrix.insert(0, 'Year', YEAR)

    output_2 = os.path.join(BASE_DIR, str(YEAR), "Carbon_Transfer.csv")
    pivot_matrix.to_csv(output_2, index=False)
    print(f"✅ Saved File 2: {output_2}")

if __name__ == "__main__":
    main_transfer()