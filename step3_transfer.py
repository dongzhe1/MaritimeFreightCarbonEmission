import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def process_route_group(df_group, year):
    """
    Processes a single ship's dataframe to extract routes and transfers.
    """
    df_group = df_group.sort_values('BaseDateTime')

    if 'PortName' not in df_group.columns or 'State' not in df_group.columns:
        return [], []

    is_at_port = (df_group['Label'].astype(str) == '1') & (df_group['PortName'].notna())

    start_of_stay_mask = is_at_port & (~is_at_port.shift(1, fill_value=False))
    stop_start_indices = df_group.index[start_of_stay_mask].tolist()

    if len(stop_start_indices) < 2:
        return [], []

    routes_data = []
    transfer_flows = []

    for i in range(len(stop_start_indices) - 1):
        idx_start = stop_start_indices[i]
        idx_end = stop_start_indices[i+1]

        mmsi = df_group.loc[idx_start, 'MMSI']
        start_port = df_group.loc[idx_start, 'PortName']
        start_state = df_group.loc[idx_start, 'State']
        end_port = df_group.loc[idx_end, 'PortName']
        end_state = df_group.loc[idx_end, 'State']

        if pd.isna(start_state) or pd.isna(end_state):
            continue

        segment_df = df_group.loc[idx_start : idx_end]
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
            if state == start_state or state == end_state:
                continue

            state_emissions = segment_df.loc[segment_df['State'] == state, 'CO2'].sum()

            if state_emissions > 0:
                route_entry[state] = state_emissions

                transfer_amount = state_emissions / 2.0
                transfer_flows.append((start_state, state, transfer_amount))
                transfer_flows.append((end_state, state, transfer_amount))

        routes_data.append(route_entry)

    return routes_data, transfer_flows

def main_transfer():
    YEAR = 2024
    BASE_DIR = os.path.join("data_processed", str(YEAR))
    INPUT_FILE = os.path.join(BASE_DIR, "step2_calculated.csv")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    all_routes = []
    all_transfers = []

    print("Processing Routes by MMSI...")
    grouped = df.groupby('MMSI')

    for mmsi, group in tqdm(grouped, total=len(grouped)):
        routes, transfers = process_route_group(group, YEAR)
        all_routes.extend(routes)
        all_transfers.extend(transfers)

    if not all_routes:
        print("No routes identified.")
        return

    print("Generating Routes Report...")
    df_routes = pd.DataFrame(all_routes)
    fixed_cols = ['Year', 'MMSI', 'Start Port', 'Start State', 'End Port', 'End State', 'Total Emissions']
    state_cols = sorted([c for c in df_routes.columns if c not in fixed_cols])
    df_routes[state_cols] = df_routes[state_cols].fillna(0)

    out1 = os.path.join(BASE_DIR, "Ship_Routes.csv")
    df_routes[fixed_cols + state_cols].to_csv(out1, index=False)

    print("Generating Transfer Matrix...")
    df_flows = pd.DataFrame(all_transfers, columns=['Source', 'Dest', 'Amount'])
    df_matrix = df_flows.groupby(['Source', 'Dest'])['Amount'].sum().reset_index()

    pivot_matrix = df_matrix.pivot(index='Source', columns='Dest', values='Amount').fillna(0)
    pivot_matrix.reset_index(inplace=True)
    pivot_matrix.rename(columns={'Source': 'States with carbon transfer out'}, inplace=True)
    pivot_matrix.insert(0, 'Year', YEAR)

    out2 = os.path.join(BASE_DIR, "Carbon_Transfer.csv")
    pivot_matrix.to_csv(out2, index=False)

    print("Step 3 Complete.")

if __name__ == "__main__":
    main_transfer()
