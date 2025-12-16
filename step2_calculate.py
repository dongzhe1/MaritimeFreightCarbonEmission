import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import glob
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def calculate_distance_haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth's surface
    using the Haversine formula.

    Returns:
        Distance in nautical miles.
    """
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3440
    return c * r

def calculate_co2(df, ship_type):
    """
    Calculates CO2 emissions for vessel activities (stop and move phases).

    Parameters:
        df (DataFrame): The vessel trajectory data.
        ship_type (int): 0 for Cargo, 1 for Tanker.

    Note on Parameters (from source paper):
        - SFC (Specific Fuel Consumption): 213.1 g/kWh
        - Emission Factor (CO2): 3.114 tonnes/tonne
        - Auxiliary Engine Power: 1776 kW (Cargo), 1985 kW (Tanker)
        - Main Engine Power: 9300 kW (Cargo), 9400 kW (Tanker)
    """
    df = df.replace([np.inf, -np.inf], 0)
    co2_values = np.zeros(len(df))

    df['TimeDiff'] = df['BaseDateTime'].diff().dt.total_seconds() / 3600
    df['TimeDiff'] = df['TimeDiff'].fillna(0)

    pa = 1776 if ship_type == 0 else 1985
    sfc = 213.1
    emission_rate_stop = pa * sfc * 1e-6

    is_stop = df['Label'] == '1'

    co2_values[is_stop] = df.loc[is_stop, 'TimeDiff'] * emission_rate_stop * 3.114

    ps = 9300 if ship_type == 0 else 9400
    emission_rate_move = ps * sfc * 1e-6

    is_move = df['Label'] == '0'

    lat1 = df['LAT']
    lon1 = df['LON']
    lat2 = df['LAT'].shift(-1)
    lon2 = df['LON'].shift(-1)

    distances = calculate_distance_haversine(lat1, lon1, lat2, lon2)
    distances = np.nan_to_num(distances, nan=0)

    speed_mean = (df['SOG'] + df['SOG'].shift(-1)) / 2
    speed_mean = speed_mean.replace(0, 0.1).fillna(0.1)

    valid_move = is_move & (df['Label'].shift(-1) == '0')

    move_emissions = emission_rate_move * (distances / speed_mean.values) * 3.114
    co2_values[valid_move] = move_emissions[valid_move]

    df['CO2'] = co2_values
    df['CO2'] = df['CO2'].fillna(0)

    if 'TimeDiff' in df.columns:
        df.drop(columns=['TimeDiff'], inplace=True)

    return df

def find_nearby_ports(row, port_data, threshold):
    """
    Identifies the nearest port and its corresponding state within a given geographical threshold.

    Returns:
        (Port Name, State Name) or (None, None) if no port is found.
    """
    lat = row['LAT']
    lon = row['LON']

    nearby = port_data[
        (abs(lat - port_data['Latitude']) <= threshold) &
        (abs(lon - port_data['Longitude']) <= threshold)
        ]

    if nearby.empty:
        return None, None

    dists = nearby.apply(lambda r: calculate_distance_haversine(lat, lon, r['Latitude'], r['Longitude']), axis=1)
    nearest_idx = dists.idxmin()
    return nearby.loc[nearest_idx, 'Main Port Name'], nearby.loc[nearest_idx, 'state']

def process_file(file_path, output_dir, port_data, usa_states, ship_type):
    """
    Processes a single AIS CSV file to compute emissions and spatial attributes.

    Steps:
    1. Identify stop/move status.
    2. Calculate CO2 emissions.
    3. Perform spatial join to identify U.S. States.
    4. Match stop points to nearby ports.
    5. Save the result to CSV.
    """
    df = pd.read_csv(file_path)
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    df['Label'] = np.where(df['SOG'] < 0.5, '1', '0')

    df = calculate_co2(df, ship_type)

    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf_joined = gpd.sjoin(gdf, usa_states, how='left', predicate='intersects')

    if 'name' in gdf_joined.columns:
        df['State'] = gdf_joined['name']
    else:
        df['State'] = None

    stops_mask = df['Label'] == '1'
    if stops_mask.any():
        results = df[stops_mask].apply(
            lambda row: find_nearby_ports(row, port_data, threshold=0.2),
            axis=1, result_type='expand'
        )
        if not results.empty:
            df.loc[stops_mask, 'PortName'] = results[0]
            df.loc[stops_mask, 'State'] = results[1].combine_first(df.loc[stops_mask, 'State'])

    filename = os.path.basename(file_path)
    out_path = os.path.join(output_dir, filename)

    cols_to_save = [c for c in df.columns if c != 'geometry']
    df[cols_to_save].to_csv(out_path, index=False)

if __name__ == "__main__":
    YEAR = 2024
    BASE_DIR = "./data_processed"

    CARGO_INPUT = os.path.join(BASE_DIR, str(YEAR), "cargo_mmsi")
    TANKER_INPUT = os.path.join(BASE_DIR, str(YEAR), "tanker_mmsi")

    CARGO_OUTPUT = os.path.join(BASE_DIR, str(YEAR), "cargo_stop")
    TANKER_OUTPUT = os.path.join(BASE_DIR, str(YEAR), "tanker_stop")

    os.makedirs(CARGO_OUTPUT, exist_ok=True)
    os.makedirs(TANKER_OUTPUT, exist_ok=True)

    print("Loading auxiliary data...")
    try:
        port_data = pd.read_csv('usa_port_data.csv')
        usa_states = gpd.read_file('map_data/ne_110m_admin_1_states_provinces.shp')
        usa_states = usa_states.to_crs("EPSG:4326")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'usa_port_data.csv' and 'map_data' folder exist in the working directory.")
        exit()

    files = glob.glob(os.path.join(CARGO_INPUT, "*.csv"))
    print(f"Found {len(files)} Cargo ships.")
    for f in tqdm(files, desc="Processing Cargo"):
        process_file(f, CARGO_OUTPUT, port_data, usa_states, ship_type=0)

    files = glob.glob(os.path.join(TANKER_INPUT, "*.csv"))
    print(f"Found {len(files)} Tanker ships.")
    for f in tqdm(files, desc="Processing Tanker"):
        process_file(f, TANKER_OUTPUT, port_data, usa_states, ship_type=1)

    print("Step 2 Calculation Complete!")