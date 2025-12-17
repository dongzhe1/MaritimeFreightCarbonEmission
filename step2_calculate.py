import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from tqdm import tqdm
import warnings
from pandarallel import pandarallel
from sklearn.neighbors import BallTree

warnings.filterwarnings('ignore')
pandarallel.initialize(progress_bar=True)

def calculate_distance_haversine(lat1, lon1, lat2, lon2):
    """
    Calculates great-circle distance (nautical miles).
    """
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3440
    return c * r

def calculate_co2_per_ship(df):
    """
    Process function applied to each ship (Group).
    Calculates CO2 and Label.
    """
    df = df.sort_values('BaseDateTime')
    import numpy as np

    def calculate_distance_haversine(lat1, lon1, lat2, lon2):
        """
        Calculates great-circle distance (nautical miles).
        """
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 3440
        return c * r

    v_type = df['VesselType'].iloc[0]
    ship_type = 1 if 80 <= v_type <= 89 else 0

    df['Label'] = np.where(df['SOG'] < 0.5, '1', '0')

    df['TimeDiff'] = df['BaseDateTime'].diff().dt.total_seconds() / 3600
    df['TimeDiff'] = df['TimeDiff'].fillna(0)

    pa = 1776 if ship_type == 0 else 1985
    sfc = 213.1
    emission_rate_stop = pa * sfc * 1e-6

    co2_values = np.zeros(len(df))

    is_stop = df['Label'] == '1'
    co2_values[is_stop] = df.loc[is_stop, 'TimeDiff'] * emission_rate_stop * 3.114

    ps = 9300 if ship_type == 0 else 9400
    emission_rate_move = ps * sfc * 1e-6

    is_move = df['Label'] == '0'

    lat1 = df['LAT']
    lon1 = df['LON']
    lat2 = df['LAT'].shift(-1)
    lon2 = df['LON'].shift(-1)

    raw_dists = calculate_distance_haversine(lat1, lon1, lat2, lon2)
    distances = np.nan_to_num(raw_dists, nan=0)

    speed_mean = (df['SOG'] + df['SOG'].shift(-1)) / 2
    speed_mean = speed_mean.replace(0, 0.1).fillna(0.1)

    valid_move = is_move & (df['Label'].shift(-1) == '0')

    move_emissions = emission_rate_move * (distances / speed_mean.values) * 3.114
    co2_values[valid_move] = move_emissions[valid_move]

    df['CO2'] = co2_values
    df['CO2'] = df['CO2'].fillna(0)

    return df

def find_nearby_ports(row, port_data, threshold=0.2):
    """
    Find nearest port within threshold.
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

def process_all_data(input_file, output_file, port_data, usa_states):
    """
    Reads combined data, calculates emissions per group, performs spatial join, and saves.
    """
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Calculating emissions per vessel...")
    tqdm.pandas(desc="Vessel Calc")

    df = df.groupby('MMSI', group_keys=False).parallel_apply(calculate_co2_per_ship)

    print("Mapping coordinates to States (Spatial Join)...")
    geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf_joined = gpd.sjoin(gdf, usa_states, how='left', predicate='intersects')

    if 'name' in gdf_joined.columns:
        df['State'] = gdf_joined['name']
    else:
        df['State'] = None

    print("Matching Stop points to Ports (Fast Vectorized)...")
    stops_mask = df['Label'] == '1'

    if stops_mask.any():
        port_lat_lon = np.radians(port_data[['Latitude', 'Longitude']].values)
        ship_lat_lon = np.radians(df.loc[stops_mask, ['LAT', 'LON']].values)
        tree = BallTree(port_lat_lon, metric='haversine')

        print(f"Querying nearest ports for {len(ship_lat_lon)} stops...")
        dists, indices = tree.query(ship_lat_lon, k=1)
        threshold_rad = np.radians(0.2)

        dists = dists.flatten()
        indices = indices.flatten()
        valid_mask = dists <= threshold_rad

        subset_idx = df.index[stops_mask]

        final_match_idx = subset_idx[valid_mask]
        final_port_idx = indices[valid_mask]

        if len(final_match_idx) > 0:
            matched_ports = port_data.iloc[final_port_idx]

            df.loc[final_match_idx, 'PortName'] = matched_ports['Main Port Name'].values
            df.loc[final_match_idx, 'State'] = matched_ports['state'].values

    print(f"Saving to {output_file}...")
    cols = [c for c in df.columns if c != 'geometry']
    df[cols].to_csv(output_file, index=False)
    print("Step 2 Complete.")

if __name__ == "__main__":
    YEAR = 2024
    BASE_DIR = os.path.join("E:\\AIS\\data_processed", str(YEAR))

    INPUT_FILE = os.path.join(BASE_DIR, "step1_combined.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "step2_calculated.csv")

    print("Loading auxiliary data...")
    try:
        port_data = pd.read_csv('usa_port_data.csv')
        usa_states = gpd.read_file('map_data/ne_110m_admin_1_states_provinces.shp')
        usa_states = usa_states.to_crs("EPSG:4326")
    except Exception as e:
        print(f"Error loading aux data: {e}")
        exit()

    if os.path.exists(INPUT_FILE):
        process_all_data(INPUT_FILE, OUTPUT_FILE, port_data, usa_states)
    else:
        print(f"Input file not found: {INPUT_FILE}")