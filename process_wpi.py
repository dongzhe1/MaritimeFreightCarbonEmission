import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import warnings

warnings.filterwarnings('ignore')

def process_wpi_data(input_csv, shapefile_path, output_csv):
    """
    Reads the NGA World Port Index CSV, filters for U.S. ports,
    and performs a spatial join to append the required 'state' column.
    """
    if not os.path.exists(input_csv):
        print(f"Error: File '{input_csv}' not found.")
        return

    try:
        df = pd.read_csv(input_csv, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df.columns = df.columns.str.upper().str.strip()

    required_cols = ['PORT_NAME', 'LATITUDE', 'LONGITUDE', 'COUNTRY']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing columns. Found: {list(df.columns)}")
        return

    df['COUNTRY'] = df['COUNTRY'].astype(str).str.strip().str.upper()
    us_mask = df['COUNTRY'] == 'US'

    df_us = df[us_mask].copy()
    print(f"Filtered {len(df_us)} U.S. ports from {len(df)} global records.")

    if df_us.empty:
        print("Warning: No ports found with COUNTRY='US'. Check the country codes in your CSV.")
        return

    print("Matching ports to states (Spatial Join)...")

    geometry = [Point(xy) for xy in zip(df_us['LONGITUDE'], df_us['LATITUDE'])]
    gdf_ports = gpd.GeoDataFrame(df_us, geometry=geometry, crs="EPSG:4326")

    try:
        gdf_states = gpd.read_file(shapefile_path)
        gdf_states = gdf_states.to_crs("EPSG:4326")
    except Exception as e:
        print(f"Error loading map data: {e}")
        return

    gdf_joined = gpd.sjoin(gdf_ports, gdf_states[['name', 'geometry']], how='inner', predicate='intersects')

    gdf_joined = gdf_joined.rename(columns={
        'PORT_NAME': 'Main Port Name',
        'LATITUDE': 'Latitude',
        'LONGITUDE': 'Longitude',
        'name': 'state'
    })

    final_df = gdf_joined[['Main Port Name', 'state', 'Latitude', 'Longitude']]

    final_df.to_csv(output_csv, index=False)
    print(f"Success! Saved '{output_csv}' containing {len(final_df)} ports.")
    print(final_df.head())

if __name__ == "__main__":
    INPUT_FILE = "WPI.csv"
    SHAPEFILE = "map_data/ne_110m_admin_1_states_provinces.shp"
    OUTPUT_FILE = "usa_port_data.csv"

    process_wpi_data(INPUT_FILE, SHAPEFILE, OUTPUT_FILE)