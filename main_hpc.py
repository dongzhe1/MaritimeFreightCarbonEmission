import os
import argparse
import warnings
import glob
import multiprocessing as mp
import traceback

import pandas as pd
import geopandas as gpd
import numpy as np

warnings.filterwarnings('ignore')

from step1_preprocess import apply_median_filter
from step2_calculate import process_single_vessel
from step3_transfer import process_route_group


_worker_states = None
_worker_ports  = None


def _init_worker(aux_data_dir):
    global _worker_states, _worker_ports
    _worker_ports = pd.read_csv(os.path.join(aux_data_dir, 'usa_port_data.csv'))
    _worker_states = gpd.read_file(
        os.path.join(aux_data_dir, 'map_data', 'ne_110m_admin_1_states_provinces.shp')
    ).to_crs("EPSG:4326")


def _process_one_mmsi(args):
    mmsi, df, year = args
    global _worker_states, _worker_ports
    try:
        if len(df) > 3:
            df = apply_median_filter(df)
        df = process_single_vessel(df, _worker_states, _worker_ports)
        routes, transfers = process_route_group(df, year)
        return routes, transfers
    except Exception as e:
        print(f"  [WARN] MMSI {mmsi} failed: {e}\n{traceback.format_exc()}")
        return [], []


def process_month(year, month, lustre_ais_dir, lustre_output_dir, aux_data_dir, n_workers):
    out_dir = os.path.join(lustre_output_dir, str(year))
    if os.path.exists(os.path.join(out_dir, f"Ship_Routes_{month:02d}.parquet")) and \
       os.path.exists(os.path.join(out_dir, f"Carbon_Transfer_{month:02d}.parquet")):
        print(f"[{year}-{month:02d}] Already done, skipping.")
        return

    pattern = os.path.join(lustre_ais_dir, str(year), f"AIS_{year}_{month:02d}_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for {year}-{month:02d}, pattern: {pattern}")
        return

    print(f"[{year}-{month:02d}] Reading {len(files)} files...")
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False,
                             usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'VesselType'])
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
            mask = df['VesselType'].between(70, 89)
            chunks.append(df[mask])
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    if not chunks:
        print(f"[{year}-{month:02d}] No valid data after filtering.")
        return

    full_df = pd.concat(chunks, ignore_index=True)
    del chunks
    n_vessels = full_df['MMSI'].nunique()
    print(f"[{year}-{month:02d}] {len(full_df)} records, "
          f"{n_vessels} vessels. Starting pool...")

    groups = [(mmsi, grp.copy(), year)
              for mmsi, grp in full_df.groupby('MMSI')]
    del full_df

    all_routes    = []
    all_transfers = []
    chunksize = max(1, len(groups) // (n_workers * 4))

    with mp.Pool(processes=n_workers,
                 initializer=_init_worker,
                 initargs=(aux_data_dir,)) as pool:
        for i, (routes, transfers) in enumerate(
            pool.imap_unordered(_process_one_mmsi, groups, chunksize=chunksize)
        ):
            all_routes.extend(routes)
            all_transfers.extend(transfers)
            if (i + 1) % 500 == 0:
                print(f"  [{year}-{month:02d}] {i+1}/{len(groups)} vessels done...")

    os.makedirs(out_dir, exist_ok=True)

    if all_routes:
        df_routes = pd.DataFrame(all_routes)
        fixed_cols = ['Year', 'MMSI', 'Start Port', 'Start State',
                      'End Port', 'End State', 'Total Emissions']
        state_cols = sorted([c for c in df_routes.columns if c not in fixed_cols])
        df_routes[state_cols] = df_routes[state_cols].fillna(0)
        df_routes[fixed_cols + state_cols].to_parquet(
            os.path.join(out_dir, f"Ship_Routes_{month:02d}.parquet"), index=False
        )

    if all_transfers:
        pd.DataFrame(all_transfers, columns=['Source', 'Dest', 'Amount']).to_parquet(
            os.path.join(out_dir, f"Carbon_Transfer_{month:02d}.parquet"), index=False
        )

    print(f"[{year}-{month:02d}] Done. "
          f"{len(all_routes)} routes, {len(all_transfers)} transfer flows.")


def merge_year(year, lustre_output_dir):
    out_dir = os.path.join(lustre_output_dir, str(year))

    route_files    = sorted(glob.glob(os.path.join(out_dir, "Ship_Routes_*.parquet")))
    transfer_files = sorted(glob.glob(os.path.join(out_dir, "Carbon_Transfer_*.parquet")))

    if route_files:
        df_routes = pd.concat([pd.read_parquet(f) for f in route_files], ignore_index=True)
        fixed_cols = ['Year', 'MMSI', 'Start Port', 'Start State',
                      'End Port', 'End State', 'Total Emissions']
        state_cols = sorted([c for c in df_routes.columns if c not in fixed_cols])
        df_routes[state_cols] = df_routes[state_cols].fillna(0)
        df_routes[fixed_cols + state_cols].to_csv(
            os.path.join(out_dir, "Ship_Routes.csv"), index=False
        )
        print(f"[{year}] Ship_Routes.csv written ({len(df_routes)} routes)")

    if transfer_files:
        df_flows = pd.concat([pd.read_parquet(f) for f in transfer_files], ignore_index=True)
        df_matrix = (
            df_flows.groupby(['Source', 'Dest'])['Amount'].sum()
            .reset_index()
            .pivot(index='Source', columns='Dest', values='Amount')
            .fillna(0)
            .reset_index()
        )
        df_matrix.rename(columns={'Source': 'States with carbon transfer out'}, inplace=True)
        df_matrix.insert(0, 'Year', year)
        df_matrix.to_csv(
            os.path.join(out_dir, "Carbon_Transfer.csv"), index=False
        )
        print(f"[{year}] Carbon_Transfer.csv written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime carbon emission — HPC job (Lustre + multiprocessing)")
    parser.add_argument("--year",             type=int, required=True)
    parser.add_argument("--month",            type=int, default=None,
                        help="Month to process (1-12). Omit to run merge instead.")
    parser.add_argument("--lustre-ais-dir",   required=True,
                        help="Lustre root containing per-year AIS CSV subdirectories")
    parser.add_argument("--lustre-output-dir",required=True,
                        help="Lustre root for output Parquet files")
    parser.add_argument("--aux-data-dir",     required=True,
                        help="Directory containing usa_port_data.csv and map_data/")
    parser.add_argument("--n-workers",        type=int,
                        default=mp.cpu_count(),
                        help="Number of worker processes (default: all cores)")
    parser.add_argument("--merge",            action="store_true",
                        help="Merge monthly parquet files into annual output")
    args = parser.parse_args()

    if args.merge:
        merge_year(args.year, args.lustre_output_dir)
    else:
        month = args.month or int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if not month:
            parser.error("Provide --month or run as a SLURM array job.")
        process_month(
            year             = args.year,
            month            = month,
            lustre_ais_dir   = args.lustre_ais_dir,
            lustre_output_dir= args.lustre_output_dir,
            aux_data_dir     = args.aux_data_dir,
            n_workers        = args.n_workers,
        )
