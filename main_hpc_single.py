import os
import glob
import argparse
import warnings
import multiprocessing as mp

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
    _worker_ports = pd.read_csv(
        os.path.join(aux_data_dir, 'usa_port_data.csv')
    )
    _worker_states = gpd.read_file(
        os.path.join(aux_data_dir, 'map_data',
                     'ne_110m_admin_1_states_provinces.shp')
    ).to_crs("EPSG:4326")


def _read_one_file(f):
    try:
        df = pd.read_csv(f, low_memory=False)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        filtered = df[df['VesselType'].between(70, 89)]
        return filtered if not filtered.empty else pd.DataFrame()
    except Exception as e:
        print(f"  Skipping {f}: {e}")
        return pd.DataFrame()


def _process_one_vessel(args):
    mmsi, df, year = args
    global _worker_states, _worker_ports
    try:
        if len(df) > 3:
            df = apply_median_filter(df)
        df = process_single_vessel(df, _worker_states, _worker_ports)
        routes, transfers = process_route_group(df, year)
        return routes, transfers
    except Exception:
        return [], []


def _read_year(raw_ais_dir, year, n_workers):
    pattern = os.path.join(raw_ais_dir, str(year), "*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found at {pattern}")

    print(f"[{year}] Reading {len(files)} files in parallel "
          f"({n_workers} workers)...")

    with mp.Pool(processes=n_workers) as pool:
        chunks = pool.map(_read_one_file, files,
                          chunksize=max(1, len(files) // n_workers))

    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        raise ValueError(f"[{year}] No valid data after filtering.")

    full_df = pd.concat(non_empty, ignore_index=True)
    print(f"[{year}] {len(full_df):,} records, "
          f"{full_df['MMSI'].nunique():,} vessels.")
    return full_df


def process_year(year, raw_ais_dir, output_dir, aux_data_dir, n_workers):
    full_df = _read_year(raw_ais_dir, year, n_workers)

    groups = [
        (mmsi, grp.copy(), year)
        for mmsi, grp in full_df.groupby('MMSI')
    ]
    total     = len(groups)
    chunksize = max(1, total // (n_workers * 10))
    print(f"[{year}] Dispatching {total} vessels to {n_workers} workers "
          f"(chunksize={chunksize})...")

    all_routes    = []
    all_transfers = []
    done          = 0

    with mp.Pool(
        processes   = n_workers,
        initializer = _init_worker,
        initargs    = (aux_data_dir,),
    ) as pool:
        for routes, transfers in pool.imap_unordered(
            _process_one_vessel, groups, chunksize=chunksize
        ):
            all_routes.extend(routes)
            all_transfers.extend(transfers)
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  [{year}] {done}/{total} vessels processed "
                      f"({100 * done // total}%)")

    print(f"[{year}] Writing results ({len(all_routes)} routes, "
          f"{len(all_transfers)} transfer flows)...")

    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    if all_routes:
        df_routes  = pd.DataFrame(all_routes)
        fixed_cols = ['Year', 'MMSI', 'Start Port', 'Start State',
                      'End Port', 'End State', 'Total Emissions']
        state_cols = sorted(
            [c for c in df_routes.columns if c not in fixed_cols]
        )
        df_routes[state_cols] = df_routes[state_cols].fillna(0)
        df_routes[fixed_cols + state_cols].to_csv(
            os.path.join(year_dir, 'Ship_Routes.csv'), index=False
        )

    if all_transfers:
        df_flows  = pd.DataFrame(
            all_transfers, columns=['Source', 'Dest', 'Amount']
        )
        df_matrix = (
            df_flows
            .groupby(['Source', 'Dest'])['Amount'].sum()
            .reset_index()
            .pivot(index='Source', columns='Dest', values='Amount')
            .fillna(0)
            .reset_index()
        )
        df_matrix.rename(
            columns={'Source': 'States with carbon transfer out'},
            inplace=True
        )
        df_matrix.insert(0, 'Year', year)
        df_matrix.to_csv(
            os.path.join(year_dir, 'Carbon_Transfer.csv'), index=False
        )

    print(f"[{year}] Done. Output: {year_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maritime carbon emission — local multi-core job'
    )
    parser.add_argument(
        '--years', nargs='+', type=int, required=True,
        help='Years to process, e.g. --years 2018 2019 2020'
    )
    parser.add_argument(
        '--raw-ais-dir', required=True,
        help='Root directory containing per-year AIS CSV subdirectories'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Root directory for output Parquet files'
    )
    parser.add_argument(
        '--aux-data-dir', required=True,
        help='Directory containing usa_port_data.csv and map_data/'
    )
    parser.add_argument(
        '--n-workers', type=int, default=mp.cpu_count(),
        help=f'Number of worker processes '
             f'(default: all cores = {mp.cpu_count()})'
    )
    args = parser.parse_args()

    for year in args.years:
        process_year(
            year         = year,
            raw_ais_dir  = args.raw_ais_dir,
            output_dir   = args.output_dir,
            aux_data_dir = args.aux_data_dir,
            n_workers    = args.n_workers,
        )

    print("All years complete.")
