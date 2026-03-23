import os
import argparse
import warnings
import datetime
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

import requests


def download_one_day(task):
    year        = task['year']
    date_str    = task['date_str']
    output_dir  = task['output_dir']

    base_url    = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}"
    zip_name    = f"AIS_{date_str}.zip"
    url         = f"{base_url}/{zip_name}"

    try:
        with requests.Session() as session:
            with session.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_zip = os.path.join(tmpdir, zip_name)
                    with open(local_zip, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192 * 4):
                            f.write(chunk)

                    uploaded = False
                    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                        for filename in zip_ref.namelist():
                            if filename.endswith('.csv'):
                                dest = os.path.join(output_dir, filename)
                                with zip_ref.open(filename) as src, open(dest, 'wb') as dst:
                                    while True:
                                        chunk = src.read(1024 * 1024)
                                        if not chunk:
                                            break
                                        dst.write(chunk)
                                uploaded = True

                    if uploaded:
                        return f"SUCCESS: {zip_name}"
                    return f"FAILED: {zip_name} | Error: No CSV found in zip"

    except Exception as e:
        return f"FAILED: {zip_name} | Error: {str(e)}"


def run(years, lustre_base_dir, max_workers):
    for year in years:
        print(f"=== Preparing download tasks for year {year} ===")

        output_dir = os.path.join(lustre_base_dir, str(year))
        os.makedirs(output_dir, exist_ok=True)

        start_date = datetime.date(year, 1, 1)
        end_date   = datetime.date(year, 12, 31)
        delta      = datetime.timedelta(days=1)

        tasks = []
        curr = start_date
        while curr <= end_date:
            tasks.append({
                'year':       year,
                'date_str':   curr.strftime("%Y_%m_%d"),
                'output_dir': output_dir,
            })
            curr += delta

        success = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one_day, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result.startswith("SUCCESS"):
                    success += 1
                else:
                    print(result)

        print(f"Year {year} summary: {success}/{len(tasks)} files downloaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel AIS Data Downloader for HPC (Lustre)")
    parser.add_argument("--years",          nargs="+", type=int, required=True,
                        help="Years to download, e.g. --years 2022 2023 2024")
    parser.add_argument("--lustre-base-dir", required=True,
                        help="Lustre root directory, e.g. /scratch/user/ais")
    parser.add_argument("--max-workers",    type=int, default=16,
                        help="Number of parallel download threads (default: 16)")
    args = parser.parse_args()

    run(args.years, args.lustre_base_dir, args.max_workers)
