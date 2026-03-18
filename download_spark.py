import os
import argparse
import warnings
import datetime
import tempfile
import zipfile

warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
import requests
import pyarrow.fs as fs

def download_and_upload_partition(iterator):
    hdfs = fs.HadoopFileSystem("default")

    with requests.Session() as session:
        for row in iterator:
            year = row['year']
            date_str = row['date_str']
            hdfs_base_dir = row['hdfs_base_dir']

            base_url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}"
            zip_filename = f"AIS_{date_str}.zip"
            url = f"{base_url}/{zip_filename}"
            hdfs_dir = f"{hdfs_base_dir}/{year}"

            with tempfile.TemporaryDirectory() as tmpdir:
                local_zip = os.path.join(tmpdir, zip_filename)

                try:
                    with session.get(url, stream=True, timeout=60) as response:
                        response.raise_for_status()
                        with open(local_zip, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192 * 4):
                                f.write(chunk)

                    uploaded = False
                    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                        for filename in zip_ref.namelist():
                            if filename.endswith(".csv"):
                                hdfs_path = f"{hdfs_dir}/{filename}"

                                with hdfs.open_output_stream(hdfs_path) as out_f:
                                    with zip_ref.open(filename) as in_f:
                                        while True:
                                            chunk = in_f.read(1024 * 1024)
                                            if not chunk:
                                                break
                                            out_f.write(chunk)

                                uploaded = True
                                yield f"SUCCESS: {zip_filename} -> {hdfs_path}"

                    if not uploaded:
                        yield f"FAILED: {zip_filename} | Error: No CSV found in zip"

                except Exception as e:
                    yield f"FAILED: {zip_filename} | Error: {str(e)}"

                finally:
                    if os.path.exists(local_zip):
                        os.remove(local_zip)

def run(spark, years, hdfs_base_dir):
    for year in years:
        print(f"=== Preparing download tasks for year {year} ===")

        hdfs = fs.HadoopFileSystem("default")
        target_hdfs_dir = f"{hdfs_base_dir}/{year}"
        hdfs.create_dir(target_hdfs_dir)

        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        delta = datetime.timedelta(days=1)

        tasks = []
        curr = start_date
        while curr <= end_date:
            tasks.append({
                'year': year,
                'date_str': curr.strftime("%Y_%m_%d"),
                'hdfs_base_dir': hdfs_base_dir
            })
            curr += delta

        tasks_rdd = spark.sparkContext.parallelize(tasks, numSlices=len(tasks))
        results = tasks_rdd.mapPartitions(download_and_upload_partition).collect()

        success_count = sum(1 for r in results if r.startswith("SUCCESS"))
        print(f"Year {year} summary: {success_count}/{len(tasks)} files successfully downloaded and uploaded.")
        for r in results:
            if r.startswith("FAILED"):
                print(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed AIS Data Downloader (Optimized)")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--hdfs-base-dir", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("AIS_Data_Downloader_Fast").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    run(spark, args.years, args.hdfs_base_dir)
    spark.stop()