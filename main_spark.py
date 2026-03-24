import os
import argparse
import warnings
import json
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, DoubleType, TimestampType
)

from step1_preprocess import apply_median_filter
from step2_calculate import process_single_vessel
from step3_transfer import process_route_group


TRAJECTORY_SCHEMA = StructType([
    StructField("MMSI",         LongType(),      True),
    StructField("BaseDateTime", TimestampType(), True),
    StructField("LAT",          DoubleType(),    True),
    StructField("LON",          DoubleType(),    True),
    StructField("SOG",          DoubleType(),    True),
    StructField("VesselType",   IntegerType(),   True),
    StructField("Label",        StringType(),    True),
    StructField("TimeDiff",     DoubleType(),    True),
    StructField("CO2",          DoubleType(),    True),
    StructField("State",        StringType(),    True),
    StructField("PortName",     StringType(),    True),
])

OUTPUT_SCHEMA = StructType([
    StructField("record_type",     StringType(),  True),
    StructField("year",            IntegerType(), True),
    StructField("mmsi",            LongType(),    True),
    StructField("start_port",      StringType(),  True),
    StructField("start_state",     StringType(),  True),
    StructField("end_port",        StringType(),  True),
    StructField("end_state",       StringType(),  True),
    StructField("total_emissions", DoubleType(),  True),
    StructField("state_emissions", StringType(),  True),
    StructField("transfer_source", StringType(),  True),
    StructField("transfer_dest",   StringType(),  True),
    StructField("transfer_amount", DoubleType(),  True),
])

_OUTPUT_COLS  = [f.name for f in OUTPUT_SCHEMA.fields]
_FIXED_ROUTE_COLS = {'Year', 'MMSI', 'Start Port', 'Start State',
                     'End Port', 'End State', 'Total Emissions'}


def extract_routes_and_transfers(df, year):
    routes, transfers = process_route_group(df, year)
    if not routes and not transfers:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    rows = []
    for r in routes:
        state_cols = {k: float(v) for k, v in r.items() if k not in _FIXED_ROUTE_COLS}
        rows.append({
            'record_type':     'route',
            'year':            int(r['Year']),
            'mmsi':            int(r['MMSI']),
            'start_port':      r['Start Port'],
            'start_state':     r['Start State'],
            'end_port':        r['End Port'],
            'end_state':       r['End State'],
            'total_emissions': float(r['Total Emissions']),
            'state_emissions': json.dumps(state_cols) if state_cols else None,
            'transfer_source': None,
            'transfer_dest':   None,
            'transfer_amount': 0.0,
        })
    for src, dst, amt in transfers:
        rows.append({
            'record_type':     'transfer',
            'year':            year,
            'mmsi':            None,
            'start_port':      None,
            'start_state':     None,
            'end_port':        None,
            'end_state':       None,
            'total_emissions': 0.0,
            'state_emissions': None,
            'transfer_source': str(src),
            'transfer_dest':   str(dst),
            'transfer_amount': float(amt),
        })
    return pd.DataFrame(rows)


def run(spark, years, hdfs_ais_dir, aux_data_dir, hdfs_output_dir):
    port_data  = pd.read_csv(os.path.join(aux_data_dir, 'usa_port_data.csv'))
    gdf_states = gpd.read_file(
        os.path.join(aux_data_dir, 'map_data', 'ne_110m_admin_1_states_provinces.shp')
    ).to_crs("EPSG:4326")

    bc_ports = spark.sparkContext.broadcast(port_data)

    states_df = pd.DataFrame(gdf_states.drop(columns='geometry'))
    states_df['geometry_wkb'] = gdf_states['geometry'].apply(lambda g: g.wkb_hex)
    bc_states = spark.sparkContext.broadcast(states_df)

    for year in years:
        print(f"=== Processing year {year} ===")
        input_path  = f"{hdfs_ais_dir}/{year}"
        output_path = f"{hdfs_output_dir}/{year}"

        raw_df = (
            spark.read.csv(input_path, header=True, inferSchema=False)
            .select(
                col("MMSI").cast(LongType()),
                col("BaseDateTime"),
                col("LAT").cast(DoubleType()),
                col("LON").cast(DoubleType()),
                col("SOG").cast(DoubleType()),
                col("VesselType").cast(IntegerType()),
            )
            .withColumn("BaseDateTime", to_timestamp("BaseDateTime"))
            .filter(col("VesselType").between(70, 89))
            .repartition(col("MMSI"))
        )

        def traj_udf(df):
            from shapely import wkb as shp_wkb
            states_raw = bc_states.value.copy()
            states_raw['geometry'] = states_raw['geometry_wkb'].apply(
                lambda h: shp_wkb.loads(bytes.fromhex(h))
            )
            gdf = gpd.GeoDataFrame(
                states_raw.drop(columns='geometry_wkb'), geometry='geometry', crs='EPSG:4326'
            )
            if len(df) > 3:
                df = apply_median_filter(df)
            return process_single_vessel(df, gdf, bc_ports.value)

        traj_df = (
            raw_df
            .groupBy("MMSI")
            .applyInPandas(traj_udf, schema=TRAJECTORY_SCHEMA)
            .repartition(col("MMSI"))
            .cache()
        )

        def route_udf(df):
            return extract_routes_and_transfers(df, year)

        combined_df = (
            traj_df
            .groupBy("MMSI")
            .applyInPandas(route_udf, schema=OUTPUT_SCHEMA)
            .cache()
        )

        (
            combined_df
            .filter(col("record_type") == "route")
            .select("year", "mmsi", "start_port", "start_state",
                    "end_port", "end_state", "total_emissions", "state_emissions")
            .write.mode("overwrite")
            .parquet(f"{output_path}/Ship_Routes.parquet")
        )

        (
            combined_df
            .filter(col("record_type") == "transfer")
            .groupBy("year", "transfer_source", "transfer_dest")
            .agg({"transfer_amount": "sum"})
            .withColumnRenamed("sum(transfer_amount)", "amount")
            .withColumnRenamed("transfer_source", "source_state")
            .withColumnRenamed("transfer_dest", "dest_state")
            .write.mode("overwrite")
            .parquet(f"{output_path}/Carbon_Transfer.parquet")
        )

        traj_df.unpersist()
        combined_df.unpersist()

        print(f"Year {year} complete. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime carbon emission — Spark job")
    parser.add_argument(
        "--years", nargs="+", type=int, required=True,
        help="Years to process, e.g. --years 2022 2023 2024"
    )
    parser.add_argument(
        "--hdfs-ais-dir", required=True,
        help="HDFS root containing per-year AIS CSV subdirectories, e.g. hdfs:///ais"
    )
    parser.add_argument(
        "--aux-data-dir", required=True,
        help="Local path on the driver containing usa_port_data.csv and map_data/"
    )
    parser.add_argument(
        "--hdfs-output-dir", required=True,
        help="HDFS root for output Parquet files, e.g. hdfs:///ais/output"
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("MaritimeCarbonEmission")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    run(spark, args.years, args.hdfs_ais_dir, args.aux_data_dir, args.hdfs_output_dir)
    spark.stop()
