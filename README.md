# Maritime Freight Carbon Emission & Transfer Analysis

This project reproduces the methodology described in:

> Cheng, C., Li, Z., Yan, Y., Cui, Q., Zhang, Y., & Liu, L. (2024). **Maritime Freight Carbon Emission in the U.S. using AIS data from 2018 to 2022.** *Scientific Data*, 11, 542. https://doi.org/10.1038/s41597-024-03391-0

The pipeline processes AIS vessel trajectory data to compute CO2 emissions for individual cargo and tanker vessels operating in U.S. waters, then aggregates the results into route-level emission records and an inter-state carbon transfer matrix.

---

## Prerequisites — External Data

Three external datasets must be downloaded before running the pipeline.

### A. AIS Vessel Traffic Data
- **Source:** [MarineCadastre.gov — Vessel Traffic Data](https://marinecadastre.gov/ais/) (U.S. Coast Guard / NOAA)
- **What to download:** Daily CSV files for your target year via the bulk order site.
- **Where to place (single machine):** `raw_data/<YEAR>/`
- **Where to place (Spark):** Upload to HDFS under the directory passed as `--hdfs-ais-dir`, one subdirectory per year (e.g. `hdfs:///ais/2024/`).
- **Key fields used:** `MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `VesselType`

### B. State Boundary Shapefile
- **Source:** [Natural Earth — Admin 1 States & Provinces (110m)](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)
- **What to download:** `ne_110m_admin_1_states_provinces.zip`
- **Where to place:** Extract all files into `map_data/`.

### C. World Port Index (WPI)
- **Source:** [NGA World Port Index](https://msi.nga.mil/Publications/WPI)
- **What to download:** The CSV version of the WPI publication.
- **Where to place:** Save as `WPI.csv` in the project root.

---

## Project Structure

```
MaritimeFreightCarbonEmission/
├── raw_data/
│   └── 2024/               ← daily AIS CSV files (single machine)
├── map_data/               ← extracted Natural Earth shapefile
├── data_processed/
│   └── 2024/               ← intermediate and final outputs (single machine)
├── process_wpi.py          ← one-time aux data preparation
├── step1_preprocess.py     ← data cleaning and smoothing
├── step2_calculate.py      ← stop identification, emission calculation, spatial join
├── step3_transfer.py       ← route segmentation and carbon transfer matrix
├── main_spark.py           ← distributed Spark job (replaces steps 1–3)
└── WPI.csv
```

---

## Single-Machine Pipeline (Validation)

Run the four scripts in order. All paths are relative to the project root.

### Step 0 — `process_wpi.py` (run once)

Filters the WPI for U.S. ports and uses a spatial join to append the correct state name to each port.

- **Input:** `WPI.csv`, `map_data/ne_110m_admin_1_states_provinces.shp`
- **Output:** `usa_port_data.csv`

```bash
python process_wpi.py
```

### Step 1 — `step1_preprocess.py`

Reads all daily AIS CSV files, filters for cargo vessels (VesselType 70–79) and tankers (VesselType 80–89), and applies a median filter (window = 3) per vessel to smooth GPS noise in LAT, LON, and SOG.

- **Input:** `raw_data/<YEAR>/*.csv`
- **Output:** `data_processed/<YEAR>/step1_combined.csv`

```bash
python step1_preprocess.py
```

### Step 2 — `step2_calculate.py`

For each vessel: identifies stop/move segments, calculates CO2 emissions, maps points to U.S. states, and matches stops to the nearest port.

- **Input:** `data_processed/<YEAR>/step1_combined.csv`, `usa_port_data.csv`, `map_data/`
- **Output:** `data_processed/<YEAR>/step2_calculated.csv`

```bash
python step2_calculate.py
```

### Step 3 — `step3_transfer.py`

Segments trajectories into routes (stop-to-stop), builds the carbon transfer matrix.

- **Input:** `data_processed/<YEAR>/step2_calculated.csv`
- **Output:** `data_processed/<YEAR>/Ship_Routes.csv`, `data_processed/<YEAR>/Carbon_Transfer.csv`

```bash
python step3_transfer.py
```

---

## Spark Pipeline (Distributed)

`main_spark.py` consolidates steps 1–3 into a single distributed job. It imports the core per-vessel logic directly from the existing step files (`apply_median_filter`, `process_single_vessel`, `process_route_group`), so both pipelines share the same computation code.

### Prerequisites

1. Run `process_wpi.py` once on any machine to produce `usa_port_data.csv`.
2. The driver node must have `usa_port_data.csv` and `map_data/` accessible on its **local filesystem** (the shapefile is broadcast to workers from the driver).
3. AIS CSV files must be uploaded to HDFS, one subdirectory per year.

### Package the virtual environment

```bash
module load python/3.x        # use your cluster's Modulair command
python -m venv myenv
source myenv/bin/activate
pip install -r requirement.txt
venv-pack -o myenv.tar.gz
```

### Download to HDFS with Spark
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --archives myenv.tar.gz \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
  --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./environment/bin/python \
  download_spark.py \
  --years 2022 2023 2024 \
  --hdfs-base-dir hdfs:///ais
```

### Submit

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --archives myenv.tar.gz \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./myenv/bin/python \
  --conf spark.executorEnv.PYSPARK_PYTHON=./myenv/bin/python \
  --py-files step1_preprocess.py,step2_calculate.py,step3_transfer.py \
  --executor-memory 8g \
  --num-executors 20 \
  main_spark.py \
  --years 2022 2023 2024 \
  --hdfs-ais-dir   hdfs:///ais \
  --aux-data-dir   /local/path/to/project \
  --hdfs-output-dir hdfs:///ais/output
```

### Outputs (per year, in Parquet)

| File | Contents |
|---|---|
| `Ship_Routes.parquet` | One row per voyage: year, MMSI, start/end port and state, total emissions |
| `Carbon_Transfer.parquet` | Long-format transfer flows: year, source\_state, dest\_state, amount |

The `Carbon_Transfer.parquet` is in long format (one row per state pair). To reproduce the paper's wide pivot matrix, read it into pandas and call `.pivot()`.

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `thre_sail` | 1 hour | Minimum duration for a confirmed stop segment |
| `thre_stop` | 1 hour | Maximum gap between segments before they are merged |
| `thre_shold` | 0.2° | Maximum distance for port matching |

> **Note:** The parameter names `thre_sail` and `thre_stop` appear to be swapped in the original paper's *Code availability* section. The values and intended meanings above follow the paper's methodology description.

---

## Dependencies

```
pandas
geopandas
shapely
numpy
scikit-learn
tqdm
pyspark          # Spark only
venv-pack        # Spark only
pyarrow          # Spark only
requests         # Spark only
```

Install:

```bash
pip install -r requirement.txt
```
