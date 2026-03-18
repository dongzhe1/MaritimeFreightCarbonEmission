# Maritime Freight Carbon Emission & Transfer Analysis

This project reproduces the methodology described in:

> Cheng, C., Li, Z., Yan, Y., Cui, Q., Zhang, Y., & Liu, L. (2024). **Maritime Freight Carbon Emission in the U.S. using AIS data from 2018 to 2022.** *Scientific Data*, 11, 542. https://doi.org/10.1038/s41597-024-03391-0

The pipeline processes AIS vessel trajectory data to compute CO2 emissions for individual cargo and tanker vessels operating in U.S. waters, then aggregates the results into route-level emission records and an inter-state carbon transfer matrix.

---

## Prerequisites — External Data

Three external datasets must be downloaded before running the pipeline.

### A. AIS Vessel Traffic Data
- **Source:** [MarineCadastre.gov — Vessel Traffic Data](https://marinecadastre.gov/ais/) (U.S. Coast Guard / NOAA)
- **What to download:** Daily CSV files for your target year. Navigate to the bulk order site, select the year, and download each day's file.
- **Where to place:** Extract all daily CSV files into `raw_data/<YEAR>/` (e.g. `raw_data/2024/`).
- **Key fields used:** `MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `VesselType`

### B. State Boundary Shapefile
- **Source:** [Natural Earth — Admin 1 States & Provinces (110m)](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)
- **What to download:** `ne_110m_admin_1_states_provinces.zip`
- **Where to place:** Extract all files (`.shp`, `.shx`, `.dbf`, etc.) into `map_data/`.

### C. World Port Index (WPI)
- **Source:** [NGA World Port Index](https://msi.nga.mil/Publications/WPI)
- **What to download:** The CSV version of the WPI publication.
- **Where to place:** Save as `WPI.csv` in the project root directory.

---

## Project Structure

```
MaritimeFreightCarbonEmission/
├── raw_data/
│   └── 2024/               ← daily AIS CSV files go here
├── map_data/               ← extracted Natural Earth shapefile
├── data_processed/
│   └── 2024/               ← all intermediate and final outputs
├── process_wpi.py
├── step1_preprocess.py
├── step2_calculate.py
├── step3_transfer.py
└── WPI.csv
```

---

## Pipeline

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

The core calculation step. For each vessel:

1. **Stop identification** — trajectory points with SOG < 0.3 knots are candidate stop points. Consecutive candidate points form a potential stay segment; segments shorter than `thre_sail` (1 hour) are discarded. Adjacent confirmed segments separated by less than `thre_stop` (1 hour) are merged. Intermediate rows within each confirmed segment are removed, leaving only the segment's start and end points (both labeled `'1'`). All remaining points are labeled `'0'` (move).
2. **Emission calculation** — stop emissions use auxiliary engine power (`pa`); move emissions use main engine power (`ps`). Parameters follow the paper's Equations (1) and (2): `ps = 9300 kW` (cargo) / `9400 kW` (tanker), `pa = 1776 kW` (cargo) / `1985 kW` (tanker), `SFC = 213.1 g/kWh`, `I_CO2 = 3.114 t/t`.
3. **Spatial join** — each trajectory point is mapped to a U.S. state using the shapefile.
4. **Port matching** — stop points are matched to the nearest port in `usa_port_data.csv` within a 0.2° radius (`thre_shold`) using a BallTree.

- **Input:** `data_processed/<YEAR>/step1_combined.csv`, `usa_port_data.csv`, `map_data/`
- **Output:** `data_processed/<YEAR>/step2_calculated.csv`

```bash
python step2_calculate.py
```

### Step 3 — `step3_transfer.py`

Segments each vessel's trajectory into routes (stop-to-stop), then builds the carbon transfer matrix.

A route is defined as the trip between two consecutive port stops. For emissions generated while the vessel transits through an intermediate state C (neither the origin A nor the destination B), those emissions are split equally between A and B as a carbon transfer. The originating and destination states are not counted as transfer recipients for their own territorial emissions.

- **Input:** `data_processed/<YEAR>/step2_calculated.csv`
- **Output:**
  - `data_processed/<YEAR>/Ship_Routes.csv` — one row per route with total emissions and per-state transfer values (matches the format of MOESM1 in the paper)
  - `data_processed/<YEAR>/Carbon_Transfer.csv` — state × state transfer matrix (matches MOESM2)

```bash
python step3_transfer.py
```

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `thre_sail` | 1 hour | Minimum duration for a confirmed stop segment |
| `thre_stop` | 1 hour | Maximum gap between segments before they are merged |
| `thre_shold` | 0.2° | Maximum distance for port matching |

> **Note:** The parameter names `thre_sail` and `thre_stop` appear to be swapped in the original paper's *Code availability* section. The values and their intended meanings above match the paper's methodology description.

---

## Dependencies

```
pandas
geopandas
shapely
numpy
scikit-learn
tqdm
pandarallel
```

Install with:

```bash
pip install pandas geopandas shapely numpy scikit-learn tqdm pandarallel
```
