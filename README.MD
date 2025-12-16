# Maritime Carbon Emission & Transfer Analysis

This project reproduces the methodology for calculating maritime carbon emissions and analyzing inter-state carbon transfers within the United States. It processes AIS (Automatic Identification System) data to compute emissions for individual vessels and aggregates them into route-based carbon transfer matrices.

## 1. Data Sources (Prerequisites)

Before running the pipeline, ensure you have downloaded the following **real-world datasets**.

### A. Vessel Traffic Data (AIS)
* **Source:** [MarineCadastre.gov](https://marinecadastre.gov/ais/)
* **Description:** Daily CSV files containing vessel location (LAT/LON), speed (SOG), and identification (MMSI).
* **Action:** Download the CSV files for your target year (e.g., 2024) and place them in your raw data directory (referenced in `step1_preprocess.py`).

### B. Map Data (Shapefile)
* **Source:** [Natural Earth Data](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)
* **File:** Admin 1 – States, Provinces (`ne_110m_admin_1_states_provinces.zip`)
* **Description:** A polygon shapefile defining the boundaries of U.S. states. Used to spatially identify which state a vessel is currently located in.
* **Action:** Extract the `.shp`, `.shx`, `.dbf`, etc., into the `map_data/` folder.

### C. Port Data (World Port Index)
* **Source:** [NGA World Port Index (WPI)](https://msi.nga.mil/Publications/WPI)
* **Description:** A comprehensive list of global ports.
* **Action:** Download the **CSV** version of the WPI. Save it as `WPI.csv` in the root directory. This will be processed by `process_wpi.py`.

---

## 2. Project Structure & Scripts

The pipeline consists of four main scripts that must be executed in order.

### `0. process_wpi.py` (Auxiliary Setup)
* **Purpose:** Prepares the port reference dataset.
* **Input:** `WPI.csv` (Raw NGA data) + `map_data/` (Shapefile).
* **Process:** Filters for U.S. ports and uses a **Spatial Join** with the shapefile to append the correct "State" name to each port (since the raw WPI lacks state labels).
* **Output:** `usa_port_data.csv` (Standardized port list used by Step 2).

### `1. step1_preprocess.py` (Data Cleaning)
* **Purpose:** Cleans raw AIS data and splits it by vessel.
* **Input:** Raw daily AIS CSV files from MarineCadastre.
* **Process:**
    * Filters for specific vessel types (Cargo: 70-79, Tanker: 80-89).
    * Splits the massive daily files into individual files per ship (MMSI).
    * Sorts records by timestamp (`BaseDateTime`).
    * Applies a median filter to smooth GPS noise.
* **Output:** Individual CSV files in `data_processed/{YEAR}/cargo_mmsi/` and `tanker_mmsi/`.

### `2. step2_calculate.py` (Core Calculation)
* **Purpose:** Calculates CO2 emissions and identifies spatial location.
* **Input:** Individual MMSI files from Step 1 + `usa_port_data.csv` + `map_data/`.
* **Process:**
    * **Distance:** Calculates Great-circle distance (Haversine formula) between consecutive points.
    * **Emission:** Computes CO2 based on engine power, specific fuel consumption (SFC), and time duration ($T_{i+1} - T_i$). Differentiates between **Stop** (Auxiliary engine only) and **Move** (Main + Auxiliary engines) phases.
    * **Spatial Join:** Maps every GPS point to a U.S. State using the shapefile.
    * **Port Matching:** Matches "Stop" points to the nearest port within a threshold distance.
* **Output:** Processed files in `data_processed/{YEAR}/cargo_stop/` and `tanker_stop/` containing calculated `CO2`, `State`, and `PortName`.

### `3. step3_transfer.py` (Route & Transfer Analysis)
* **Purpose:** Segments trajectories into routes and aggregates carbon transfers.
* **Input:** Calculated files from Step 2.
* **Process:**
    * **Route Segmentation:** Defines a route as a trip between two consecutive port stops (Stop A -> Stop B).
    * **Carbon Attribution:** Aggregates total emissions for the route.
    * **Transfer Logic:** For any emission generated in an intermediate state (State C), the responsibility is split 50/50 between the Start State (A) and End State (B).
* **Output:**
    * `Ship_Routes_2018-2022_Style.csv`: Detailed list of all voyages and emissions per state (matches *MOESM1* format).
    * `Carbon_Transfer_2018-2022_Style.csv`: A state-by-state matrix showing carbon transfer flows (matches *MOESM2* format).

---