#!/bin/bash
# ============================================================
# submit_download.slurm
# Downloads one year of AIS data to Lustre.
# Usage: sbatch submit_download.slurm
# ============================================================

#SBATCH --job-name=ais_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu

# ---- user settings ----------------------------------------
YEARS="2022 2023 2024"
LUSTRE_BASE="$SCRATCH/ais"
MAX_WORKERS=16
PROJECT_DIR="$SCRATCH/MaritimeFreightCarbonEmission"
# -----------------------------------------------------------

mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source "$PROJECT_DIR/.venv/bin/activate"

python "$PROJECT_DIR/download_hpc.py" \
    --years $YEARS \
    --lustre-base-dir "$LUSTRE_BASE" \
    --max-workers "$MAX_WORKERS"
