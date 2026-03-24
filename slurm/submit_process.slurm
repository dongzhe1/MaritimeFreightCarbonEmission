#!/bin/bash
# ============================================================
# submit_process.slurm
# Array job: one task per month (1-12).
# After all tasks finish, a merge step runs automatically.
#
# Usage:
#   sbatch submit_process.slurm
# ============================================================

#SBATCH --job-name=ais_process
#SBATCH --output=logs/process_%A_%a.out
#SBATCH --error=logs/process_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --array=1-12
#SBATCH --partition=cpu

# ---- user settings ----------------------------------------
YEAR=2024
LUSTRE_AIS="$SCRATCH/ais"
LUSTRE_OUTPUT="$SCRATCH/ais/output"
AUX_DATA_DIR="$SCRATCH/MaritimeFreightCarbonEmission"
PROJECT_DIR="$SCRATCH/MaritimeFreightCarbonEmission"
N_WORKERS=96
# -----------------------------------------------------------

mkdir -p logs "$LUSTRE_OUTPUT/$YEAR"

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source "$PROJECT_DIR/.venv/bin/activate"

echo "Processing year=$YEAR month=$SLURM_ARRAY_TASK_ID on $(hostname)"

python "$PROJECT_DIR/main_hpc.py" \
    --year "$YEAR" \
    --lustre-ais-dir    "$LUSTRE_AIS" \
    --lustre-output-dir "$LUSTRE_OUTPUT" \
    --aux-data-dir      "$AUX_DATA_DIR" \
    --n-workers         "$N_WORKERS"