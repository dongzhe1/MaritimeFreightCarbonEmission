#!/bin/bash
# ============================================================
# submit_merge.slurm
# Merges monthly parquet files into annual Ship_Routes.parquet
# and Carbon_Transfer.parquet.
#
# Run after all monthly process jobs finish:
#   PROCESS_JOB=$(sbatch --parsable submit_process.slurm)
#   sbatch --dependency=afterok:$PROCESS_JOB submit_merge.slurm
#
# Or manually after verifying all months are done:
#   sbatch submit_merge.slurm
# ============================================================
#SBATCH --job-name=ais_merge
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# ---- user settings ----------------------------------------
YEAR=2024
LUSTRE_OUTPUT="/scratch/$USER/ais/output"
PROJECT_DIR="$SCRATCH/MaritimeFreightCarbonEmission"
# -----------------------------------------------------------

mkdir -p logs

module purge
module load Python/3.11

source "$PROJECT_DIR/.venv/bin/activate"

echo "Merging year=$YEAR on $(hostname)"

python "$PROJECT_DIR/main_hpc.py" \
    --year "$YEAR" \
    --lustre-ais-dir   "unused" \
    --lustre-output-dir "$LUSTRE_OUTPUT" \
    --aux-data-dir     "unused" \
    --merge
