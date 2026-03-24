#!/bin/bash
# ============================================================
# submit_process.slurm
# Array job: one task per month (1-12).
# After all tasks finish, a merge step runs automatically.
#
# Usage:
#   sbatch submit_process.slurm
# ============================================================
#SBATCH --job-name=ais_single
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --account=158484364623
#SBATCH --partition=cpu

module load GCCcore/13.2.0
module load Python/3.11.5
source $SCRATCH/MaritimeFreightCarbonEmission/.venv/bin/activate
python $SCRATCH/MaritimeFreightCarbonEmission/main_local.py \
  --years 2018 2019 2020 2021 2022\
  --raw-ais-dir   $GROUP_DIR/ais \
  --output-dir    $GROUP_DIR/ais/output_1 \
  --aux-data-dir  $SCRATCH/MaritimeFreightCarbonEmission \
  --n-workers 96