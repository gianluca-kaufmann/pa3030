#!/bin/bash
#SBATCH --job-name=mini_sample_gate
#SBATCH --partition=normal.24h
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/%u/logs/mini_sample_gate_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/mini_sample_gate_%j.err
#SBATCH --chdir=/cluster/home/gikaufmann/master_thesis

# Issue MS — mini-sample validation gate.
# Trains on mini_sample.parquet (4M rows), evaluates on full SA test.parquet.
# Pass condition: Lift@1% > 2.81x (naive dist_wdpa baseline).
#
# Technique: symlink mini_sample.parquet as train.parquet under a temp directory.
# STAGE2_DATA_ROOT redirects resolve_parquet_file() to that directory for train.parquet;
# earlystop.parquet and test.parquet are absent there and fall through to the full
# SA splits at $SCRATCH/data/south_america/ml/main/ automatically.

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data/south_america/ml/mini_sample_run/main"
ln -sf "$SCRATCH/data/south_america/ml/mini_sample.parquet" \
       "$SCRATCH/data/south_america/ml/mini_sample_run/main/train.parquet"

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.4.1 eth_proxy

source "$HOME/venvs/master_thesis/bin/activate"
export PYTHONPATH="$HOME/master_thesis${PYTHONPATH:+:$PYTHONPATH}"

export STAGE2_DATA_ROOT="$SCRATCH/data/south_america/ml/mini_sample_run"
export STAGE2_NEG_RATIO=100
export STAGE2_ECO_GROUPS=1

echo "Mini-sample validation gate: train=mini_sample (4M rows), eval=full SA test..."
srun --cpu-bind=cores --distribution=block:block python -u \
  scripts/regions/south_america/5_training/model1_LGBM_stage2 \
  --variant mini_sample_gate
echo "Done. Check outputs/south_america/results/ml_models/ for *mini_sample_gate* metrics."
echo "Pass condition: Lift@1% > 2.81x (naive dist_wdpa baseline)."
