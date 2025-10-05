#!/bin/bash
#SBATCH --job-name=joint-train-array
#SBATCH --time=01:00:00                    # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G                  # Total ~28G memory
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=1-2                       # Run 5 jobs (adjust number as needed)

#SBATCH --output=/cluster/scratch/%u/logs/uav_joint_train_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/logs/uav_joint_train_%A_%a.err

# Create necessary folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/datasets"
mkdir -p "$SCRATCH/models"
mkdir -p "$SCRATCH/wandb/joint_model"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment (uav-genAI-opti)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venv/uav-genAI-opti"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Set WANDB environment variables
#------------------------------------------------------------------------------
# make sure to set wandb api key and entity in the cluster environment variables
export WANDB_PROJECT="gen_ai_opti_uav"
export WANDB_API_KEY="remember to put your api key here"

#-----------------------------------------------------------------------------
# Navigate to training script directory
#------------------------------------------------------------------------------
cd "$HOME/projects/uav-genAI-opti/UAV/8_train_joint_model" || { echo "ERROR: Could not cd into training directory"; exit 1; }

#------------------------------------------------------------------------------
# Run the training
#------------------------------------------------------------------------------
echo "Starting joint model training..."
srun python train_aircraft_joint_model.py \
    --wandb-dir "$SCRATCH/wandb/joint_model" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --video \
    --case-name "case_1_b" \
    --sampling-mode "latin_hypercube" \
    --weights-mode "hypercube_ALM" \
    --batchsize 1296 \
    --dpp-method "z_individual" --sigma-dpp 0.3 \
    --epochs 2500 \
    --lambda-dpp 0.0 \
    --lambda-wing-dpp 0.2 \
    --lambda-internals-dpp-start 30 --lambda-internals-dpp-end 0.4 \
    --lambda-fuselage-dpp 0.2 \
    --lambda-performance 0.25 \
    --lambda-mi-start 8 --lambda-mi-end 0.2 \
    --wandb-group "MI annealing 4 - no detached wing placement"