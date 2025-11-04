# ============================================================
#  ETH ZURICH EULER WORKFLOW — MASTER'S THESIS
#  Author: Gian-Luca Kaufmann
# ============================================================

# 1️⃣ Log in to Euler
ssh gikaufmann@login.euler.ethz.ch

# 2️⃣ Navigate to project
cd ~/master_thesis

# 3️⃣ Sync latest code
git pull --rebase

# 4️⃣ Load modules & activate virtual environment
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load eth_proxy
source ~/venv/master-thesis/bin/activate

# 5️⃣ Prepare data/outputs links (done once per setup)
rm -rf ~/master_thesis/data ~/master_thesis/outputs
ln -s $SCRATCH/data ~/master_thesis/data
ln -s $SCRATCH/outputs ~/master_thesis/outputs
ls -l ~/master_thesis | grep -E 'data|outputs'

# 6️⃣ (One-time) Login to Weights & Biases
wandb login <YOUR_API_KEY>
wandb settings set entity gikaufmann
wandb settings set project master_thesis

# 7️⃣ Submit SLURM job
sbatch merge_2012.slurm

# 8️⃣ Monitor queue and logs
squeue -u $USER
tail -f $SCRATCH/logs/merge_2012_*.out

# 9️⃣ Inspect finished jobs and outputs
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
ls -lh $SCRATCH/outputs/Results/

# 🔟 Copy results to local machine
scp gikaufmann@login.euler.ethz.ch:/cluster/scratch/gikaufmann/outputs/Results/merged_panel_2012_2024.parquet ~/Desktop/

# 1️⃣1️⃣ Quick debugging commands
nano merge_2012.slurm          # edit SLURM script
bash -n merge_2012.slurm       # check syntax
module load tmux               # safe monitoring
tmux new -s merge
tail -f $SCRATCH/logs/merge_2012_*.out
# (reconnect later with: tmux attach -t merge)

# ============================================================
#  WORKING CONFIGURATION (final stable run)
# ------------------------------------------------------------
#  CPUs: 8
#  Memory: 16 GB per CPU (128 GB total)
#  Partition: normal.24h
#  Data: /cluster/scratch/gikaufmann/data/ready/WDPA
#  Output: /cluster/scratch/gikaufmann/outputs/Results/
#  Output file: merged_panel_2012_2024.parquet (≈30 GB)
#  Runtime: ~108 minutes
#  W&B: connected to gikaufmann/master_thesis
# ============================================================