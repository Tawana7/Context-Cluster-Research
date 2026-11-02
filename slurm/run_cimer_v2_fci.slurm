#!/bin/bash
#SBATCH --job-name=cimer_fci_v2
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=logs/cimer_%x_%j.out
#SBATCH --error=logs/cimer_%x_%j.err

# v2 protocol: validation subjects + early stopping + checkpoints.
# If the job hits the time limit, just resubmit this same script: --resume
# skips finished folds and continues the interrupted one from last.pt.

module load anaconda/anaconda-2025-12-2
export WANDB_MODE=offline

apptainer exec --nv "$ANACONDA_SIF" bash -c "
  source /opt/anaconda/etc/profile.d/conda.sh
  conda activate cimer
  cd \$HOME/Context-Cluster
  python -u train_cimer_casme2_ver2.py \
    --data_root \$HOME/Context-Cluster/Flow \
    --xlsx \$HOME/CASME_II/CASME2-coding-20140508.xlsx \
    --pretrained \$HOME/Context-Cluster/checkpoints/model_best.pth.tar \
    --graph_mode fci \
    --fci_test chisq --save_graphs_dir fci_graphs_v2 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --val_subjects 3 \
    --patience 10 \
    --min_epochs 10 \
    --ckpt_dir \$HOME/Context-Cluster/ckpts \
    --resume \
    --results_csv results_v2_fci.csv \
    --use_wandb --wandb_project Research_Project --wandb_run_name fci_v2
"
