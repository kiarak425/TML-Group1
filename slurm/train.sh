#!/bin/bash
#SBATCH --job-name=p1-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00
#SBATCH --partition=l40s_normal_q     # ARC Falcon
#SBATCH --qos=fal_l40s_normal_short   # higher priority than the default QOS, max 1 day
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:2
#SBATCH --output=train-%j.log

# Full SFT of Qwen2.5-3B-Instruct on one of our 15K datasets, using configs/train.yaml.
# Run from the repo root after scripts/select_data.py has written data/<dataset>.json:
#   sbatch slurm/train.sh random_15k random_fal
#   sbatch slurm/train.sh concise_15k concise5_fal num_train_epochs=5
# Anything after the first two arguments is passed to LLaMA-Factory as key=value overrides.
#
# Notes:
#   - The global batch size stays at 8 no matter how many GPUs you request (we used 2 L40S).
#   - On TinkerCliffs use: sbatch -p h200_normal_q --qos=tc_h200_normal_short ... (the QOS has to match the partition).
#   - The model is saved to /scratch/$USER/p1/models/<name>, with one checkpoint per epoch.
#   - A full 3-epoch run on the longer datasets took ~27 h on 2 L40S, over the 24 h limit. For those we stopped
#     after the epoch-2 checkpoint and trained epoch 3 as a second job:
#       sbatch slurm/train.sh advanced_15k advanced_fal_e3 \
#         model_name_or_path=/scratch/$USER/p1/models/advanced_fal/checkpoint-3750 \
#         num_train_epochs=1 learning_rate=1.25e-6 seed=44
#     (1.25e-6 is where the 3-epoch cosine schedule is at the end of epoch 2.)
set -eo pipefail

DATASET=$1
NAME=$2
cd "$SLURM_SUBMIT_DIR"
[ -n "$NAME" ] || { echo "usage: sbatch slurm/train.sh <dataset> <name>"; exit 1; }
[ -f "data/$DATASET.json" ] || { echo "data/$DATASET.json missing: run scripts/select_data.py first"; exit 1; }

module load Miniconda3
module load CUDA/12.6.0
source activate p1train

# Print the node and GPUs we got. One Falcon node (fal033) had a broken GPU, which showed up here;
# you can skip a node with sbatch -x <node>.
hostname
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv

NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}   # GPUs given to this job, not all GPUs on the node
echo "GPUs: $NGPU, gradient accumulation: $((8 / NGPU)) (global batch 8)"
export WANDB_PROJECT=ece6514-p1
FORCE_TORCHRUN=1 llamafactory-cli train configs/train.yaml \
  dataset="$DATASET" run_name="$NAME" output_dir="/scratch/$USER/p1/models/$NAME" \
  gradient_accumulation_steps=$((8 / NGPU)) "${@:3}"
echo "TRAIN DONE: /scratch/$USER/p1/models/$NAME"
