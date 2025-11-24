#!/bin/bash
#SBATCH --job-name=ViT_Sokoban
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress,llaima,hydra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.palma@uc.cl
#SBATCH --chdir=/home/pedropalmav/archive/masters-experiments/vit_bc
#SBATCH --export=ALL
#SBATCH --array=0-3

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

LAYERS=(4 5)
NUM_LAYERS=${LAYERS[$SLURM_ARRAY_TASK_ID / 2]}
LEARNING_RATES=(0.002 0.001)
NUM_LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID % 2]}

# Diagnostic information
echo "----------------------------------------"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Checking GPU state before starting Python:"
nvidia-smi
echo "----------------------------------------"

# Run the training script
pyenv activate env
echo "Running training for $NUM_LAYERS layers..."
python train_vit.py --epochs 200 --num_layers $NUM_LAYERS  --batch_size 4096 --lr $NUM_LR
pyenv deactivate
