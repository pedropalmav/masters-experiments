#!/bin/bash
#SBATCH --job-name=ViT_Sokoban
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=ialab
#SBATCH --nodelist=llaima,hydra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.palma@uc.cl
#SBATCH --chdir=/home/pedropalmav/archive/masters-experiments/vit_bc
#SBATCH --export=ALL

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"


# Diagnostic information
echo "----------------------------------------"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Checking GPU state before starting Python:"
nvidia-smi
echo "----------------------------------------"

# Run the training script
pyenv activate env
python train_vit.py --epochs 300 --num_layers 7 --batch_size 4096 --lr 0.002 --filename $SLURM_JOB_ID --hidden_dim 128 --early_stopping
pyenv deactivate
