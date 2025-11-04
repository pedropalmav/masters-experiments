#!/bin/bash
#SBATCH --job-name=Success_rate_Sokoban
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress,llaima,hydra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro.palma@uc.cl
#SBATCH --chdir=/home/pedropalmav/archive/masters-experiments/vit_bc
#SBATCH --export=ALL

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# 100.000 levels for train, 100 levels for test
# Run the training script
pyenv activate env
python test_vit.py --num_levels 100 --env_split test
pyenv deactivate
