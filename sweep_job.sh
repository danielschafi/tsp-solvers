#!/bin/bash
#SBATCH --job-name=tsp-sweep
#SBATCH --array=1-50%2        # 50 trials, max 2 running at once (one per GPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=students
#SBATCH --time=05:30:00
#SBATCH --output=logs/sweep_%A_%a.out

# source .venv/bin/activate
uv run wandb agent --count 1 schafhdaniel-/tsp-solvers-neural_scripts/iwgpawd8 # replace with result from uv run wandb sweep neural/config/sweep.yaml
