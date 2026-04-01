#!/bin/bash
#SBATCH --partition=students
#SBATCH --gpus=a100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=train_%j.log

uv run -m neural.scripts.train
