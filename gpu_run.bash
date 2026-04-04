#!/bin/bash
#SBATCH --partition=students
#SBATCH --gpus=a100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=train_%j.log

# uv run -m neural.scripts.train
uv run -m neural.scripts.train --batch_size=64 --gamma=0.9163692294837598 --hidden_dim=128 --lambda_1=5.27321703849012 --lambda_2=0.001879291559180024 --lr=0.0008723454838963124 --n_layers=4 --node_features=coords --step_size=20 --temperature=3.2322223351117327 --weight_decay=0
tail -f train_%j.log