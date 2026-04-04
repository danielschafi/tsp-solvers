#!/bin/bash

#SBATCH --output=logs/nn_inference_%A_%a.out
#SBATCH --time=3:00:00           ## Zeitlimite. Diese sollte gleich oder kleiner der Partitions Zeitlimite sein. In diesem Fall ist diese auf 1 Stunde und 30 Minuten gesetzt.
#SBATCH --job-name="TSP-GNN-Solver"   ## Job Name.
#SBATCH --partition=students	 ## Partitionsname. Die zur Verfügung stehenden Partitionen können mit dem Befehl sinfo angezeigt werden
#SBATCH --mem=100G               ## Der Arbeitsspeicher, welcher für den Job reserviert wird
#SBATCH --cpus-per-task=16     ## Die Anzahl virtueller Cores, die für den Job reserviert werden
#SBATCH --gpus=a100:1		 ## Die Anzahl GPUs (in diesem Beispiel zwei GPUs, mit der Syntax :2)

uv run -m neural.scripts.inference --tsp_file data/tsp_dataset/25/zurich_25_0.tsp
# uv run -m src.benchmark.run_benchmark --solvers neural --sizes 25 --timeouts neural=1200