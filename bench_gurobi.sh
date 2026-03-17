#!/bin/bash

#SBATCH --output="slurm-%j.out"  ## Im Verzeichnis aus dem sbatch aufgerufen wird, wird ein Logfile mit dem Namen slurm-[Jobid].out erstellt.
#SBATCH --error="slurm-%j.err"   ## Ähnlich wie --output. Jedoch ein Log für Fehlermeldungen.
#SBATCH --time=6:00:00           ## Zeitlimite. Diese sollte gleich oder kleiner der Partitions Zeitlimite sein. In diesem Fall ist diese auf 1 Stunde und 30 Minuten gesetzt.
#SBATCH --job-name="TSP-Gurobi"   ## Job Name.
#SBATCH --partition=students	 ## Partitionsname. Die zur Verfügung stehenden Partitionen können mit dem Befehl sinfo angezeigt werden
#SBATCH --mem=200G               ## Der Arbeitsspeicher, welcher für den Job reserviert wird
#SBATCH --cpus-per-task=32     ## Die Anzahl virtueller Cores, die für den Job reserviert werden

# uv run -m src.solvers.concorde_solver --path data/tsp_dataset/10/zurich_10_0.tsp
uv run -m src.benchmark.run_benchmark --solvers gurobi --timeouts gurobi=1200