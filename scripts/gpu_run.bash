srun --gpus=a100:1 --time=06:00:00 -p students uv run -m src.solvers.cuopt_solver "$@"
