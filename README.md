# TSP Solvers

Benchmark and compare Traveling Salesperson Problem solvers on real-world road-network instances. The project includes exact solvers (Gurobi, Concorde, CuOpt), a neural UTSP pipeline (GNN heatmap + guided MCTS), and an MCTS-only ablation to measure how much the heatmap contributes.

## Solvers

| Solver | Module | Description |
|--------|--------|-------------|
| **Gurobi** | `src.solvers.gurobi_solver` | Exact MIP solver with lazy subtour elimination |
| **Concorde** | `src.solvers.concorde_solver` | Exact TSP solver (external binary) |
| **CuOpt** | `src.solvers.cuopt_solver` | NVIDIA GPU routing solver |
| **UTSP** | `src.solvers.utsp_solver` | Scattering GNN heatmap + guided MCTS local search |
| **MCTS-only** | `src.solvers.utsp_mcts_only_solver` | MCTS without heatmap guidance (ablation baseline) |

## Setup

Requires Python 3.11 on Linux x86_64.

```bash
uv sync
```

### Concorde

```bash
mkdir -p bin/concorde && cd bin/concorde
wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/linux24/concorde.gz
gunzip concorde.gz
chmod +x concorde
./concorde -h
```

The solver expects the binary at `bin/concorde/concorde`.

### Gurobi

Gurobi is used through its Python API (`gurobipy`). The free tier only handles small instances; an [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) is needed for larger problems.

Place the license file (`gurobi.lic`) in one of Gurobi's default search paths, e.g. `/opt/gurobi` or your home directory.

### CuOpt

Installed via `uv sync`. Requires a CUDA-capable GPU for large instances.

### MCTS (UTSP local search)

The UTSP and MCTS-only solvers call a compiled C++ MCTS binary. It must live at `bin/MCTS-UTSP/test`:

```bash
# Copy or clone the UTSP Search/ code into bin/MCTS-UTSP, then:
cd bin/MCTS-UTSP
make
```

Ensure `Rec_Num` in `code/include/TSP_IO.h` matches `REC_NUM = 20` in `neural/scripts/inference.py`. Trained models are loaded from `saved_models/{size}/best.pt` (one model per problem size).

## Data

TSP instances live under `data/`. Files use an extended TSPLib format that adds OpenStreetMap node IDs for street-map visualization — see `src/data_handling/tsplib_extension.py`.

### Generating benchmark instances from real cities

Instances are sampled from city road networks (OpenStreetMap via osmnx):

```bash
uv run -m src.data_handling.build_dataset
uv run -m src.data_handling.build_dataset -h   # all options
```

Common options: `--city`, `--sizes`, `--repetitions`, `--out_dir` (default `data/tsp_dataset`), `--clean_build`.

Each size gets its own subdirectory (`data/tsp_dataset/25/`, etc.). A `.graphml` road network is saved alongside for visualization.

## Running a single solver

All solvers accept a `--path` to a `.tsp` file or a directory (all `.tsp` files inside are solved recursively):

```bash
uv run -m src.solvers.gurobi_solver     --path data/tsp_dataset/25/zurich_25_0.tsp
uv run -m src.solvers.concorde_solver   --path data/tsp_dataset/25/
uv run -m src.solvers.cuopt_solver      --path data/tsp_dataset/25/zurich_25_0.tsp
uv run -m src.solvers.utsp_solver         --path data/tsp_dataset/25/zurich_25_0.tsp
uv run -m src.solvers.utsp_mcts_only_solver --path data/tsp_dataset/25/zurich_25_0.tsp
```

Results are written to `results/{solver}/n{size}/{problem}.json`.

## Benchmark

Run multiple solvers across problem sizes:

```bash
uv run -m src.benchmark.run_benchmark \
  --solvers gurobi concorde utsp mcts_only \
  --sizes 25 50 100 200 \
  --results_dir results
```

Useful flags:

- `--force` — re-run even if a result JSON already exists (default: skip existing)
- `--clean_build` — delete the results directory before starting
- `--plot` — generate per-instance plots inline (off by default; prefer the viz scripts below)
- `--timeouts gurobi=60 mcts_only=1200` — per-solver timeout in seconds

Problem sizes are processed smallest-first. An aggregated `summary.csv` is written to the results directory when the run finishes.

## Visualizations

**Per-instance plots** from result JSONs:

```bash
uv run -m src.visualization.viz_streetmap --path results/UTSPSolver/n25
uv run -m src.visualization.viz_plain      --path results/concorde/n25
```

Both accept a single JSON or a directory (searched recursively). Existing PNGs are skipped. Use `--workers N` for parallel plotting.

**Aggregate benchmark plots and statistics:**

```bash
uv run -m src.visualization.viz_benchmark_results --results_dir results --out_dir results/plots
uv run -m src.visualization.analysis --results_dir results --out_dir results/stats
```

The analysis script prints coverage, cost, time, and optimality-gap tables. With `--solvers MCTSOnly UTSPSolver` it also reports a head-to-head comparison.

## Neural network solver (UTSP)

Implementation details and differences from the UTSP reference paper are documented in [`neural/COMPARISON.md`](neural/COMPARISON.md).

### 1. Generate training instances

```bash
uv run -m src.data_handling.build_dataset \
  --repetitions 3000 \
  --sizes 100 \
  --out_dir data/gnn_data
```

Adjust `--sizes` and `--repetitions` as needed. The UTSP paper used 3000 repetitions per size.

### 2. Convert to HDF5 for training

```bash
uv run python scripts/prepare_data_for_gnn_training.py --src_dir data/gnn_data
```

Creates one `processed.h5` per problem-size directory.

### 3. Hyperparameter tuning

Per-size sweep configs live in `neural/config/sweep_{size}.yaml`:

```bash
uv run wandb sweep neural/config/sweep_25.yaml

# Local agent (replace with sweep output):
uv run wandb agent --count 1 <entity>/<project>/<sweep_id>

# On Slurm:
sbatch slurm_neural_hyperparameter_tuning.sh
```

### 4. Store best configs

After a sweep, save winning hyperparameters to `neural/config/best/{size}.yaml`:

```yaml
# neural/config/best/25.yaml
model:
  hidden_dim: 64
  n_layers: 3
  node_features: "coords"

training:
  lr: 0.003
  weight_decay: 0.0
  step_size: 20
  gamma: 0.8
  lambda_1: 10.0
  lambda_2: 0.1
  temperature: 3.5
  batch_size: 32
  epochs: 300

data:
  path: "data/gnn_data/25/processed.h5"
```

The `model` section is used for both training and inference. `training` and `data` are training-only.

### 5. Train

```bash
uv run -m neural.scripts.train --config 25              # load best config for size 25
uv run -m neural.scripts.train --config 25 --lr 0.001   # override specific params
```

Checkpoints go to `checkpoints/`; the best model is saved to `saved_models/{size}/best.pt`.

### 6. Inference

```bash
# Low-level pipeline (GNN + MCTS):
uv run -m neural.scripts.inference --tsp_file data/tsp_dataset/25/zurich_25_0.tsp

# Through the solver interface (loads model from saved_models/{size}/best.pt):
uv run -m src.solvers.utsp_solver --path data/tsp_dataset/25/zurich_25_0.tsp
```

Models are size-specific — train (or load) a separate checkpoint for each problem size you want to solve.
