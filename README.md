# TSP Solvers

Solving the Traveling Salesperson Problem exactly using different solvers

Solvers:

- Gurobi
- Concorde
- CuOpt

## Setup

To install the necessary python packages, run:

```bash
uv sync
```

### Concorde

#### Create a folder for the solver

```bash
mkdir -p bin/ && cd bin/
```

#### Download the pre-compiled binary

```bash
wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/linux24/concorde.gz
```

#### Decompress and set permissions

```bash
gunzip concorde.gz
chmod +x concorde
```

#### Verify it works

```bash
./concorde -h
./concorde -s 99 -k 100
```

### Gurobi

**Installation**
We will use Gurobi through its Python API.
The free version of Gurobi only allows the solving of small problems.
You can obtain an academic license on their website. *Where?*

Place the license file in one of the following locations that Gurobi searches for by default:

- /opt/gurobi
- /home/\{YourUsername\}

### CuOpt

#### Installation

- Already installed through uv

## Data

The data for the TSP instances is in the `data/` folder. The files are in the TSPLib format, which is a standard format for representing TSP instances.
However, for visualization puroposes we save some extra fields compared to the Standard. Check `src/data_handling/tsplib_extension.py` for the implementation.

### Generating benchmark TSP data based on actual cities
To generate the TSP Data based on real city graphs we rely on OpenStreetMaps and osmnx to download the city Graph and randomly select nodes to generate problems. 
```bash
uv run src/data_handling/build_dataset.py
```
There are a some command line args you can view with `uv run src/data_handling/build_dataset.py -h` that make it easy to generate problems of various sizes for different cities, sizes etc.

The generated problems are saved to `data/tsp_dataset` by default. As TSPLib95 files. 
The format has been extended to include the NodeIDs of the nodes that were used in the problem.
They can then be used to do visualizations on the graph (the saved .graphml file)

## Solving a TSP with a solver

From the root of the project run:

```bash
uv run -m src.solvers.{gurobi_solver, cuopt_solver, concorde_solver, ...}
```

TODO: Provide some cmd line args: like gurobi_solver --tsp-file zurich_10.tsp --outdir --vizualize  etc. 

## Visualizations

Visualizations can either be done on street maps or plain.

```bash
    uv run -m src.visualization.viz_streetmap --path results/20260304_002601/concorde
```

# Neural Network Solver
## Data Preparation

1. Generate .tsp instances
```bash
 uv run -m src.data_handling.build_dataset --repetitions 3000 --sizes 100  --out_dir data/gnn_data
```
- Adjust sizes to the problem size you need
- For a bigger/smaller dataset adjust repetitions. The UTSP Paper used 3000

2. Load .tsp instances into numpy arrays/hdf5 file format for efficient model training
```bash
 uv run -m scripts.prepare_data_for_gnn_training --src_dir data/gnn_data
```
- Generates one h5 file per problem size. 

## Hyperparameter Tuning
```bash
uv run wandb sweep neural/config/sweep.yaml

# If slurm managed
sbatch sweep_job.sh

# Else (might be possible to adjust count)
uv run wandb agent --count 1 schafhdaniel-/tsp-solvers-neural_scripts/iwgpawd8 # replace with result from uv run wandb sweep neural/config/sweep.yaml
```

## Using the generated heatmap to produce optimal tours for problems of size n

We give the heatmap to LKH-3 as a starting point. We need to download  and compile its binary to use it. 
```bash
mkdir bin/lkh3
cd bin/lkh3

wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz
tar xvfz LKH-3.0.13.tgz
cd LKH-3.0.13
make

```
The model is trained on problems with a specific model dim. It needs to be retrained if you want to solve problems of another size. 

```bash
uv run -m src.solvers.neural_solver
```

## Running the Neural Network Solver in the benchmark