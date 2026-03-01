# TSP Gurobi

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

