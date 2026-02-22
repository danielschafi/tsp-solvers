# TSP Gurobi

Solving the Traveling Salesperson Problem exactly using different solvers

Solvers:

- Gurobi
- Concorde
- CuOpt
- OR-Lib (google OR-Tools)

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
It is

**Get Gurobi license key**

*tbd*

### CuOpt

#### Installation

- Already installed through uv

## Data

The data for the TSP instances is available in the `data/` folder. The files are in the TSPLib format, which is a standard format for representing TSP instances.

### TSPLib files

```bash
cd data/
git clone https://github.com/mastqe/tsplib
```

### Generating benchmark TSP data based on actual cities

```bash
python build_dataset.py
```
