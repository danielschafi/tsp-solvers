"""
Runs a solver/list of solvers on a number of TSP Problems and computes metrics.
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.solvers.concorde_solver import ConcordeSolver
from src.solvers.gurobi_solver import GurobiSolver
from src.solvers.solver_base import TSPSolver

load_dotenv()

BENCHMARK_DATA_DIR = os.getenv("BENCHMARK_DATA_DIR", "data/tsp_dataset")


def get_new_solver(solver_name: str, results_dir: str = None) -> TSPSolver:
    if solver_name.lower() == "gurobi":
        return GurobiSolver(results_dir=results_dir)
    elif solver_name.lower() == "concorde":
        return ConcordeSolver(results_dir=results_dir)
    elif solver_name.lower() == "cuopt":
        from src.solvers.cuopt_solver import CuOptSolver

        return CuOptSolver(results_dir=results_dir)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def run_benchmark(solvers: List[str], data_dirs: List[Path], results_dir: Path) -> None:
    """
    Runs the specified solvers on all the instances of the specified problem sizes and saves the results

    Args:
        solvers (List[str]): List of solver names to run in the benchmark.
        data_dirs (List[Path]): List of directories containing the .tsp files to run the benchmark on. Each directory should correspond to a problem size.
        results_dir (Path): Base directory to save the results.
    """

    benchmark_ts = str(datetime.now().strftime("%Y%m%d_%H%M%S"))

    for data_dir in data_dirs:
        print(f"Running benchmark on {data_dir}, at time {benchmark_ts}...")
        files = list(data_dir.rglob("*.tsp"))
        files = sorted(files)
        for i, tsp_file in enumerate(files):
            print(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            for solver_name in solvers:
                results_dir_for_run = Path(results_dir) / benchmark_ts

                solver_instance = get_new_solver(solver_name, str(results_dir_for_run))
                solver_instance.run(str(tsp_file))


def create_aggregated_results(results_dir: Path) -> None:
    """
    Aggregate the results from the different solvers and problem sizes into a single file for easier analysis and plotting.
    """
    # TODO: implement this function to read the individual result files and create an aggregated results file (e.g. a csv or json file) that contains the relevant metrics for each solver and problem size.
    pass


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run a benchmark of TSP solvers on a set of .tsp files."
    )
    arg_parser.add_argument(
        "--solvers",
        nargs="+",
        help="List of solver names to run in the benchmark. one or more of 'gurobi', 'concorde', 'cuopt'",
        required=True,
        choices=["gurobi", "concorde", "cuopt"],
    )
    arg_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
        ],
        help="List of problem sizes to run the benchmark on.",
    )
    arg_parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save the results.",
    )
    arg_parser.add_argument(
        "--clean_build",
        action="store_true",
        help="Whether to clean the old results directory before running the benchmark.",
    )

    args = arg_parser.parse_args()

    print(f"""Running benchmark on problem sizes: {args.sizes}. 
        Saving results to {args.results_dir}.
        Clean build: {args.clean_build}""")

    sizes = args.sizes
    data_dirs: List[Path] = []
    for size in sizes:
        data_dir = Path(BENCHMARK_DATA_DIR) / f"{size}"
        if not data_dir.exists():
            raise ValueError(
                f"Data directory {data_dir} does not exist. Please make sure to build the dataset before running the benchmark."
            )
        data_dirs.append(data_dir)

    results_dir = Path(args.results_dir)
    if args.clean_build and results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    if not all(solver in ["gurobi", "concorde", "cuopt"] for solver in args.solvers):
        raise ValueError(
            "Invalid solver name. Must be one of 'gurobi', 'concorde', 'cuopt'."
        )

    run_benchmark(args.solvers, data_dirs, results_dir)


if __name__ == "__main__":
    main()
