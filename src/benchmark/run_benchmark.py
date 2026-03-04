"""
Runs a solver/list of solvers on a number of TSP Problems and computes metrics.
"""

import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.logger import setup_logging
from src.solvers.concorde_solver import ConcordeSolver
from src.solvers.gurobi_solver import GurobiSolver
from src.solvers.solver_base import TSPSolver

logger = logging.getLogger(__name__)

load_dotenv()

BENCHMARK_DATA_DIR = os.getenv("BENCHMARK_DATA_DIR", "data/tsp_dataset")


def get_new_solver(
    solver_name: str, results_dir: str | None = None, timeout: float | None = None
) -> TSPSolver:
    if solver_name.lower() == "gurobi":
        return GurobiSolver(results_dir=results_dir, timeout=timeout)
    elif solver_name.lower() == "concorde":
        return ConcordeSolver(results_dir=results_dir, timeout=timeout)
    elif solver_name.lower() == "cuopt":
        # Import only here to avoid having to run this as a gpu job every time.
        from src.solvers.cuopt_solver import CuOptSolver

        return CuOptSolver(results_dir=results_dir, timeout=timeout)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def run_benchmark(
    solvers: list[str],
    data_dirs: list[Path],
    results_dir: Path,
    benchmark_ts: str | None = None,
    plot: bool = False,
    timeouts: dict | None = None,
) -> None:
    """
    Runs the specified solvers on all the instances of the specified problem sizes and saves the results.

    Problem sizes are processed in ascending order. If a solver times out without producing a tour,
    it is dropped from all subsequent (larger) problem sizes.

    Args:
        solvers (List[str]): List of solver names to run in the benchmark.
        data_dirs (List[Path]): List of directories containing the .tsp files to run the benchmark on. Each directory should correspond to a problem size.
        results_dir (Path): Base directory to save the results.
        benchmark_ts (str): Timestamp string for this run (used in results directory naming).
        plot (bool): Whether to generate plots during the run. Disabled by default; use the viz scripts afterwards.
        timeouts (dict): Optional per-solver timeout in seconds, e.g. {"gurobi": 60, "concorde": 300}.
    """

    if benchmark_ts is None:
        benchmark_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if timeouts is None:
        timeouts = {}

    # Sort by problem size (directory name is the size) so we process smallest first
    data_dirs = sorted(data_dirs, key=lambda d: int(d.name))

    active_solvers = list(solvers)

    for data_dir in data_dirs:
        if not active_solvers:
            logger.info("No active solvers remaining — stopping benchmark early.")
            break

        logger.info(
            f"Running benchmark on {data_dir} (size {data_dir.name}), at time {benchmark_ts}. "
            f"Active solvers: {active_solvers}"
        )
        files = sorted(data_dir.rglob("*.tsp"))
        results_dir_for_run = Path(results_dir) / benchmark_ts

        solvers_to_drop = set()

        for i, tsp_file in enumerate(files):
            logger.info(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            for solver_name in list(active_solvers):
                if solver_name in solvers_to_drop:
                    continue

                timeout = timeouts.get(solver_name.lower())
                solver_instance = get_new_solver(
                    solver_name, str(results_dir_for_run), timeout=timeout
                )
                solver_instance.run(str(tsp_file), plot=plot)

                if solver_instance.result.get("timed_out_without_tour"):
                    logger.warning(
                        f"Solver '{solver_name}' timed out on {tsp_file} without a tour. "
                        f"Dropping it from all remaining problem sizes."
                    )
                    solvers_to_drop.add(solver_name)

        for solver_name in solvers_to_drop:
            active_solvers.remove(solver_name)
            logger.info(
                f"Dropped solver '{solver_name}'. Remaining active solvers: {active_solvers}"
            )


def create_aggregated_results(results_dir: Path) -> None:
    """
    Aggregate the results from the different solvers and problem sizes into a single file for easier analysis and plotting.
    """
    # TODO: implement this function to read the individual result files and create an aggregated results file (e.g. a csv or json file) that contains the relevant metrics for each solver and problem size.
    pass


def main():
    # E.g. uv run -m src.benchmark.run_benchmark -h --solvers gurobi concorde cuopt --sizes 10 --results_dir results
    # srun --gpus=a100:1 --time=00:30:00 -p students --pty uv run -m src.benchmark.run_benchmark --solvers gurobi concorde cuopt --sizes 10 25 50 100 200 500 1000 2000 5000 10000 --results_dir results --clean_build

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
    arg_parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate plots inline during the benchmark run. Off by default — use viz_plain.py / viz_streetmap.py afterwards for faster batch plotting.",
    )
    arg_parser.add_argument(
        "--timeouts",
        nargs="*",
        default=[],
        metavar="SOLVER=SECONDS",
        help=(
            "Per-solver timeout in seconds. If a solver times out without a tour, it is dropped "
            "from all larger problem sizes. Example: --timeouts gurobi=60 concorde=300 cuopt=120"
        ),
    )

    args = arg_parser.parse_args()

    benchmark_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(log_dir=Path("logs"), run_ts=benchmark_ts)

    logger.info(
        f"Running benchmark on problem sizes: {args.sizes}. "
        f"Saving results to {args.results_dir}. "
        f"Clean build: {args.clean_build}"
    )

    sizes = args.sizes
    data_dirs: list[Path] = []
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

    timeouts = {}
    for entry in args.timeouts:
        try:
            solver_name, seconds = entry.split("=")
            timeouts[solver_name.lower()] = float(seconds)
        except ValueError as err:
            raise ValueError(
                f"Invalid --timeouts entry '{entry}'. Expected format: solver=seconds (e.g. gurobi=60)"
            ) from err

    if timeouts:
        logger.info(f"Solver timeouts: {timeouts}")

    run_benchmark(
        args.solvers,
        data_dirs,
        results_dir,
        benchmark_ts,
        plot=args.plot,
        timeouts=timeouts,
    )


if __name__ == "__main__":
    main()
