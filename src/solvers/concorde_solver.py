import argparse
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.logger import setup_logging
from src.solvers.solver_base import TSPSolver

logger = logging.getLogger("src.solvers.concorde_solver")

load_dotenv()

np.random.seed(42)


class ConcordeSolver(TSPSolver):
    """
    Solver for TSP using concorde
    """

    def __init__(self, results_dir=None, timeout: float | None = None):
        super().__init__(solver="concorde", results_dir=results_dir, timeout=timeout)
        self.CONCORDE_BIN = Path(os.getcwd()) / "bin/concorde/concorde"
        if not self.CONCORDE_BIN.exists():
            raise ValueError(
                f"The Concorde binary is not found at {self.CONCORDE_BIN}, download it according to the instructions in the readme.md to use the concorde solver"
            )

    def setup_problem(self, tsp_file: str):
        """
        Prepares the data for the solver.
        Builds the adjacendy matrix,
        """
        self.load_tsp_file(tsp_file)
        self.edges = np.array(self.problem.edge_weights)
        self.nodes = np.array(self.problem.node_locations)

    def extract_tour_from_sol_file(self, tempdir: str | Path) -> list[int]:
        # Get the info we need.
        tsp_solution = Path(tempdir) / f"{Path(self.tsp_file).stem}.sol"
        text_sol = tsp_solution.read_text()

        text_sol = text_sol.replace("\n", " ")
        split_text = text_sol.split(" ")
        split_text_cleaned = [int(node) for node in split_text if len(node) > 0]
        # remove node count.
        split_text_cleaned.pop(0)

        # Finish cycle (add start node to end)
        if split_text_cleaned[0] != split_text_cleaned[-1]:
            split_text_cleaned.append(split_text_cleaned[0])

        return split_text_cleaned

    def parse_concorde_output(self, process_result: subprocess.CompletedProcess[str]):
        """Extract the info we need from concordes output string"""
        stdout = process_result.stdout

        # Define patterns for the data we want
        patterns = {
            "lower_bound": r"Final lower bound\s+([\d.]+)",
            "upper_bound": r"upper bound\s+([\d.]+)",
            "optimal_val": r"Optimal Solution:\s+([\d.]+)",
            "gap": r"DIFF:\s+([\d.]+)",
            "total_time": r"Total Running Time:\s+([\d.]+)",
            "seed": r"Using random seed\s+(\d+)",
        }

        results = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, stdout)
            if match:
                # Convert to float/int if possible, otherwise keep as string
                val = match.group(1)
                try:
                    results[key] = float(val) if "." in val else int(val)
                except ValueError:
                    results[key] = val

        # Check for success/failure in returncode or stderr
        results["success"] = process_result.returncode == 0

        return results

    def solve_tsp(self, seed: int = 42):
        """
        Solves the TSP using concorde
        """

        # Tempdir to capture output files
        with tempfile.TemporaryDirectory() as tempdir:
            self._start_time = time.perf_counter()

            try:
                result = subprocess.run(
                    [self.CONCORDE_BIN, "-s", str(seed), str(self.tsp_file.resolve())],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=tempdir,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                self._end_time = time.perf_counter()
                self.result["time_to_solve"] = self._end_time - self._start_time
                self.result["solution_status"] = "TIMEOUT"
                self.result["timed_out_without_tour"] = True
                logger.warning(
                    f"SOLVER TIMED OUT — concorde exceeded {self.timeout}s without finding a tour"
                )
                return

            self._end_time = time.perf_counter()
            self.result["time_to_solve"] = self._end_time - self._start_time

            tour = self.extract_tour_from_sol_file(tempdir)
            self.result["tour"] = tour

            logger.info(result)
            structured_output = self.parse_concorde_output(result)

            self.result["cost"] = self.calculate_tour_cost(tour)
            self.result["additional_metadata"]["lower_bound"] = structured_output[
                "lower_bound"
            ]
            self.result["additional_metadata"]["upper_bound"] = structured_output[
                "upper_bound"
            ]
            self.result["additional_metadata"]["gap"] = structured_output["gap"]
            self.result["additional_metadata"]["seed"] = structured_output["seed"]

            if result.returncode != 0:
                status = "Process call to concorde returned with an error code"
                logger.warning(
                    "SOLVER NOT SUCCESSFUL — concorde returned a non-zero exit code"
                )
            elif len(self.result["tour"]) == 0:
                status = "Failed to find tour"
                logger.warning("SOLVER NOT SUCCESSFUL — failed to find tour")
            elif structured_output["gap"] == 0:
                status = "Success: Exact solution found. gap is zero"
            else:
                status = "Success: Solution found with non-zero gap"

            self.result["solution_status"] = status


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the concorde solver on a .tsp file or all .tsp files in a folder."
    )
    arg_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the .tsp file to solve.",
    )

    args = arg_parser.parse_args()
    setup_logging()
    path = Path(args.path)

    # check if it is a file or a folder    if path.is_file():
    if path.is_file():
        solver = ConcordeSolver()
        solver.run(str(path))
    elif path.is_dir():
        files = list(path.rglob("*.tsp"))
        files = sorted(
            files,
        )
        for i, tsp_file in enumerate(files):
            logger.info(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver = ConcordeSolver()
            solver.run(str(tsp_file))


if __name__ == "__main__":
    main()
