import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.solvers.solver_base import TSPSolver

load_dotenv()


class ConcordeSolver(TSPSolver):
    """
    Solver for TSP using concorde
    """

    def __init__(self):
        super().__init__(solver="concorde")
        self.CONCORDE_BIN = os.getenv("CONCORDE_BIN", None)
        if not self.CONCORDE_BIN:
            raise ValueError(
                "CONCORDE_BIN needs to be specified in the .env file to use the ConcordeSolver "
            )

    def setup_problem(self, tsp_file: str):
        """
        Prepares the data for the solver.
        Builds the adjacendy matrix,
        """
        self.load_tsp_file(tsp_file)
        self.edges = np.array(self.problem.edge_weights)
        self.nodes = np.array(self.problem.node_locations)

    def extract_tour_from_sol_file(self, tempdir: Path) -> None:
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

    def parse_concorde_output(self, process_result: str):
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
            # Solve the routing problem
            self._start_time = time.perf_counter()

            result = subprocess.run(
                [self.CONCORDE_BIN, "-s", str(seed), str(self.tsp_file)],
                capture_output=True,
                text=True,
                check=True,
                cwd=tempdir,
            )

            self._end_time = time.perf_counter()
            self.result["time_to_solve"] = self._end_time - self._start_time

            tour = self.extract_tour_from_sol_file(tempdir)
            self.result["tour"] = tour

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

            if structured_output["gap"] == 0:
                status = "Success: Exact solution found. gap is zero"
            elif result.returncode != 0:
                status = "Process call to concorde returned with an error code"
                print("!!! WARNING !!!")
                print("SOLVER NOT SUCCESSFUL")
            elif len(self.result["tour"]) == 0:
                status = "Failed to find tour"
                print("!!! WARNING !!!")
                print("SOLVER NOT SUCCESSFUL")

            self.result["solution_status"] = status


def main():
    solver = ConcordeSolver()
    solver.run(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/10_conv/zurich_10_0.tsp"
    )


if __name__ == "__main__":
    main()
