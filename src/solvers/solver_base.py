import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import tsplib95
from dotenv import load_dotenv

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs

logger = logging.getLogger("src.solvers.solver_base")
load_dotenv()

np.random.seed(42)


class TSPSolver(ABC):
    """
    Base Class for TSP Solvers

    Ensures, that each solver can be used with a common interface, tracks metrics and provides convenience methods.
    """

    all_results: dict = {}

    def __init__(self, solver: str, results_dir=None, timeout: float | None = None):
        self.solver = solver
        self.timeout = timeout
        self.result: dict = {
            "timestamp": None,
            "problem": "undefined",
            "problem_size": 0,
            "type": "undefined",
            "comment": "",
            "solver": solver,
            "time_to_solve": 0,
            "cost": 0,
            "tour": [],
            "solution_status": None,
            "timed_out_without_tour": False,
            "additional_metadata": {},
            "valid_solution": None,
        }
        self._start_time = None
        self._end_time = None

        self.nodes: list | np.ndarray = []
        self.edges: np.ndarray | None = None

        if results_dir is not None:
            self.RESULTS_DIR = Path(results_dir)
        else:
            self.RESULTS_DIR = Path("results")

        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def setup(self, tsp_file):
        """
        Sets up the TSP Problem.
        Reads a .tsp file and prepares the data for the solver
        """
        pass

    @abstractmethod
    def solve_tsp(self):
        """Solves the TSP problem with the solver"""
        pass

    def load_tsp_file(self, tsp_file: str):
        """Loads the tsp file and saves its metadata"""
        if not Path(tsp_file).exists():
            raise FileNotFoundError(f"tsp_file: {tsp_file} does not exist.")

        self.tsp_file = Path(tsp_file)
        self.problem = tsplib95.load(self.tsp_file, problem_class=TSPProblemWithOSMIDs)
        self._tsp_problem_dict = self.problem.as_name_dict()

        self.result.update(
            {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "problem": self._tsp_problem_dict["name"],
                "problem_size": self._tsp_problem_dict["dimension"],
                "type": self._tsp_problem_dict["type"],
                "comment": self._tsp_problem_dict["comment"],
                "tsp_file": str(self.tsp_file.resolve()),
            }
        )

    def run(
        self,
        tsp_file: str,
        verbose: bool = True,
        plot: bool = True,
        show_plot: bool = False,
    ):
        """
        Runs the solver from start to finish
        1. setup_problem
        2. solve_tsp
        3. print_solution
        4. plot_solution
        5. save_results
        """

        self.result.update({
            "timestamp": None,
            "problem": "undefined",
            "problem_size": 0,
            "type": "undefined",
            "comment": "",
            "time_to_solve": 0,
            "cost": 0,
            "tour": [],
            "solution_status": None,
            "timed_out_without_tour": False,
            "additional_metadata": {},
            "valid_solution": None,
        })

        self.load_tsp_file(tsp_file)

        logger.info("=" * 100)
        logger.info(f"Solver: {self.solver}")
        logger.info(f"Problem: {self.result['problem']}")
        logger.info(f"Comment: {self.result['comment']}")
        logger.info(f"Problem Size: {self.result['problem_size']}")
        logger.info("=" * 100)

        logger.info("Setting up problem")
        self.setup(tsp_file)

        logger.info("Start solving TSP")
        self.solve_tsp()

        if self.result["tour"]:
            logger.info("Printing tour")
            self.print_tour(self.result["tour"])
        else:
            logger.warning("No tour computed — skipping tour print")

        logger.info("Checking validity of tour")
        self.check_solution_validity(self.result["tour"])

        logger.info("Printing results")
        self.print_results()

        logger.info("Saving results")
        self.save_results()

        if plot and self.result["tour"]:
            logger.info("Making plot of solution")
            self.plot_solution()

        logger.info("Done!")

    def check_solution_validity(self, tour: list[int] | None):
        """Checks if the tour is valid.
        - Each node is visited exactly once
        - Tour starts and ends at the same node
        - All nodes in the problem are visited

        tour format is: [1,5,2,3,1]
        """

        if tour is None:
            logger.warning("Tour is empty, is invalid.")
            self.result["valid_solution"] = False
            return

        if len(set(tour)) != len(tour) - 1:
            logger.warning("Some nodes visited more than once.")
            self.result["valid_solution"] = False
            return

        # -1 because we include the return to the start in the tour
        if self.result["problem_size"] != len(tour) - 1:
            logger.warning("Not all nodes in problem were visited.")
            self.result["valid_solution"] = False
            return

        if tour[0] != tour[-1]:
            logger.warning("Tour is not finished, tour[0] must equal tour[-1]")
            self.result["valid_solution"] = False
            return

        logger.info("Solution is valid")
        self.result["valid_solution"] = True

    def calculate_tour_cost(self, tour):
        """
        Calculate tour cost by summing up edge weights along the tour
        Assumes tour is a list of node indices: [0, 5, 2, 1]
        """
        assert self.edges is not None
        total_cost = 0
        n = len(tour)
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]  # Connects back to start
            # Look up symmetric distance
            # This assumes symmetric TSP, need to change this for non symmetric case
            total_cost += max(self.edges[i, j], self.edges[j, i])

        return float(total_cost)

    def print_tour(self, tour: list[int]):
        route_str = str(tour[0])
        for i, node in enumerate(tour[1:]):
            if (i + 1) % 10 == 0:
                route_str += "\n "
            route_str += " -> " + str(node)
        logger.info("Tour:\n" + route_str)

    def print_results(self):
        """Prints the solution found by the sover on the terminal"""
        logger.info("Results:\n" + json.dumps(self.result, indent=4))

    def save_results(self):
        """Saves the results json to a file"""
        problem_dir = (
            self.RESULTS_DIR / self.result["solver"] / f"n{self.result['problem_size']}"
        )
        problem_dir.mkdir(parents=True, exist_ok=True)
        with open(problem_dir / f"{self.result['problem']}.json", "w") as f:
            f.write(json.dumps(self.result, indent=4))

    def plot_solution(self):
        """Plots the solution plain and on streetmap"""
        problem_dir = (
            self.RESULTS_DIR / self.result["solver"] / f"n{self.result['problem_size']}"
        )
        problem_dir.mkdir(parents=True, exist_ok=True)
        from src.visualization.viz_plain import plot_solution_plain
        from src.visualization.viz_streetmap import plot_solution_streetmap

        plot_solution_plain(self.result, self.nodes, problem_dir)
        plot_solution_streetmap(self.result, str(self.tsp_file), problem_dir)
