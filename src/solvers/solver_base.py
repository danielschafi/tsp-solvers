import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tsplib95
from dotenv import load_dotenv

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs
from src.visualization.viz_plain import plot_solution_plain
from src.visualization.viz_streetmap import plot_solution_streetmap

logger = logging.getLogger(__name__)

load_dotenv()

np.random.seed(42)


class TSPSolver(ABC):
    """
    Base Class for TSP Solvers

    Ensures, that each solver can be used with a common interface, tracks metrics and provides convenience methods.
    """

    all_results: dict = {}

    def __init__(self, solver: str, results_dir=None):
        self.solver = solver
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
            "additional_metadata": {},
        }
        self._start_time = None
        self._end_time = None

        self.nodes: List = []
        self.edges: np.ndarray = None

        if results_dir is not None:
            self.RESULTS_DIR = Path(results_dir)
        else:
            self.RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))

        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def setup_problem(self, tsp_file):
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

        self.load_tsp_file(tsp_file)

        logger.info("=" * 100)
        logger.info(f"Solver: {self.solver}")
        logger.info(f"Problem: {self.result['problem']}")
        logger.info(f"Comment: {self.result['comment']}")
        logger.info(f"Problem Size: {self.result['problem_size']}")
        logger.info("=" * 100)

        logger.info("Setting up problem")
        self.setup_problem(tsp_file)

        logger.info("Start solving TSP")
        self.solve_tsp()

        logger.info("Printing tour")
        self.print_tour(self.result["tour"])

        logger.info("Printing results")
        self.print_results()

        logger.info("Saving results")
        self.save_results()

        if plot:
            logger.info("Making plot of solution")
            self.plot_solution()

        logger.info("Done!")

    def calculate_tour_cost(self, tour):
        """
        Calculate tour cost by summing up edge weights along the tour
        Assumes tour is a list of node indices: [0, 5, 2, 1]
        """
        total_cost = 0
        n = len(tour)
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]  # Connects back to start
            # Look up symmetric distance
            # TODO: This assumes symmetric TSP, need to change this for non symmetric case
            total_cost += max(self.edges[i, j], self.edges[j, i])

        return float(total_cost)

    def print_tour(self, tour: List[int]):
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
        plot_solution_plain(self.result, self.nodes, problem_dir)
        plot_solution_streetmap(self.result, self.tsp_file, problem_dir)
