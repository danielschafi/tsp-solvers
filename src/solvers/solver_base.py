import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tsplib95


class TSPSolver(ABC):
    """
    Base Class for TSP Solvers

    Ensures, that each solver can be used with a common interface, tracks metrics and provides convenience methods.
    """

    all_results: dict = {}

    def __init__(self, solver: str):
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
        self._results_dir: Path = Path("results")

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
        self.problem = tsplib95.load(self.tsp_file)
        self._tsp_problem_dict = self.problem.as_name_dict()

        self.result.update(
            {
                "timestamp": str(datetime.fromtimestamp(time.time())),
                "problem": self._tsp_problem_dict["name"],
                "problem_size": self._tsp_problem_dict["dimension"],
                "type": self._tsp_problem_dict["type"],
                "comment": self._tsp_problem_dict["comment"],
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

        print("\n")
        print("=" * 100)
        print(f"Solver: {self.solver}")
        print(f"Problem: {self.result['problem']}")
        print(f"Comment: {self.result['comment']}")
        print(f"Problem Size: {self.result['problem_size']}")
        print("=" * 100)

        print("Setting up problem")
        self.setup_problem(tsp_file)
        print("Start solving TSP")

        self.solve_tsp()
        print("Printing solution")

        self.print_solution()
        print("Making plot of solution")
        self.plot_solution()

        print("Done!")

    def print_tour(self, tour: List[int]):
        print("Route:")
        route_str = str(tour[0])
        for i, node in enumerate(tour[1:]):
            if (i + 1) % 10 == 0:
                route_str += "\n"
            route_str += " -> " + str(node)
        print(route_str)

    def print_solution(self):
        """Prints the solution found by the sover on the terminal"""
        print(json.dumps(self.result, indent=4))

    def plot_solution(self):
        """Plots the solution found by the solver and optionally saves it too"""
        if not self.result["tour"] or self.nodes is None:
            print("No tour or nodes available to plot.")
            return

        nodes_array = np.array(self.nodes)
        tour = self.result["tour"]
        tour_coords = nodes_array[tour]

        fig, ax = plt.subplots(2, figsize=(12, 5), sharex=True, sharey=True)
        # Plot raw nodes
        ax[0].set_title("Raw nodes")
        ax[0].scatter(nodes_array[:, 0], nodes_array[:, 1], color="blue", s=20)

        # PLot tour
        ax[1].set_title(f"{self.solver} Tour (Cost:{self.result['cost']:.2f})")
        ax[1].scatter(nodes_array[:, 0], nodes_array[:, 1], color="blue", s=20)

        # Draw path
        ax[1].plot(
            tour_coords[:, 0],
            tour_coords[:, 1],
            color="red",
            linestyle="-",
            linewidth=1,
            alpha=0.7,
        )
        plt.tight_layout()
        plt.savefig("test.png")
        plt.show()

    @classmethod
    def track_results():
        """
        Tracks the results of each solver on each problem, saves them in a suitable format to disk
        """
        raise NotImplementedError
