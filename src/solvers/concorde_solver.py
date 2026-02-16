import subprocess
from typing import List


class ConcordeSolver(TSPSolver):
    """
    Concorde TSP Solver
    """

    def __init__(self):
        super().__init__(solver="Concorde")
        self.result: dict = {"total_time": 0, "tour_length": 0, "tour": []}
        self._start_time = None
        self._end_time = None

    @abstractmethod
    def setup_problem(self, tsp_file: str):
        """
        Sets up the TSP Problem.
        Reads a .tsp file and prepares the data for the solver
        """

        # TODO:Check if it is a valid file here

        self.tsp_file_path = tsp_file

    @abstractmethod
    def solve_tsp(self):
        """
        Solves the TSP problem with the solver
        """
        subprocess.run("")

    def print_tour(self, tour: List[int]):
        print("Route:")
        route_str = str(tour[0])
        for i, node in enumerate(tour[1:]):
            if (i + 1) % 10 == 0:
                route_str += "\n"
            route_str += " -> " + str(node)
        print(route_str)

    def print_solution(self):

        print(f"Total time:\t {self.result['total_time']}")
        print(f"Tour length:\t {self.result['tour_length']}")
        print(f"Tour:\t {self.result['tour']}")

    @classmethod
    def track_results():
        """
        Tracks the results of each solver on each problem, saves them in a suitable format to disk
        """
        raise NotImplementedError

    def plot_solution(self):
        raise NotImplementedError()
