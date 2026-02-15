from abc import ABC, abstractmethod


class TSPSolver(ABC):
    """
    Base Class for TSP Solvers

    Ensures, that each solver can be used with a common interface, tracks metrics and provides convenience methods.
    """

    all_results: dict = {}

    def __init__(self, solver: str = "undefined", problem_name: str = "undefined"):
        self.solver: str = solver
        self.result: dict = {"total_time": 0, "tour_length": 0, "tour": []}
        self._start_time = None
        self._end_time = None

        print("\n")
        print("=" * 100)
        print(f"Solver: {self.solver}")
        print(f"Problem: {problem_name}")
        print("=" * 100)

    @abstractmethod
    def setup_problem(self, tsp_file):
        """
        Sets up the TSP Problem.
        Reads a .tsp file and prepares the data for the solver
        """
        pass

    @abstractmethod
    def solve_tsp(self):
        """
        Solves the TSP problem with the solver
        """
        pass

    def print_solution(self):

        print(f"Total time:\t {self.result['total_time']}")
        print(f"Tour length:\t {self.result['tour_length']}")
        print(f"Tour:\t {self.result['tour']}")

    @classmethod
    def track_results():
        """
        Tracks the results of each solver on each problem, saves them in a suitable format to disk
        """

    def plot_solution(self):
        raise NotImplementedError()
