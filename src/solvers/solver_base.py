from abc import ABC, abstractmethod


class TSPSolver(ABC):
    def __init__(self, name: str = "undefined", problem_name: str = "undefined"):
        self.name: str = name
        self.result: dict = {"total_time": 0, "tour_length": 0, "tour": []}

        print("\n")
        print("=" * 100)
        print(f"Solver: {self.name}")
        print(f"Problem: {self.problem}")
        print("=" * 100)

    @abstractmethod
    def load_data(self, tsplib_problem):
        pass

    @abstractmethod
    def solve_tsp(self):
        pass

    def print_solution(self):
        print(f"Total time:\t {self.result['total_time']}")
        print(f"Tour length:\t {self.result['tour_length']}")
        print(f"Tour:\t {self.result['tour']}")

    def plot_solution(self):
        raise NotImplementedError()
