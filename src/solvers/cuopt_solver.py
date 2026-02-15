import time
from pathlib import Path

import cudf
import numpy as np
import tsplib95
from cuopt import routing
from solver_base import TSPSolver


class CuOptSolver(TSPSolver):
    """
    Solver for TSP using cuOpt
    """

    def __init__(self):
        super().__init__(solver="cuOpt")
        self._warmup()

    def _warmup(self):
        """
        Warms up the solver.
        The first call to solve often includes some overhead for initialization, JIT compilation and memory allocation.
        We do not want to include that time in the benchmark
        """

        print("Warming up GPU...")
        cost_matrix = cudf.DataFrame(
            [[0, 2, 2, 2], [2, 0, 2, 2], [2, 2, 0, 2], [2, 2, 2, 0]], dtype="float32"
        )

        # Create data model and settings
        n_vehicles = 1
        dm = routing.DataModel(cost_matrix.shape[0], n_vehicles)
        dm.add_cost_matrix(cost_matrix)
        dm.add_transit_time_matrix(cost_matrix.copy(deep=True))

        ss = routing.SolverSettings()
        sol = routing.Solve(dm, ss)
        print("Finshed warmup")
        if sol.get_status() == 0:
            sol.display_routes()
            print(self.result["tour"])
        else:
            print("Solver failed to find a solution.")

    def setup_problem(self, tsp_file: str):
        """
        Prepares the data for the cuOpt solver.
        Builds the adjacendy matrix,
        """
        if not Path(tsp_file).exists():
            raise FileNotFoundError(f"tsp_file: {tsp_file} does not exist.")

        self._tsp_file = Path(tsp_file)

        problem = tsplib95.load(self._tsp_file)
        edge_weights = np.zeros((problem.dimension, problem.dimension))

        n_locations = problem.dimension
        for i in range(n_locations):
            for j in range(n_locations):
                # problem.node_coords dict starts at index 1
                edge_weights[i][j] = problem.get_weight(i + 1, j + 1)

        cost_matrix = cudf.DataFrame(edge_weights, dtype="float32")

        # Create data model
        n_vehicles = 1
        self.data_model = routing.DataModel(n_locations, n_vehicles)
        self.data_model.add_cost_matrix(cost_matrix)
        self.data_model.add_transit_time_matrix(cost_matrix.copy(deep=True))

    def solve_tsp(self):
        """
        Solves the TSP using cuOpt
        """
        print("Start solving TSP")
        # Configure solver settings
        ss = routing.SolverSettings()
        ss.set_time_limit(36)  # 360 seconds

        # Solve the routing problem
        self._start_time = time.perf_counter()
        sol = routing.Solve(self.data_model, ss)
        self._end_time = time.perf_counter()
        self.result["total_time"] = self._end_time - self._start_time

        # Display results
        if sol.get_status() == 0:
            sol.display_routes()
            self.result["tour"] = sol.get_route().to_arrow().to_pylist()

        else:
            print("Solver failed to find a solution.")

        # print(sol.get_route())

    def print_solution(self):
        print(f"Total time:\t {self.result['total_time']}")
        print(f"Tour length:\t {self.result['tour_length']}")
        print(f"Tour:\t {self.result['tour']}")

    def plot_solution(self):
        raise NotImplementedError()


def main():

    solver = CuOptSolver()
    solver.setup_problem(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsplib/att48.tsp"
    )
    solver.solve_tsp()


if __name__ == "__main__":
    main()
