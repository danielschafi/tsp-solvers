import time

import cudf
import numpy as np
from cuopt import routing
from dotenv import load_dotenv

from src.solvers.solver_base import TSPSolver

load_dotenv()


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
        routing.Solve(dm, ss)
        print("Finshed warmup")

    def setup_problem(self, tsp_file: str):
        """
        Prepares the data for the cuOpt solver.
        Builds the adjacendy matrix,
        """
        self.load_tsp_file(tsp_file)
        self.edges = np.array(self.problem.edge_weights)
        self.nodes = np.array(self.problem.node_locations)

        cost_matrix = cudf.DataFrame(self.edges, dtype="float32")
        # Create data model
        n_vehicles = 1
        self.data_model = routing.DataModel(n_locations, n_vehicles)
        self.data_model.add_cost_matrix(cost_matrix)
        self.data_model.add_transit_time_matrix(cost_matrix.copy(deep=True))

    def solve_tsp(self):
        """
        Solves the TSP using cuOpt
        """
        # Configure solver settings
        ss = routing.SolverSettings()
        ss.set_time_limit(600)  # seconds

        # Solve the routing problem
        self._start_time = time.perf_counter()
        sol = routing.Solve(self.data_model, ss)
        self._end_time = time.perf_counter()
        self.result["time_to_solve"] = self._end_time - self._start_time

        # Display results
        if sol.get_status() == 0:
            result = sol.get_route().to_arrow().to_pylist()

            tour = [point["location"] for point in result]
            arrival_times = [point["arrival_stamp"] for point in result]

            self.result["solution_status"] = "success"
            self.result["tour"] = tour
            self.result["cost"] = self.calculate_tour_cost(tour)  # arrival_times[-1]

        elif sol.get_status() == 1:
            self.result["solution_status"] = "fail"
        elif sol.get_status() == 2:
            self.result["solution_status"] = "timeout"
        elif sol.get_status() == 3:
            self.result["solution_status"] = "empty"

        if sol.get_status() in [1, 2, 3]:
            print("!!! WARNING !!!")
            print("SOLVER NOT SUCCESSFUL")


def main():
    solver = CuOptSolver()
    solver.run(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/10/zurich_10_0.tsp"
    )


if __name__ == "__main__":
    main()
