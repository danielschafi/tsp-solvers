import argparse
import logging
import time
from pathlib import Path

import cudf
import numpy as np
from cuopt import routing
from dotenv import load_dotenv

from src.logger import setup_logging
from src.solvers.solver_base import TSPSolver

logger = logging.getLogger("src.solvers.cuopt_solver")

load_dotenv()
np.random.seed(42)


class CuOptSolver(TSPSolver):
    """
    Solver for TSP using cuOpt
    """

    _gpu_warmed_up: bool = False

    def __init__(self, results_dir=None, timeout: float | None = None):
        super().__init__(solver="cuOpt", results_dir=results_dir, timeout=timeout)
        if not CuOptSolver._gpu_warmed_up:
            self._warmup()
            CuOptSolver._gpu_warmed_up = True

    def _warmup(self):
        """
        Warms up the solver.
        The first call to solve often includes some overhead for initialization, JIT compilation and memory allocation.
        We do not want to include that time in the benchmark
        """

        logger.info("Warming up GPU...")
        n = 20
        rng = np.random.default_rng(42)
        a = rng.integers(1, 10, size=(n, n)).astype(np.float32)
        sym = a @ a.T
        np.fill_diagonal(sym, 0)

        cost_matrix = cudf.DataFrame(sym)

        # Create data model and settings
        n_vehicles = 1
        dm = routing.DataModel(n, n_vehicles)
        dm.add_cost_matrix(cost_matrix)
        dm.add_transit_time_matrix(cost_matrix.copy(deep=True))

        ss = routing.SolverSettings()
        routing.Solve(dm, ss)
        logger.info("Finished warmup")

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
        n_locations = self.problem.dimension
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

        """
        Accuracy may be impacted. Problem under 100 locations may be solved with reasonable accuracy under a second.
        Larger problems may need a few minutes.
        A generous upper bond is to set the number of seconds to num_locations.
        By default it is set to num_locations/5.
        If increased accuracy is desired, this needs to set to higher numbers.

        """
        time_limit = (
            self.timeout if self.timeout is not None else self.problem.dimension
        )
        ss.set_time_limit(time_limit)  # seconds
        # ss.set_solution_scope(routing.SolutionScope.FEASIBLE)

        # Solve the routing problem
        self._start_time = time.perf_counter()
        sol = routing.Solve(self.data_model, ss)
        self._end_time = time.perf_counter()

        if hasattr(sol, "get_status"):
            logger.info(f"Status : {sol.get_status()}")
        logger.info(f"Time   : {self._end_time - self._start_time}s")

        if hasattr(sol, "get_route"):
            logger.info(f"Tour length : {len(sol.get_route())}")
        if hasattr(sol, "get_cost"):
            logger.info(f"Tour cost : {sol.get_cost()}")
        if hasattr(sol, "get_vehicle_count"):
            logger.info(f"Vehicle count : {sol.get_vehicle_count()}")

        self.result["time_to_solve"] = self._end_time - self._start_time

        # Display results
        if sol.get_status() == 0:
            result = sol.get_route().to_arrow().to_pylist()

            tour = [point["location"] for point in result]

            self.result["solution_status"] = "success"
            self.result["tour"] = tour
            self.result["cost"] = self.calculate_tour_cost(tour)  # arrival_times[-1]

        elif sol.get_status() == 1:
            self.result["solution_status"] = "fail"
            self.result["timed_out_without_tour"] = False
        elif sol.get_status() == 2:
            self.result["solution_status"] = "timeout"
            self.result["timed_out_without_tour"] = True
        elif sol.get_status() == 3:
            self.result["solution_status"] = "empty"
            self.result["timed_out_without_tour"] = False

        if sol.get_status() in [1, 2, 3]:
            logger.warning(
                f"SOLVER NOT SUCCESSFUL — status: {self.result['solution_status']}"
            )


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the cuOpt solver on a .tsp file or all .tsp files in a folder."
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
        solver = CuOptSolver()
        solver.run(str(path))
    elif path.is_dir():
        files = sorted(path.rglob("*.tsp"))
        solver = CuOptSolver()
        for i, tsp_file in enumerate(files):
            logger.info(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver.run(str(tsp_file))


if __name__ == "__main__":
    main()
