import cudf
import tsplib95
from cuopt import routing
from solver_base import TSPSolver


class CuOptSolver(TSPSolver):
    def __init__(self):
        super().__init__(solver="CuOpt")

    def load_data(self):
        raise NotImplementedError()

    def solve_tsp(self):
        raise NotImplementedError()


def main():

    # load one tsplib problem
    problem_path = (
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsplib/att48.tsp"
    )
    problem = tsplib95.load(problem_path)

    # Create cost matrix (symmetric distance matrix for 4 locations)
    cost_matrix = cudf.DataFrame(
        [[0, 2, 2, 2], [2, 0, 2, 2], [2, 2, 0, 2], [2, 2, 2, 0]], dtype="float32"
    )

    # Task locations (indices into the cost matrix)
    # Tasks at locations 1, 2, and 3
    task_locations = cudf.Series([1, 2, 3])

    # Number of vehicles
    n_vehicles = 2

    # Create data model
    dm = routing.DataModel(cost_matrix.shape[0], n_vehicles, len(task_locations))
    dm.add_cost_matrix(cost_matrix)
    dm.add_transit_time_matrix(cost_matrix.copy(deep=True))

    # Configure solver settings
    ss = routing.SolverSettings()

    # Solve the routing problem
    sol = routing.Solve(dm, ss)

    # Display results
    print(sol.get_route())
    print("\n\n****************** Display Routes *************************")
    sol.display_routes()


if __name__ == "__main__":
    main()
