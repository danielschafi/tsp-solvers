import argparse
import logging
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.logger import setup_logging
from src.solvers.solver_base import TSPSolver

logger = logging.getLogger("src.solvers.neural_solver")

load_dotenv()
np.random.seed(42)


class NeuralSolver(TSPSolver):
    """
    Solver for TSP using cuOpt
    """

    _gpu_warmed_up: bool = False

    def __init__(self, results_dir=None, timeout: float | None = None):
        super().__init__(solver="cuOpt", results_dir=results_dir, timeout=timeout)
        if not NeuralSolver._gpu_warmed_up:
            self._warmup()
            NeuralSolver._gpu_warmed_up = True

    def _warmup(self):
        """
        Warms up the solver.
        The first call to solve often includes some overhead for initialization, JIT compilation and memory allocation.
        We do not want to include that time in the benchmark
        """

        raise NotImplementedError()

    def setup_problem(self, tsp_file: str):
        """
        Prepares the data for the cuOpt solver.
        Builds the adjacendy matrix,
        """
        self.load_tsp_file(tsp_file)
        self.edges = np.array(self.problem.edge_weights)
        self.nodes = np.array(self.problem.node_locations)

        raise NotImplementedError()

    def solve_tsp(self):
        """
        Solves the TSP
        """

        raise NotImplementedError()


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the neural solver on a .tsp file or all .tsp files in a folder. Or a h5 file that has been created using the prepare_data_for_gnn_training.py"
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
    # if path.is_file():
    #     solver = CuOptSolver()
    #     solver.run(str(path))
    # elif path.is_dir():
    #     files = sorted(path.rglob("*.tsp"))
    #     solver = CuOptSolver()
    #     for i, tsp_file in enumerate(files):
    #         logger.info(f"Solving {tsp_file} ({i + 1}/{len(files)})")
    #         solver.run(str(tsp_file))

    # Maybe batch inference for this solver, creating of the dataset internally would be nice.


if __name__ == "__main__":
    main()
