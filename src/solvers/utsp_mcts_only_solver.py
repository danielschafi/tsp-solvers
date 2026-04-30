import argparse
import logging
import time
from pathlib import Path

import numpy as np

from neural.scripts import inference
from src.logger import setup_logging
from src.solvers.solver_base import TSPSolver

logger = logging.getLogger("src.solvers.mcts_only_solver")

np.random.seed(42)


class MCTSOnlySolver(TSPSolver):
    """
    Solver for TSP using only MCTS without any heatmap input. This should show how much the heatmap contributes to the solution quality.
    """

    def __init__(self, results_dir=None, timeout: float | None = None):
        super().__init__(solver="MCTSOnly", results_dir=results_dir, timeout=timeout)
        self.dim: int = None

    def setup(self, tsp_file: str):
        """
        Prepares data for solving (no model loading needed).
        """
        problem = inference._load_tsp_file(tsp_file)
        adj_raw, coords_raw, dim = inference._extract_relevant_parts_from_tsp_problem(
            problem
        )
        self.dim = dim

        self.load_tsp_file(tsp_file)
        self.edges = adj_raw
        self.nodes = coords_raw

    def solve_tsp(self):
        """
        Solves the TSP
        """

        if self.edges is None:
            raise ValueError("self.edges are empty!")

        self._start_time = time.perf_counter()

        # MCTS Search, need dummy top k input.
        # k must match compiled Rec_Num, but indices must be valid node ids (< n).
        k = inference.REC_NUM
        dummy_top_k_idx = np.tile(
            np.arange(k) % self.dim, (1, self.dim, 1)
        )  # [1, n, k]
        dummy_top_k_val = np.zeros((1, self.dim, k))

        dist_matrix = self.edges[np.newaxis, :, :].astype(np.int64)
        tours = inference.run_mcts(
            dist_matrix,
            dummy_top_k_idx,
            dummy_top_k_val,
            n_threads=1,
            use_rec=False,
            rec_only=False,
        )

        tour = tours[0]
        if tour[0] != tour[-1]:
            tour.append(tour[0])
        self._end_time = time.perf_counter()

        logger.info(f"Time   : {self._end_time - self._start_time}s")

        self.result["time_to_solve"] = self._end_time - self._start_time
        self.result["solution_status"] = "success"
        self.result["tour"] = tour
        self.result["cost"] = self.calculate_tour_cost(tour)


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

    if path.is_file():
        solver = MCTSOnlySolver()
        solver.run(str(path))
    elif path.is_dir():
        files = sorted(path.rglob("*.tsp"))
        solver = MCTSOnlySolver()
        for i, tsp_file in enumerate(files):
            logger.info(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver.run(str(tsp_file))


if __name__ == "__main__":
    main()
