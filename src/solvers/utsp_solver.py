import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from neural.model.gnn import ScatteringAttentionGNN
from neural.scripts import inference
from src.logger import setup_logging
from src.solvers.solver_base import TSPSolver

logger = logging.getLogger("src.solvers.utsp_solver")

np.random.seed(42)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class UTSPNeuralSolver(TSPSolver):
    """
    Solver for TSP using cuOpt
    """

    _gpu_warmed_up: bool = False

    def __init__(self, results_dir=None, timeout: float | None = None):
        super().__init__(
            solver="UTSPNeuralSolver", results_dir=results_dir, timeout=timeout
        )

        self.model: ScatteringAttentionGNN | None = None

    def _warmup(self):
        """
        Warms up the solver.
        The first call to solve often includes some overhead for initialization, JIT compilation and memory allocation.
        We do not want to include that time in the benchmark
        """

        logger.info("Warming up GPU...")

        if self.model is None:
            raise ValueError("Model needs to be loaded before warmup")

        rng = np.random.default_rng(42)
        coords_raw = rng.integers(0, 10, size=(self.dim, 2)).astype(np.float32)
        adj_raw = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                adj_raw[i, j] = np.linalg.norm(coords_raw[i] - coords_raw[j]).astype(
                    np.float32
                )

        adj, coords = inference._preprocess_data(adj_raw, coords_raw)
        adj = adj.unsqueeze(0)  # [1, n, n]
        coords = coords.unsqueeze(0)  # [1, n, 2]

        inference._model_predict_problem(self.model, adj, coords)  # [1, n, n]

        logger.info("Finished warmup")

    def setup(self, tsp_file: str):
        """
        Prepares data and model for solving
        """
        # Load data, get dim to load correct model
        problem = inference._load_tsp_file(tsp_file)
        adj_raw, coords_raw, dim = inference._extract_relevant_parts_from_tsp_problem(
            problem
        )
        adj, coords = inference._preprocess_data(adj_raw, coords_raw)
        self.adj = adj.unsqueeze(0)  # [1, n, n]
        self.coords = coords.unsqueeze(0)  # [1, n, 2]

        # Only reload model if dimension changed
        if dim != self.dim or self.model is None:
            self.dim = dim
            self.model_path = Path("saved_models") / str(self.dim) / "best.pt"
            logger.info(f"Loading model for dim={self.dim} from {self.model_path}")
            self.model = inference._load_model(str(self.model_path))

            # Warming up
            if not UTSPNeuralSolver._gpu_warmed_up:
                self._warmup()
                UTSPNeuralSolver._gpu_warmed_up = True

        self.load_tsp_file(tsp_file)
        self.edges = adj_raw
        self.nodes = coords_raw

    def solve_tsp(self):
        """
        Solves the TSP
        """
        if self.model is None:
            raise ValueError("Model not initialized!")
        if self.edges is None:
            raise ValueError("self.edges are empty!")

        self._start_time = time.perf_counter()
        # GNN
        inference_time_start = time.perf_counter()
        heatmap = inference._model_predict_problem(
            self.model, self.adj, self.coords
        )  # [1, n, n]
        inference_time_end = time.perf_counter()

        # Guided Local Search
        local_search_time_start = time.perf_counter()
        tour = inference._run_guided_local_search(heatmap, self.edges)
        if tour[0] != tour[-1]:
            tour.append(tour[0])
        local_search_time_end = time.perf_counter()
        self._end_time = time.perf_counter()

        logger.info(f"Time   : {self._end_time - self._start_time}s")
        logger.info(
            f"Model Inference Time: {inference_time_end - inference_time_start}"
        )
        logger.info(
            f"Local Search Time: {local_search_time_end - local_search_time_start}"
        )

        self.result["time_to_solve"] = self._end_time - self._start_time
        self.result["additional_metadata"] = {
            "local_search_time": local_search_time_end - local_search_time_start,
            "model_inference_time": inference_time_end - inference_time_start,
        }
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
        solver = UTSPNeuralSolver()
        solver.run(str(path))
    elif path.is_dir():
        files = sorted(path.rglob("*.tsp"))
        solver = UTSPNeuralSolver()
        for i, tsp_file in enumerate(files):
            logger.info(f"Solving {tsp_file} ({i + 1}/{len(files)})")
            solver.run(str(tsp_file))


if __name__ == "__main__":
    main()


"""
1. Run one file through this and successfully predict -> done
2. run a folder through this -> For size 25 ca 17 seconds per problem. 
3. fix warmup and model loading
4. run through benchmark

- Fix benchmark cuopt loading if necessary
- fix saving / writing of results, add aggregation, auto plot on completion with updated runs. 
    Maybe for this also do two separate jsons. 

"""
