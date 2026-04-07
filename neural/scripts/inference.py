import argparse
from pathlib import Path

import numpy as np
import torch
import tsplib95
from torch import Tensor

from neural.config.loader import get_model_config
from neural.local_search.mcts_wrapper import run_mcts
from neural.model.gnn import ScatteringAttentionGNN
from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Must match Rec_Num compiled into bin/MCTS-UTSP/code/include/TSP_IO.h
REC_NUM = 20


def _load_model(
    model_path: str,
    problem_size: int,
) -> ScatteringAttentionGNN:
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path} - train a model first"
        )

    model_cfg = get_model_config(problem_size)
    hidden_dim = model_cfg.get("hidden_dim", 64)
    n_layers = model_cfg.get("n_layers", 3)
    node_features = model_cfg.get("node_features", "node_stats")

    model = ScatteringAttentionGNN(
        hidden_dim=hidden_dim,
        output_dim=problem_size,
        n_layers=n_layers,
        node_features=node_features,
    )
    checkpoint = torch.load(Path(model_path), weights_only=False, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


def _load_tsp_file(tsp_file: Path | str) -> TSPProblemWithOSMIDs:
    if not Path(tsp_file).exists():
        raise FileNotFoundError(f"tsp_file: {tsp_file} does not exist.")
    return tsplib95.load(Path(tsp_file), problem_class=TSPProblemWithOSMIDs)  # type: ignore[return-value]


def _extract_relevant_parts_from_tsp_problem(
    problem: TSPProblemWithOSMIDs,
) -> tuple[np.ndarray, np.ndarray, int]:
    adj = np.array(problem.edge_weights)
    coords = np.array(problem.node_locations)
    dim: int = problem.dimension  # type: ignore[assignment]
    return adj, coords, dim


def _preprocess_data(
    adjacency: np.ndarray, coordinates: np.ndarray
) -> tuple[Tensor, Tensor]:
    """Same preprocessing as training data."""
    adj = torch.tensor(adjacency, dtype=torch.float32)
    coords = torch.tensor(coordinates, dtype=torch.float32)

    adj = adj + adj.T

    diag_mask = ~torch.eye(n=adj.size(0), dtype=torch.bool)
    adj = adj / adj[diag_mask].mean()

    coords = coords - coords.mean(dim=0)
    return adj, coords


def _model_predict_problem(
    model: ScatteringAttentionGNN,
    distances: Tensor,
    coords: Tensor,
    temperature: float = 3.5,
) -> Tensor:
    """Run the GNN forward pass.

    Args:
        distances: [1, n, n] normalized distance matrix.
        coords: [1, n, 2] centered coordinates.
        temperature: temperature for exp adjacency conversion.
    """
    model.eval()

    adj = torch.exp(-1 * distances / temperature)
    adj.diagonal(dim1=1, dim2=2).fill_(0)

    with torch.no_grad():
        output = model(
            adj.to(DEVICE), distances=distances.to(DEVICE), coords=coords.to(DEVICE)
        )
    return output  # [1, n, n]


def _heatmap_to_topk(heatmap: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract top-k neighbor indices and scores per node from the heatmap.

    Args:
        heatmap: [1, n, n] model output (edge probability matrix).
        k:       Number of top neighbors to keep (must match Rec_Num in TSP_IO.h).

    Returns:
        topk_idx: [1, n, k] int64 array of 0-based neighbor indices.
        topk_val: [1, n, k] float32 array of corresponding heatmap scores.
    """
    hm = heatmap.detach().clone().cpu().float()
    n = hm.size(1)

    # Mask diagonal so a node is never its own top-k neighbor
    eye = torch.eye(n, dtype=torch.bool).unsqueeze(0)  # [1, n, n]
    hm[eye] = float("-inf")

    k = min(k, n - 1)
    topk = torch.topk(hm, k=k, dim=2)  # values/indices: [1, n, k]
    # Use tolist() to avoid the PyTorch↔NumPy C bridge (broken with NumPy 2.x + compiled modules)
    topk_idx = np.array(topk.indices.tolist(), dtype=np.int64)
    topk_val = np.array(topk.values.tolist(), dtype=np.float32)
    return topk_idx, topk_val


def _compute_tour_length(adj_raw: np.ndarray, tour: list[int]) -> int:
    # max(A, A^T) matches solver_base.calculate_tour_cost and handles both
    # full-symmetric and upper-triangular TSPLIB storage formats.
    adj_sym = np.maximum(adj_raw, adj_raw.T)
    n = len(tour)
    return int(sum(adj_sym[tour[i], tour[(i + 1) % n]] for i in range(n)))


def _run_guided_local_search(heatmap: Tensor, adj_raw: np.ndarray) -> list[int]:
    """Run MCTS guided by the GNN heatmap to find a good tour.

    Args:
        heatmap:  [1, n, n] model output (on any device).
        adj_raw:  [n, n] raw (unnormalized) adjacency matrix from the .tsp file.

    Returns:
        0-based tour as a list of city indices.
    """
    adj_sym = np.maximum(adj_raw, adj_raw.T)  # same logic as _compute_tour_length
    dist_matrix = adj_sym[np.newaxis, :, :].astype(np.int64)  # [1, n, n]

    topk_idx, topk_val = _heatmap_to_topk(heatmap, k=REC_NUM)  # [1, n, REC_NUM]

    tours = run_mcts(dist_matrix, topk_idx, topk_val)
    return tours[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a single TSP instance using GNN heatmap + MCTS local search. "
            "Reads a .tsp file, runs the GNN to get edge probabilities, then guides "
            "MCTS with those probabilities to find a good tour."
        )
    )
    parser.add_argument(
        "--tsp_file",
        default="data/tsp_dataset/25/zurich_25_0.tsp",
    )
    parser.add_argument(
        "--model_weights",
        default="saved_models/25/best_29ov4804.pt",
    )
    args = parser.parse_args()

    tour = run_utsp_pipeline(args.tsp_file, args.model_weights)


def run_utsp_pipeline(tsp_file: str, model_weights: str) -> list[int]:
    print(f"Device: {DEVICE}")
    print(f"TSP file: {tsp_file}")
    print(f"Model weights: {model_weights}")

    problem = _load_tsp_file(tsp_file)
    adj_raw, coords_raw, dim = _extract_relevant_parts_from_tsp_problem(problem)
    print(f"Problem size: {dim} cities")

    model = _load_model(model_weights, problem_size=dim)

    distances, coords = _preprocess_data(adj_raw, coords_raw)
    distances = distances.unsqueeze(0)  # [1, n, n]
    coords = coords.unsqueeze(0)  # [1, n, 2]

    heatmap = _model_predict_problem(model, distances, coords)  # [1, n, n]

    tour = _run_guided_local_search(heatmap, adj_raw)

    # To follow the other solvers, include return to home
    if tour[0] != tour[-1]:
        tour.append(tour[0])

    tour_length = _compute_tour_length(adj_raw, tour)

    print(f"Tour (0-based): {tour}")
    print(f"Tour length:    {tour_length}")

    return tour


if __name__ == "__main__":
    main()
