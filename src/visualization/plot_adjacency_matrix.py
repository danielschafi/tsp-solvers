from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from dotenv import load_dotenv

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs


def plot_heatmap(matrix: np.ndarray, title: str, results_dir: Path | None = None):
    """
    Plots the adjacency matrix as a heatmap and saves it.
    Parameters:
    matrix (np.ndarray): The adjacency matrix to plot.
    title (str): The title of the plot, used for the filename.
    results_dir (Path | None): The directory where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Distance")
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.tight_layout()
    save_path = (
        results_dir / f"{title.replace(' ', '_')}_adjacency_matrix.png"
        if results_dir
        else f"{title.replace(' ', '_')}_adjacency_matrix.png"
    )
    # plt.show()

    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    load_dotenv()
    # Example usage
    tsp_file = "data/tsp_dataset/100/zurich_100_0.tsp"
    # results_dir = Path("data/tsp_dataset/100")
    # results_dir.mkdir(parents=True, exist_ok=True)

    problem = tsplib95.load(tsp_file, problem_class=TSPProblemWithOSMIDs)
    adjacency_matrix = np.array(problem.edge_weights)

    print(
        f"Min: {adjacency_matrix.min()}, Max: {adjacency_matrix.max()}, Mean: {adjacency_matrix.mean()}"
    )
    plot_heatmap(adjacency_matrix, "TSP Adjacency Matrix")
