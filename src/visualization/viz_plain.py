import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()


RESULTS_DIR = Path(os.getenv("RESULTS_DIR", None))


def plot_solution_plain(result: dict, nodes: list):
    """
    Plots the solution as a plain graph with straight edges found by the solver and saves it.
    Note: Length of an edge does not represent the cost from node to node  (the travel time)

    Parameters:
    result (dict): The data about the solution including the tour and other metadata
    nodes (list): list of nodes in the graph that the solution was computed on.

    """
    if not result["tour"] or nodes is None:
        print("No tour or nodes available to plot.")
        return

    nodes_array = np.array(nodes)
    tour = result["tour"]
    tour_coords = nodes_array[tour]
    # 1. Create figure and set aspect ratio
    plt.figure(figsize=(10, 10))
    plt.axis("equal")

    # 2. Plot nodes
    plt.title(
        f"{result['problem']}: {result['solver']} Tour (Cost:{result['cost']:.2f})"
    )
    plt.scatter(nodes_array[:, 1], nodes_array[:, 0], color="blue", s=20, zorder=2)

    # 3. Draw path
    plt.plot(
        tour_coords[:, 1],
        tour_coords[:, 0],
        color="red",
        linestyle="-",
        linewidth=1,
        alpha=0.7,
        zorder=1,
    )

    # 4. Save and show
    solver_results_dir = Path(RESULTS_DIR / result["solver"])
    solver_results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        solver_results_dir
        / f"{result['timestamp']}_{result['problem']}_{result['solver']}_plain.png"
    )
