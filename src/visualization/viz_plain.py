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
    Note: Length of an edge does not represent the cost from node to node (the travel time)
    Parameters:
    result (dict): The data about the solution including the tour and other metadata
    nodes (list): list of nodes in the graph that the solution was computed on.
    """
    if not result["tour"] or nodes is None:
        print("No tour or nodes available to plot.")
        return

    # --- Match street map styling ---
    BACKGROUND = "#F5F5F0"
    ROUTE_COLOR = "#D62728"
    STOP_COLOR = "#1F77B4"
    STOP_EDGE = "#FFFFFF"

    nodes_array = np.array(nodes)
    tour = result["tour"]
    tour_coords = nodes_array[tour]

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")

    # Tour route
    ax.plot(
        tour_coords[:, 1],
        tour_coords[:, 0],
        color=ROUTE_COLOR,
        linestyle="-",
        linewidth=2.0,
        alpha=0.85,
        zorder=1,
        label="TSP Tour",
    )

    # Stop nodes
    ax.scatter(
        nodes_array[:, 1],
        nodes_array[:, 0],
        color=STOP_COLOR,
        edgecolors=STOP_EDGE,
        linewidths=0.8,
        s=60,
        zorder=2,
        label="TSP Stops",
    )

    ax.set_title(
        f"TSP Optimization: {result['problem']}\n"
        f"Solver: {result['solver']}  |  Total Cost: {result['cost']:.2f}",
        fontsize=11,
        fontweight="bold",
        color="#222222",
        pad=10,
    )
    ax.legend(
        loc="lower right",
        framealpha=0.9,
        fontsize=9,
        edgecolor="#AAAAAA",
    )

    fig.tight_layout(pad=1.5)

    solver_results_dir = Path(RESULTS_DIR / result["solver"])
    solver_results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        solver_results_dir
        / f"{result['timestamp']}_{result['problem']}_{result['solver']}_plain.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=BACKGROUND,
    )
    plt.close(fig)
