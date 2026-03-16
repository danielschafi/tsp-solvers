import argparse
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tsplib95
from dotenv import load_dotenv

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs
from src.logger import setup_logging

load_dotenv()

logger = logging.getLogger("src.visualization.viz_plain")


def plot_solution_plain(result: dict, nodes: list | np.ndarray, results_dir: Path):
    """
    Plots the solution as a plain graph with straight edges found by the solver and saves it.
    Note: Length of an edge does not represent the cost from node to node (the travel time)
    Parameters:
    result (dict): The data about the solution including the tour and other metadata
    nodes (list): list of nodes in the graph that the solution was computed on.
    """
    if not result["tour"] or nodes is None:
        logger.warning("No tour or nodes available to plot.")
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

    plt.savefig(
        results_dir / f"{result['problem']}_plain.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=BACKGROUND,
    )
    plt.close(fig)


def _plot_one_plain(json_path: Path) -> None:
    """Plot a single result JSON as a plain graph, skipping if the PNG already exists."""
    json_path = Path(json_path)
    out_path = json_path.parent / f"{json_path.stem}_plain.png"
    if out_path.exists():
        logger.info(f"Skipping {json_path.name} (plain PNG already exists)")
        return

    with open(json_path) as f:
        result = json.load(f)

    tsp_file_path = result.get("tsp_file")
    if not tsp_file_path or not Path(tsp_file_path).exists():
        logger.warning(f"tsp_file not found for {json_path.name}, skipping")
        return

    problem = tsplib95.load(tsp_file_path, problem_class=TSPProblemWithOSMIDs)
    nodes = np.array(problem.node_locations)
    plot_solution_plain(result, nodes, json_path.parent)
    logger.info(f"Plotted {json_path.name} → {out_path.name}")


def main():
    arg_parser = argparse.ArgumentParser(
        description=(
            "Plot TSP plain solution(s) from result JSON file(s). "
            "Provide a path to a single JSON or a directory; directories are searched recursively. "
            "Files that already have a matching *_plain.png are skipped."
        )
    )
    arg_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a result JSON file or a directory containing result JSONs.",
    )
    arg_parser.add_argument(
        "--workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel worker processes (default: os.cpu_count()).",
    )

    args = arg_parser.parse_args()
    setup_logging()

    path = Path(args.path)
    if path.is_file():
        json_files = [path]
    elif path.is_dir():
        json_files = sorted(path.rglob("*.json"))
    else:
        logger.error(f"Path {path} does not exist.")
        return

    logger.info(f"Found {len(json_files)} JSON file(s) under {path}")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            pool.map(_plot_one_plain, json_files)
    else:
        for json_file in json_files:
            _plot_one_plain(json_file)


if __name__ == "__main__":
    main()
