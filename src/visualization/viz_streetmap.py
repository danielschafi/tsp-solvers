import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import tsplib95
from matplotlib.lines import Line2D

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs

if not hasattr(np, "float_"):
    np.float_ = np.float64

from dotenv import load_dotenv

load_dotenv()

np.random.seed(42)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _load_graph_cached(graph_file: str):
    """
    Loads the graph from the graphml file, with caching to speed up repeated calls for the same graph.
    """
    return ox.load_graphml(Path(graph_file))


@lru_cache(maxsize=200000)
def _shortest_path_cached(graph_file: str, start, end, weight: str = "travel_time"):
    """
    Shortest path between two nodes in the graph, with caching to speed up repeated calls.
    """
    graph = _load_graph_cached(graph_file)
    return tuple(nx.shortest_path(graph, start, end, weight=weight))


def clear_streetmap_caches():
    _shortest_path_cached.cache_clear()
    _load_graph_cached.cache_clear()


def plot_solution_streetmap(result: dict, tsp_problem_file: str, results_dir: Path):
    """
    Parameters:
    tsp_problem_file (str): The TSP problem instance, containing the OSM IDs of the nodes and the path to the graphml file.
    result (dict): The data about the solution including the tour and other metadata
    """
    problem = tsplib95.load(tsp_problem_file, problem_class=TSPProblemWithOSMIDs)
    graph_file = Path(problem.graphml_file).expanduser().resolve()
    if not graph_file.exists():
        raise ValueError(
            f"Graphml {graph_file} file does not exists, cant visualize on streetmap"
        )
        return
    graph_key = str(graph_file)
    G = _load_graph_cached(graph_key)
    # get the osm ids (Node identifiers) of the solution (-1 because nodes in tsplib are 1 indexed [1,2,3,...])
    route_osm_ids = tuple(problem.osm_ids[idx - 1] for idx in result["tour"])
    full_street_route = []
    for i in range(len(route_osm_ids) - 1):
        start = route_osm_ids[i]
        end = route_osm_ids[i + 1]
        path = _shortest_path_cached(graph_key, start, end, weight="travel_time")
        full_street_route.extend(path[:-1])
    full_street_route.append(route_osm_ids[-1])

    # --- Publication-ready styling ---
    BACKGROUND = "#F5F5F0"  # warm off-white page background
    STREET_COLOR = "#CCCCCC"  # light grey street network
    ROUTE_COLOR = "#D62728"  # vivid red tour route
    STOP_COLOR = "#1F77B4"  # muted blue stop markers
    STOP_EDGE = "#FFFFFF"  # white ring around stops for contrast

    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor=BACKGROUND,
        node_size=0,
        edge_color=STREET_COLOR,
        edge_linewidth=0.6,
        edge_alpha=1.0,
    )

    fig, ax = ox.plot_graph_route(
        G,
        full_street_route,
        ax=ax,
        show=False,
        close=False,
        bgcolor=BACKGROUND,
        route_color=ROUTE_COLOR,
        route_linewidth=2.0,
        route_alpha=0.85,
        node_size=0,
        edge_color=STREET_COLOR,
        edge_linewidth=0.6,
        edge_alpha=1.0,
    )

    # TSP stop nodes
    stop_coords = [problem.node_locations[idx] for idx in result["tour"]]
    y, x = zip(*stop_coords)
    ax.scatter(
        x,
        y,
        s=60,
        color=STOP_COLOR,
        edgecolors=STOP_EDGE,
        linewidths=0.8,
        zorder=5,
        label="TSP Stops",
    )

    legend_elements = [
        Line2D([0], [0], color=ROUTE_COLOR, linewidth=2.0, label="TSP Tour"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=STOP_COLOR,
            markeredgecolor=STOP_EDGE,
            markersize=7,
            label="TSP Stops",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        framealpha=0.9,
        fontsize=9,
        edgecolor="#AAAAAA",
    )

    # Title & layout
    ax.set_title(
        f"TSP Optimization: {result['problem']}\n"
        f"Solver: {result['solver']}  |  Total Cost: {result['cost']:.2f}",
        fontsize=11,
        fontweight="bold",
        color="#222222",
        pad=10,
    )
    fig.patch.set_facecolor(BACKGROUND)

    # osmnx sets axes to fill the full figure — shrink it down to leave title room
    ax.set_position([0, 0, 1, 0.88])

    plt.savefig(
        results_dir / f"{result['problem']}_streetmap.png",
        dpi=300,
        facecolor=BACKGROUND,  # no bbox_inches="tight" — it was overriding subplots_adjust
    )
    plt.close(fig)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Plot the TSP solution on a streetmap. Provide the path to the TSP problem file and the results json."
    )
    arg_parser.add_argument(
        "--tsp_file",
        type=str,
        required=True,
        help="Path to the .tsp file containing the problem instance with OSM IDs and graphml file path.",
    )
    arg_parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="Path to the results json file containing the solution tour and metadata.",
    )

    args = arg_parser.parse_args()
    tsp_file = Path(args.tsp_file)
    results_json = Path(args.results_json)

    if not tsp_file.exists():
        logger.error(f"TSP file {tsp_file} does not exist.")
        return
    if not results_json.exists():
        logger.error(f"Results json file {results_json} does not exist.")
        return

    with open(results_json, "r") as f:
        result = json.load(f)

    plot_solution_streetmap(result, str(tsp_file), results_json.parent)


if __name__ == "__main__":
    main()
