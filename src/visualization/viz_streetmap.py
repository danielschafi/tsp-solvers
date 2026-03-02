import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import tsplib95

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs

if not hasattr(np, "float_"):
    np.float_ = np.float64

from dotenv import load_dotenv

load_dotenv()

np.random.seed(42)


RESULTS_DIR = Path(os.getenv("RESULTS_DIR", None))


def plot_solution_streetmap(result: dict, tsp_problem_file: str):
    """
    Parameters:
    tsp_problem_file (str): The TSP problem instance, containing the OSM IDs of the nodes and the path to the graphml file.
    result (dict): The data about the solution including the tour and other metadata
    """
    problem = tsplib95.load(tsp_problem_file, problem_class=TSPProblemWithOSMIDs)
    graph_file = Path(problem.graphml_file)
    if not graph_file.exists():
        raise ValueError(
            f"Graphml {graph_file} file does not exists, cant visualize on streetmap"
        )
        return
    G = ox.load_graphml(graph_file)
    # get the osm ids (Node identifiers) of the solution (-1 because nodes in tsplib are 1 indexed [1,2,3,...])
    route_osm_ids = [problem.osm_ids[idx - 1] for idx in result["tour"]]
    full_street_route = []
    for i in range(len(route_osm_ids) - 1):
        start = route_osm_ids[i]
        end = route_osm_ids[i + 1]
        path = nx.shortest_path(G, start, end, weight="travel_time")
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

    # Legend
    from matplotlib.lines import Line2D

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

    solver_results_dir = Path(RESULTS_DIR / result["solver"])
    solver_results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        solver_results_dir
        / f"{result['timestamp']}_{result['problem']}_{result['solver']}_streetmap.png",
        dpi=300,
        facecolor=BACKGROUND,  # no bbox_inches="tight" — it was overriding subplots_adjust
    )
    plt.close(fig)


def main():
    plot_solution_streetmap(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/100/zurich_100_0.tsp",
        "~/thesis/tsp-solvers/results/",
    )


if __name__ == "__main__":
    main()
