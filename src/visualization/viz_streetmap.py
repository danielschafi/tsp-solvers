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

        # shortest path for vizualization
        path = nx.shortest_path(G, start, end, weight="travel_time")
        full_street_route.extend(path[:-1])  # otherwise last one duplicate
    full_street_route.append(route_osm_ids[-1])

    # 4. Plot
    # Plot the background graph in light gray
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color="#999999",
        edge_linewidth=0.5,
        show=False,
        close=False,
    )

    # Plot the specific TSP tour in red
    fig, ax = ox.plot_graph_route(
        G,
        full_street_route,
        route_color="r",
        route_linewidth=3,
        node_size=0,
        ax=ax,
        show=False,
        close=False,
    )

    # Highlight the TSP "Stop" nodes in blue
    stop_coords = [problem.node_locations[idx] for idx in result["tour"]]
    y, x = zip(*stop_coords)
    ax.scatter(x, y, c="blue", s=20, zorder=5, label="TSP Stops")

    solver_results_dir = Path(RESULTS_DIR / result["solver"])
    solver_results_dir.mkdir(parents=True, exist_ok=True)
    plt.title(
        f"{result['problem']}: {result['solver']} Tour (Cost:{result['cost']:.2f})"
    )
    plt.savefig(
        solver_results_dir
        / f"{result['timestamp']}_{result['problem']}_{result['solver']}_streetmap.png"
    )


def main():
    plot_solution_streetmap(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/100/zurich_100_0.tsp",
        "~/thesis/tsp-solvers/results/",
    )


if __name__ == "__main__":
    main()
