import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import tsplib95

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs

if not hasattr(np, "float_"):
    np.float_ = np.float64


RESULTS_DIR = Path(os.getenv("RESULTS_DIR", None))
  
def plot_solution_on_streetmap(tsp_problem_file: str, result: dict):
    """
    Parameters:
    tsp_problem_file (str): The TSP problem instance, containing the OSM IDs of the nodes and the path to the graphml file.
    result (dict): The data about the solution including the tour and other metadata
    """

    problem = tsplib95.load(tsp_problem_file, problem_class=TSPProblemWithOSMIDs)
    graph_file = Path(problem.graphml_file)

    if not graph_file.exists():
        raise ValueError(f"Graphml {graph_file} file does not exists, cant visualize on streetmap")
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
    print("Rendering map...")
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
    stop_coords = [problem.node_coords[idx + 1] for idx in solution]
    y, x = zip(*stop_coords)
    ax.scatter(x, y, c="blue", s=20, zorder=5, label="TSP Stops")

    solver_results_dir = Path(RESULTS_DIR / result["solver"])
    solver_results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        solver_results_dir
        / f"{result['timestamp']}_{result['problem']}_{result['solver']}_streetmap.png"
    )



def plot_from_solution_json(json_path: str):
    json_path = Path(json_path)

    if not json_path.exists()

    result = json.load(json_path)


def plot_all_in_folder():
    pass


def main():
    test_solution = [
        0,
        32,
        93,
        60,
        11,
        30,
        76,
        85,
        21,
        5,
        48,
        16,
        51,
        61,
        3,
        59,
        99,
        50,
        82,
        1,
        36,
        17,
        15,
        44,
        42,
        94,
        35,
        14,
        83,
        7,
        41,
        92,
        38,
        66,
        43,
        53,
        75,
        95,
        10,
        28,
        8,
        37,
        64,
        98,
        68,
        18,
        91,
        34,
        26,
        65,
        69,
        24,
        22,
        97,
        27,
        86,
        74,
        87,
        19,
        80,
        56,
        54,
        47,
        89,
        33,
        39,
        46,
        67,
        90,
        81,
        78,
        57,
        70,
        25,
        52,
        20,
        58,
        62,
        4,
        12,
        73,
        88,
        31,
        55,
        2,
        84,
        63,
        23,
        40,
        45,
        72,
        49,
        79,
        13,
        29,
        6,
        71,
        9,
        77,
        96,
        0,
    ]  # Example solution (tour)
    G = ox.load_graphml(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/graph_zurich_dist_20000.graphml"
    )

    plot_solution_on_streetmap(
        G,
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/100/zurich_100_0.tsp",
        test_solution,
    )


if __name__ == "__main__":
    main()
