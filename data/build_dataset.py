"""
Generates TSP problem instances for a specified city by sampling nodes from the city's road network graph and creating travel time matrices.
The generated problem instances are saved in TSPLib format for use in TSP solvers.
"""

import argparse
import random
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from tqdm import tqdm


def build_city_graph(city_name: str = "Zurich, Switzerland") -> nx.MultiDiGraph:
    """
    Build a graph for the specified city using OSMnx.

    Parameters:
    city_name (str): The name of the city to build the graph for.

    Returns:
    nx.MultiDiGraph: A graph representing the city's road network.
    """

    G = ox.graph_from_place(city_name, network_type="drive")

    # Keeps the graphs largest strongly connected component.
    # for all nodes in the graph, there is a path from each node to every other node.
    G = ox.truncate.largest_component(G, strongly=True)

    G = ox.add_edge_speeds(G)  # impute edge speeds
    G = ox.add_edge_travel_times(G)  # impute edge travel times

    return G


def sample_nodes(graph: nx.MultiDiGraph, num_samples: int = 10) -> list:
    """
    Samples a specified number of nodes from the graph.

    Parameters:
    graph (nx.MultiDiGraph): The graph to sample nodes from.
    num_samples (int): The number of nodes to sample.

    Returns:
    list: A list of sampled node IDs.
    """
    nodes = list(graph.nodes())
    random_sample = random.sample(nodes, num_samples)
    return random_sample


def create_travel_time_matrix(graph: nx.MultiDiGraph, nodes: list) -> np.ndarray:
    """
    Creates a travel time matrix for the specified nodes in the graph.

    Parameters:
    graph (nx.MultiDiGraph): The graph to compute travel times from.
    nodes (list): A list of node IDs to include in the matrix.

    Returns:
    np.ndarray: A 2D array representing the travel time matrix.
    """
    num_nodes = len(nodes)
    travel_time_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                try:
                    travel_time = nx.shortest_path_length(
                        graph, source=nodes[i], target=nodes[j], weight="travel_time"
                    )
                    travel_time_matrix[i][j] = travel_time
                except nx.NetworkXNoPath:
                    travel_time_matrix[i][j] = float("inf")  # No path exists

    return travel_time_matrix


from tsplib95.models import StandardProblem


def save_problem_instance(
    travel_time_matrix: np.ndarray,
    node_coords: list,
    save_path: Path,
    seed: int,
    city: str = "Zurich, Switzerland",
):
    """Save problem in TSPLib format.

    Parameters:
    travel_time_matrix (np.ndarray): The travel time matrix to save.
    node_coords (list): A list of tuples containing the coordinates of the nodes.
    save_path (Path): The path to save the problem instance to.
    """

    problem = StandardProblem()
    problem.name = save_path.stem
    problem.type = "TSP"
    problem.comment = f"TSP instance with {len(node_coords)} nodes. In city: {city}, generated with seed: {seed}"
    problem.dimension = len(node_coords)
    problem.edge_weight_type = "EXPLICIT"
    problem.edge_weight_format = "FULL_MATRIX"
    problem.edge_weights = travel_time_matrix.tolist()
    problem.node_coords = {
        i + 1: (coord[0], coord[1]) for i, coord in enumerate(node_coords)
    }
    problem.save(save_path)


def main():
    arg_parser = argparse.ArgumentParser(description="Build TSP dataset for a city.")
    arg_parser.add_argument(
        "--city",
        type=str,
        default="Zurich, Switzerland",
        help="Name of the city to build the graph for.",
    )
    arg_parser.add_argument(
        "--repetitions",
        type=int,
        default=30,
        help="Number of repetitions for each sample size.",
    )
    arg_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            20000,
            50000,
            100000,
        ],
        help="List of sample sizes to generate.",
    )
    arg_parser.add_argument(
        "--out_dir",
        type=str,
        default="data/tsp_dataset",
        help="Directory to save the dataset.",
    )
    arg_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = arg_parser.parse_args()

    print(
        f"""Building TSP dataset for {args.city} with {args.repetitions} repetitions per sample size and sample sizes: {args.sizes}. 
            Saving to {args.out_dir}.
            Generating with random seed: {args.seed}"""
    )
    city = args.city
    repetitions = args.repetitions
    sizes = args.sizes
    output_dir = Path(args.out_dir)
    seed = args.seed

    city_basename = city.split(",")[0].strip().replace(" ", "_").lower()

    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    print("Building city graph...")
    G = build_city_graph(city)

    for size in sizes:
        save_dir = output_dir / str(size)
        save_dir.mkdir(parents=True, exist_ok=True)

        for n in tqdm(range(repetitions), desc=f"Size {size}"):
            # Create problem instance
            sampled_nodes = sample_nodes(G, num_samples=size)
            travel_time_matrix = create_travel_time_matrix(G, sampled_nodes)
            node_coords = [
                (G.nodes[node]["y"], G.nodes[node]["x"]) for node in sampled_nodes
            ]

            # Save Problem instance in TSPLib format
            save_problem_instance(
                travel_time_matrix,
                node_coords,
                save_dir / f"{city_basename}_{size}_{n}.tsp",
                seed=seed,
                city=city,
            )


if __name__ == "__main__":
    main()

# nodes_geodf = ox.graph_to_gdfs(G, nodes=True, edges=False)
# print(nodes_geodf.loc[sampled_nodes][["y", "x"]])
