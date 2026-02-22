"""
Generates TSP problem instances for a specified city by sampling nodes from the city's road network graph and creating travel time matrices.
The generated problem instances are saved in TSPLib format for use in TSP solvers.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np

# Fix for NumPy 2.0 compatibility with older NetworkX/OSMnx GraphML writers
if not hasattr(np, "float_"):
    np.float_ = np.float64
import osmnx as ox
from joblib import Parallel, delayed
from tqdm import tqdm
from tsplib_extension import TSPProblemWithOSMIDs


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


def sample_nodes(graph: nx.MultiDiGraph, num_samples: int = 10) -> List[int]:
    """
    Samples a specified number of nodes from the graph.

    Parameters:
    graph (nx.MultiDiGraph): The graph to sample nodes from.
    num_samples (int): The number of nodes to sample.

    Returns:
    List[int]: A list of sampled node IDs.
    """
    nodes = list(graph.nodes())
    random_sample = random.sample(nodes, num_samples)
    return random_sample


def create_travel_time_matrix(graph: nx.MultiDiGraph, nodes: List[int]) -> np.ndarray:
    """
    Creates a travel time matrix for the specified nodes in the graph.

    Parameters:
    graph (nx.MultiDiGraph): The graph to compute travel times from.
    nodes (List[int]): A list of node IDs to include in the matrix.

    Returns:
    np.ndarray: A 2D array representing the travel time matrix.
    """
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    travel_time_matrix = np.full((num_nodes, num_nodes), fill_value=float("inf"))

    # Run Dijkstra from each sampled node to all other sampled nodes
    for i, source_node in enumerate(nodes):
        distances = nx.single_source_dijkstra_path_length(
            graph, source_node, weight="travel_time"
        )

        for target_node, distance in distances.items():
            if target_node in node_to_idx:
                j = node_to_idx[target_node]
                travel_time_matrix[i, j] = distance

    # Resulting graph is asymmetric, we need it symmetric
    symmetric_ttm = (travel_time_matrix + travel_time_matrix.T) / 2

    # Round time to nearest integer and convert to int
    return np.round(symmetric_ttm).astype(np.int32)


def save_problem_instance(
    travel_time_matrix: np.ndarray,
    node_coords: List[Tuple[float, float]],
    sampled_nodes: List[int],
    save_path: Path,
    seed: int,
    city: str,
    graphml_file: Path,
) -> None:
    """Save problem in TSPLib format.

    Parameters:
    travel_time_matrix (np.ndarray): The travel time matrix to save.
    node_coords (List[Tuple[float, float]]): A list of tuples containing the coordinates of the nodes.
    sampled_nodes (List[int]): The original OSM IDs of the nodes.
    save_path (Path): The path to save the problem instance to.
    seed (int): The seed used for generation.
    city (str): Name of the city.
    graphml_file (Path): The path to the corresponding graph file, for later reconstruction and visualization.
    """

    problem = TSPProblemWithOSMIDs()
    problem.name = save_path.stem
    problem.type = "TSP"

    # Save original IDs in the comment field for reconstruction/viz
    ids_str = " ".join(map(str, sampled_nodes))
    problem.comment = (
        f"TSP instance with {len(node_coords)} nodes. City: {city}, Seed: {seed}"
    )
    problem.osm_ids = sampled_nodes
    problem.graphml_file = str(graphml_file)
    problem.dimension = len(node_coords)
    problem.edge_weight_type = "EXPLICIT"
    problem.edge_weight_format = "FULL_MATRIX"
    problem.edge_weights = travel_time_matrix.tolist()
    problem.node_coords = {
        i + 1: (coord[0], coord[1]) for i, coord in enumerate(node_coords)
    }
    problem.save(save_path)


def process_single_instance(
    G: nx.MultiDiGraph,
    size: int,
    n: int,
    save_dir: Path,
    city: str,
    city_basename: str,
    graphml_file: Path,
    seed: int,
) -> None:
    """
    Generates a single TSP problem instance for the specified graph, sample size, and repetition number, and saves it in TSPLib format.
    """
    np.random.seed(seed + n)
    random.seed(seed + n)

    # if restarting, skip if already exists
    filepath = save_dir / f"{city_basename}_{size}_{n}.tsp"

    if filepath.exists():
        return

    # Create problem instance
    sampled_nodes = sample_nodes(G, num_samples=size)
    travel_time_matrix = create_travel_time_matrix(G, sampled_nodes)
    node_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in sampled_nodes]

    # Save Problem instance in TSPLib format
    save_problem_instance(
        travel_time_matrix=travel_time_matrix,
        node_coords=node_coords,
        sampled_nodes=sampled_nodes,
        save_path=filepath,
        seed=seed + n,
        city=city,
        graphml_file=graphml_file,
    )


def start_problem_generation(
    city: str,
    city_basename: str,
    output_dir: Path,
    repetitions: int,
    seed: int,
    sizes: List[int],
) -> None:
    """
    Starts the process of building TSP problem instances for the specified city and sample sizes.
    Runs in parallel
    """
    print("Building city graph...")
    graph = build_city_graph(city)

    graphml_file = output_dir / f"graph_{city_basename}.graphml"
    print(
        f"Saving graph of {city} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {graphml_file}..."
    )
    ox.save_graphml(graph, filepath=graphml_file)

    for size in sizes:
        save_dir = output_dir / str(size)
        save_dir.mkdir(parents=True, exist_ok=True)

        Parallel(n_jobs=-1)(
            delayed(process_single_instance)(
                graph, size, n, save_dir, city, city_basename, graphml_file, seed
            )
            for n in tqdm(range(repetitions), desc=f"Size {size}")
        )


def main() -> None:
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
    arg_parser.add_argument(
        "--clean_build",
        action="store_true",
        help="Whether to clean the output directory before building the dataset.",
    )

    args = arg_parser.parse_args()

    print(
        f"""Building TSP dataset for {args.city} with {args.repetitions} repetitions per sample size and sample sizes: {args.sizes}. 
            Saving to {args.out_dir}.
            Generating with random seed: {args.seed}"""
    )

    output_dir = Path(args.out_dir)

    if args.clean_build and output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    city_basename = args.city.split(",")[0].strip().replace(" ", "_").lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_problem_generation(
        args.city, city_basename, output_dir, args.repetitions, args.seed, args.sizes
    )


if __name__ == "__main__":
    main()
