import random

import networkx as nx
import numpy as np
import osmnx as ox


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


CITY = "Zurich, Switzerland"
SIZES = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
REPETITIONS = 30


print("Building city graph...")
G = build_city_graph("Zurich, Switzerland")


print("Sampling nodes...")
sampled_nodes = sample_nodes(G, num_samples=10)

print("Creating travel time matrix...")
travel_time_matrix = create_travel_time_matrix(G, sampled_nodes)

print("Sampled Nodes:", sampled_nodes)

print("Travel Time Matrix:\n", travel_time_matrix)


# print coordinates of the sampled nodes

print("Coordinates of Sampled Nodes:")
for node in sampled_nodes:
    node_data = G.nodes[node]
    print(f"Node ID: {node}, Latitude: {node_data['y']}, Longitude: {node_data['x']}")


# nodes_geodf = ox.graph_to_gdfs(G, nodes=True, edges=False)
# print(nodes_geodf.loc[sampled_nodes][["y", "x"]])
