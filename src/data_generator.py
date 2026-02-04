from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DOMAIN_SIZE = np.array([1, 1])


def plot_nodes(node_coords: np.ndarray):
    plt.title("Nodes")
    plt.xlim(0, DOMAIN_SIZE[0])
    plt.ylim(0, DOMAIN_SIZE[1])
    plt.scatter(x=node_coords[:, 0], y=node_coords[:, 1], label="Nodes")
    plt.legend()
    plt.show()


def generate_nodes(n_nodes: int = 10):
    """
    Generate List of nodes
    """

    node_coords = np.zeros((n_nodes, 2))
    node_coords[:, 0] = np.random.uniform(0, DOMAIN_SIZE[0], size=(n_nodes,))
    node_coords[:, 1] = np.random.uniform(0, DOMAIN_SIZE[1], size=(n_nodes,))
    # print(node_coords)
    # plot_nodes(node_coords)
    return node_coords


def generate_edges(node_coords: np.ndarray):
    """
    Returns the Adjacency Matrix of size (n_nodes x n_nodes) where the values is the distance between the nodes.
    """
    n_nodes = len(node_coords)
    distance_matrix = np.zeros((n_nodes, n_nodes))

    max_dist = np.sqrt(np.sum(DOMAIN_SIZE**2))

    # Get  distances
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_ij = np.linalg.norm(node_coords[i] - node_coords[j])
            dist_norm_ij = dist_ij / max_dist
            # print(dist_norm_ij)
            # edge exists not always
            exists = np.random.binomial(n=1, p=dist_norm_ij)
            if exists:
                distance_matrix[i, j] = dist_ij
            else:
                distance_matrix[i, j] = 0

    return distance_matrix

    # Compute if edge exists


def plot_graph(adj: np.ndarray):
    G = nx.from_numpy_array(adj)
    nx.draw(
        G,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=16,
    )
    plt.title("Network Graph for TSP")
    plt.show()


def main():
    nodes = generate_nodes(10)
    adj = generate_edges(nodes)
    print(adj)
    plot_graph(adj)


if __name__ == "__main__":
    main()
