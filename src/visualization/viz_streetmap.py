import argparse

import matplotlib.pyplot as plt
import osmnx as ox
import tsplib95

from data_handling.tsplib_extension import TSPProblemWithOSMIDs


def plot_solution_on_streetmap(
    graph: ox.MultiDiGraph, tsp_problem_file: str, solution: list
):
    """
    Parameters:
    graph: The original graph of the city, loaded from the graphml file.
    problem: The TSP problem instance, containing the OSM IDs of the nodes and the path to the graphml file.
    solution: The solution to the TSP problem, a list of nodes of the tour (including the return to the start node).
    """

    G = ox.load_graphml(problem.graphml_file)
    problem = tsplib95.load(tsp_problem_file, problem_cls=TSPProblemWithOSMIDs)


def main():
    test_solution = [1, 2, 3, 4, 5, 7, 9, 8, 6, 1]  # Example solution (tour)
    G = ox.load_graphml()


if __name__ == "__main__":
    main()
