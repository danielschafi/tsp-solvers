"""
The code for setting up the solver was taken from the TSP example on gurobis website.
The GurobiSolver class just is a wrapper around that.
Below is their copyright notice.

Code copied:
  - shortest_subtour
  - TSPCallback
  - solve_tsp

"""

# Copyright 2026, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of points
# using lazy constraints.  The base MIP model only includes 'degree-2'
# constraints, requiring each node to have exactly two incident edges.
# Solutions to this model may contain subtours - tours that don't visit every
# city.  The lazy constraint callback adds new constraints to cut them off.

import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import gurobipy as gp
import numpy as np
import tsplib95
from gurobipy import GRB

from src.solvers.solver_base import TSPSolver


class GurobiSolver(TSPSolver):
    """
    Solver for TSP using gurobi.
    """

    def __init__(self):
        super().__init__(solver="gurobi")

    def setup_problem(self, tsp_file: str):
        """
        Prepares the data for the cuOpt solver.
        Builds the adjacendy matrix,
        """
        self.load_tsp_file(tsp_file)
        self.edges = np.array(self.problem.edge_weights)
        points = np.array(self.problem.node_locations)
        # Gurobi
        self.node_idx_list = list(range(self.problem.dimension))

        # Gurobi requires a dictionary instead
        # This is for the symmetric case
        self.distances = {
            (i, j): self.edges[i, j] for i, j in combinations(self.node_idx_list, 2)
        }

        self.nodes = np.array(points)

    def solve_tsp(self):
        """
        Solve a dense symmetric TSP using the following base formulation:

        min  sum_ij d_ij x_ij
        s.t. sum_j x_ij == 2   forall i in V
            x_ij binary       forall (i,j) in E

        and subtours eliminated using lazy constraints.
        """

        with gp.Env() as env, gp.Model(env=env) as m:
            # Create variables, and add symmetric keys to the resulting dictionary
            # 'x', such that (i, j) and (j, i) refer to the same variable.
            x = m.addVars(
                self.distances.keys(), obj=self.distances, vtype=GRB.BINARY, name="e"
            )
            x.update({(j, i): v for (i, j), v in x.items()})

            # Create degree 2 constraints
            # Each node only of degree 2 -> Visited only once
            for i in self.node_idx_list:
                m.addConstr(
                    gp.quicksum(x[i, j] for j in self.node_idx_list if i != j) == 2
                )

            # Optimize model using lazy constraints to eliminate subtours
            m.Params.LazyConstraints = 1
            cb = TSPCallback(self.node_idx_list, x)
            m.optimize(cb)

            # Extract the solution as a tour
            edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
            tour = GurobiSolver.shortest_subtour(edges)
            assert set(tour) == set(self.node_idx_list)

            # We assume that the return to the start node is included in the tour
            if tour[0] != tour[-1]:
                tour.append(tour[0])
            self.result["tour"] = tour
            self.result["cost"] = self.calculate_tour_cost(tour)
            self.result["time_to_solve"] = m.Runtime

            # Status codes: https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html#secstatuscodes
            # Check if any solution exists before accessing .X or .ObjVal
            has_solution = m.SolCount > 0

            # Determine the status message
            if m.Status == GRB.OPTIMAL:
                status_msg = "OPTIMAL"
            elif m.Status == GRB.TIME_LIMIT:
                status_msg = "TIME_LIMIT"
            elif m.Status == GRB.INFEASIBLE:
                status_msg = "INFEASIBLE"
            elif m.Status == GRB.INTERRUPTED:
                status_msg = "USER_INTERRUPTED"
            else:
                status_msg = f"OTHER_{m.Status}"

            self.result["solution_status"] = status_msg

            self.result["additional_metadata"] = {
                "status_code": m.Status,
                "node_count": m.NodeCount,  # Number of branch-and-cut nodes explored
                "runtime": m.Runtime,  # Total solve time in seconds
                "sol_count": m.SolCount,  # Number of feasible solutions found
                "work": m.Work,  # Work units (deterministic measure of effort)
            }

            # Only add these if a solution or bound actually exists
            if has_solution:
                self.result["additional_metadata"].update(
                    {
                        "lower_bound": m.ObjBound,  # Best possible theoretical cost
                        "gap": m.MIPGap,  # Relative gap between ObjBound and ObjVal
                        "obj_val": m.ObjVal,  # The cost of the best found tour
                    }
                )

            print(
                f"Solve complete. Status: {status_msg}, Gap: {self.result['additional_metadata'].get('gap', 'N/A')}"
            )
            return tour, (m.ObjVal if has_solution else None)

    @classmethod
    def shortest_subtour(self, edges):
        """Given a list of edges, return the shortest subtour (as a list of nodes)
        found by following those edges. It is assumed there is exactly one 'in'
        edge and one 'out' edge for every node represented in the edge list."""

        # Create a mapping from each node to its neighbours
        node_neighbors = defaultdict(list)
        for i, j in edges:
            node_neighbors[i].append(j)
        assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

        # Follow edges to find cycles. Each time a new cycle is found, keep track
        # of the shortest cycle found so far and restart from an unvisited node.
        unvisited = set(node_neighbors)
        shortest = None
        while unvisited:
            cycle = []
            neighbors = list(unvisited)
            while neighbors:
                current = neighbors.pop()
                cycle.append(current)
                unvisited.remove(current)
                neighbors = [j for j in node_neighbors[current] if j in unvisited]
            if shortest is None or len(cycle) < len(shortest):
                shortest = cycle

        assert shortest is not None
        return shortest


class TSPCallback:
    """Callback class implementing lazy constraints for the TSP.  At MIPSOL
    callbacks, solutions are checked for subtours and subtour elimination
    constraints are added if needed."""

    def __init__(self, nodes, x):
        self.nodes = nodes
        self.x = x

    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found. Stop the optimization if there is an exception in
        user code."""
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    def eliminate_subtours(self, model):
        """Extract the current solution, check for subtours, and formulate lazy
        constraints to cut off the current solution if subtours are found.
        Assumes we are at MIPSOL."""
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        tour = GurobiSolver.shortest_subtour(edges)
        if len(tour) < len(self.nodes):
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(
                gp.quicksum(self.x[i, j] for i, j in combinations(tour, 2))
                <= len(tour) - 1
            )


def main():
    solver = GurobiSolver()
    solver.run(
        "/home/schafhdaniel@edu.local/thesis/tsp-solvers/data/tsp_dataset/10_conv/zurich_10_0.tsp"
    )


if __name__ == "__main__":
    main()
