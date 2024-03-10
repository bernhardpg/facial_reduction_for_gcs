from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve


def draw_graph(G: nx.Graph | nx.DiGraph) -> None:
    # Draw the graph
    pos = nx.spring_layout(G)  # Calculate the layout for the nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=16,
        font_weight="bold",
    )

    # Show the plot
    plt.axis("off")
    plt.show()


def draw_path_in_graph(G: nx.Graph | nx.DiGraph, path: List[Tuple[int, int]]) -> None:
    # Set edge colors
    edge_colors = ["black" if edge not in path else "red" for edge in G.edges()]

    # Set edge widths
    edge_widths = [1 if edge not in path else 3 for edge in G.edges()]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths
    )
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight="bold")

    # Show the plot
    plt.axis("off")
    plt.show()


# Create a graph
G = nx.DiGraph()

# Add edges to the graph
G.add_edge(0, 1)
G.add_edge(1, 4)
G.add_edge(4, 3)
G.add_edge(0, 2)
G.add_edge(2, 3)

source = 0
target = 3


def _construct_b(source: int, target: int, N: int) -> npt.NDArray:
    b = np.zeros((N,))
    b[source] = 1
    b[target] = -1
    return b


# draw_graph(G)

vertices = list(G.nodes())
edges = list(G.edges())
N = len(vertices)

A = nx.incidence_matrix(G, oriented=True).toarray()
b = _construct_b(source, target, N)

prog = MathematicalProgram()
f = prog.NewContinuousVariables(N, "f")

prog.AddLinearConstraint(ge(f, 0))
prog.AddLinearConstraint(le(f, 1))

cost = prog.AddLinearCost(np.sum(f))  # equal cost for all edges
flow_constraint = prog.AddLinearConstraint(eq(A @ f + b, 0))

result = Solve(prog)
assert result.is_success()


f_sols = result.GetSolution(f)
edge_idxs = np.where(np.isclose(f_sols, 1))[0].tolist()
path = [edges[idx] for idx in edge_idxs]

draw_path_in_graph(G, path)
