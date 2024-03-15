from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import sympy
from networkx.classes import DiGraph
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


def _construct_b(source: int, target: int, N: int) -> npt.NDArray:
    b = np.zeros((N,))
    b[source] = 1
    b[target] = -1
    return b


def _formulate_spp_problem(
    G: nx.DiGraph | nx.Graph, source: int, target: int
) -> List[Tuple[int, int]]:
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

    return path


def e_i(i: int, dims: int) -> npt.NDArray:
    """
    Return the i-th unit vector.
    """
    e = np.zeros((dims,))
    e[i] = 1
    return e


def _solve_facial_reduction_auxiliary_prob(A: npt.NDArray, b: npt.NDArray):

    m, N = A.shape
    prog = MathematicalProgram()
    y = prog.NewContinuousVariables(m, "y")

    # pick an x in the relative interior of positive orthant
    x_hat = np.ones((N,))

    prog.AddLinearConstraint(ge(A.T @ y, 0))
    prog.AddLinearConstraint(b.T @ y == 0)
    prog.AddLinearConstraint(x_hat.T @ A.T @ y == 1)

    result = Solve(prog)

    if result.is_success():
        y_sol = result.GetSolution(y)
        z = A.T @ y_sol
        # x must be zero where z is nonzero
        x_zero_idxs = np.where(~np.isclose(z, 0))[0].tolist()
        return x_zero_idxs
    else:
        breakpoint()
        raise NotImplementedError("Not yet implemented")
        breakpoint()


def graph_to_standard_form(
    G: nx.DiGraph, source: int, target: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    vertices = list(G.nodes())
    edges = list(G.edges())

    num_edges = len(edges)
    num_vertices = len(vertices)

    # Construct incidence matrix
    A = nx.incidence_matrix(G, oriented=True).toarray()
    b = _construct_b(source, target, num_vertices)

    # Add slack variables for each flow fᵤᵥ
    # to encode 0 ≤ fᵤᵥ ≤ 1 in standard form

    # Pad A with extra columns for all the slack variables we need to add
    A = np.hstack((A, np.zeros((num_vertices, num_edges))))
    for e_idx in range(num_edges):
        row = np.zeros((1, num_edges * 2))
        row[0, e_idx] = 1
        row[0, num_edges + e_idx] = 1
        A = np.vstack((A, row))
        b = np.concatenate((b, [1]))

    # Remove linearly independent rows in A
    sym_mat = sympy.Matrix(A.T)
    inds = list(sym_mat.rref()[1])

    A = A[inds, :]
    b = b[inds]

    return A, b


# Implements Example 4.1.1 from
# D. Drusvyatskiy and H. Wolkowicz, “The many faces of degeneracy in conic
# optimization.” arXiv, Jun. 12, 2017. doi: 10.48550/arXiv.1706.03705.
# TODO(bernhardpg): This can be turned into a unit test
def test_example_4_1_1() -> None:
    A = np.array([[1, 1, 1, 1, 0], [1, -1, -1, 0, 1]])
    b = np.array([1, -1])
    x_zero_idxs = _solve_facial_reduction_auxiliary_prob(A, b)
    assert x_zero_idxs == [0, 3, 4]


def test_spp_simple():
    # Create a graph
    G = nx.DiGraph()

    # Add edges to the graph
    G.add_edge(0, 1)
    G.add_edge(1, 0)

    source = 0
    target = 1

    # draw_graph(G)
    A, b = graph_to_standard_form(G, source, target)
    # path = _formulate_spp_problem(G, source, target)
    x_zero_idxs = _solve_facial_reduction_auxiliary_prob(A, b)

    breakpoint()

    draw_path_in_graph(G, path)


# test_example_4_1_1()
test_spp_simple()
