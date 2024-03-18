from typing import List, Optional, Tuple

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
    b[source] = -1
    b[target] = 1
    return b


def _formulate_fr_problem(
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


def solve_facial_reduction_auxiliary_prob(
    A: npt.NDArray, b: npt.NDArray, zero_idxs: Optional[List[int]] = None
) -> Tuple[bool, List[int]]:
    m, N = A.shape
    prog = MathematicalProgram()
    y = prog.NewContinuousVariables(m, "y")
    s = prog.NewContinuousVariables(N, "s")

    if zero_idxs is None:
        zero_idxs = []

    # pick an x in the relative interior of positive orthant
    x_hat = np.ones((N,))

    for idx in zero_idxs:
        x_hat[idx] = 0

    non_zero_idxs = [i for i in range(N) if i not in zero_idxs]
    for idx in non_zero_idxs:
        prog.AddLinearConstraint(s[idx] >= 0)

    prog.AddLinearConstraint(eq(s, A.T @ y))
    prog.AddLinearConstraint(b.T @ y == 0)
    prog.AddLinearConstraint(x_hat.T @ s == 1)

    result = Solve(prog)

    if result.is_success():
        y_sol = result.GetSolution(y)
        z = A.T @ y_sol
        # x must be zero where z is nonzero
        x_zero_idxs = np.where(~np.isclose(z, 0))[0].tolist()

        if zero_idxs is not None:
            return False, list(set(x_zero_idxs + zero_idxs))
        else:
            return False, x_zero_idxs
    else:
        # The problem must be strictly feasible
        return True, zero_idxs


def get_graph_description(
    G: nx.DiGraph, source: int, target: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    vertices = list(G.nodes())
    edges = list(G.edges())

    num_edges = len(edges)
    num_vertices = len(vertices)

    # Construct incidence matrix
    # Ax = b corresponds to incoming = outgoing
    # Note that this is different from the convention in
    # "Introduction to Linear Programming" where sum incoming + b = outgoing
    # (we have Ax = incoming - outgoing = b)
    A = nx.incidence_matrix(G, oriented=True).toarray()
    b = _construct_b(source, target, num_vertices)

    return A, b


def graph_to_standard_form(
    A: npt.NDArray, b: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    num_vertices, num_edges = A.shape
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


# path = _formulate_fr_problem(G, source, target)
# draw_path_in_graph(G, path)
# test_example_4_1_1()
# test_fr_simple()
# test_fr_simple_2()
# test_get_graph_description_simple()
# test_get_graph_description_split()
# test_graph_to_standard_form_simple()
# test_graph_to_standard_form_split()
# test_fr_flow_split()
# test_fr_flow_split_bidirectional()
