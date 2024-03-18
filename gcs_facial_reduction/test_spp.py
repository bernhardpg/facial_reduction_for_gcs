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


def _solve_facial_reduction_auxiliary_prob(
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


# Implements Example 4.1.1 from
# D. Drusvyatskiy and H. Wolkowicz, “The many faces of degeneracy in conic
# optimization.” arXiv, Jun. 12, 2017. doi: 10.48550/arXiv.1706.03705.
# TODO(bernhardpg): This can be turned into a unit test
def test_example_4_1_1() -> None:
    A = np.array([[1, 1, 1, 1, 0], [1, -1, -1, 0, 1]])
    b = np.array([1, -1])
    x_zero_idxs = _solve_facial_reduction_auxiliary_prob(A, b)
    assert x_zero_idxs == [0, 3, 4]


def test_fr_simple_1():
    G = nx.DiGraph()

    G.add_edge(0, 1)
    G.add_edge(1, 0)

    source = 0
    target = 1

    # draw_graph(G)
    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form(A, b)
    _, x_zero_idxs = _solve_facial_reduction_auxiliary_prob(A, b)

    assert x_zero_idxs == [1, 2]  # we want f_21 = 0 and s_12 = 0 i.e. f_12 = 1


def test_fr_simple_2():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)
    G.add_node(2)

    G.add_edge(0, 1)
    G.add_edge(1, 0)
    G.add_edge(1, 2)
    G.add_edge(2, 1)

    source = 0
    target = 2

    # draw_graph(G)
    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form(A, b)
    strictly_feasible, x_zero_idxs = _solve_facial_reduction_auxiliary_prob(A, b)
    strictly_feasible, x_zero_idxs = _solve_facial_reduction_auxiliary_prob(
        A, b, x_zero_idxs
    )
    strictly_feasible, x_zero_idxs = _solve_facial_reduction_auxiliary_prob(
        A, b, x_zero_idxs
    )

    assert strictly_feasible

    # edges: (0, 1), (1, 0), (1, 2), (2, 1)
    assert x_zero_idxs == [1, 3, 4, 6]


def test_fr_flow_split():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_edge(0, 1)
    G.add_edge(1, 3)
    G.add_edge(0, 2)
    G.add_edge(2, 3)

    source = 0
    target = 3

    # draw_graph(G)
    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form(A, b)
    x_zero_idxs = []
    strictly_feasible = False
    while not strictly_feasible:
        strictly_feasible, x_zero_idxs = _solve_facial_reduction_auxiliary_prob(
            A, b, x_zero_idxs
        )

    # No facial reduction should be possible
    assert len(x_zero_idxs) == 0


def test_fr_flow_split_bidirectional():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_edge(0, 1)
    G.add_edge(1, 3)
    G.add_edge(0, 2)
    G.add_edge(2, 3)

    G.add_edge(1, 0)
    G.add_edge(3, 1)
    G.add_edge(2, 0)
    G.add_edge(3, 2)

    source = 0
    target = 3

    # draw_graph(G)
    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form(A, b)
    x_zero_idxs = []
    strictly_feasible = False
    while not strictly_feasible:
        strictly_feasible, x_zero_idxs = _solve_facial_reduction_auxiliary_prob(
            A, b, x_zero_idxs
        )

    # (0, 1), (0, 2), (1, 3), (1, 0), (2, 3), (2, 0), (3, 1), (3, 2)
    f_feasible = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    assert np.all(A @ f_feasible == b)

    # It should not be possible to push more than one flow through each edge
    f_infeasible = np.array([2, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    assert not np.all(A @ f_infeasible == b)

    breakpoint()
    # TODO(bernhardpg): Should it not be possible to remove more edges here??
    assert len(x_zero_idxs) > 0


def test_get_graph_description_simple():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)

    G.add_edge(0, 1)
    G.add_edge(1, 0)

    A, b = get_graph_description(G, source=0, target=1)
    assert A.shape == (2, 2)
    assert b.shape == (2,)

    assert b[0] == -1
    assert b[1] == 1

    f_feasible = np.array([1, 0])

    assert np.all(A @ f_feasible == b)

    f_infeasible = np.array([0, 1])
    assert not np.all(A @ f_infeasible == b)


def test_get_graph_description_split():
    G = nx.DiGraph()

    # TODO(bernhardpg): NetworkX changes the
    # ordering of the nodes unless you add them like
    # this. This messes with my code!
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)

    source = 0
    target = 3

    A, b = get_graph_description(G, source=source, target=target)
    # edges: (0, 1), (0, 2), (1, 3), (2, 3)
    # (after NetworkX reorders them)
    f_feasible_1 = np.array([1, 0, 1, 0])
    assert np.all(A @ f_feasible_1 == b)

    f_feasible_2 = np.array([0, 1, 0, 1])
    assert np.all(A @ f_feasible_2 == b)

    f_infeasible_1 = np.array([1, 0, 0, 1])
    assert not np.all(A @ f_infeasible_1 == b)

    f_infeasible_2 = np.array([0, 1, 1, 0])
    assert not np.all(A @ f_infeasible_2 == b)


def test_graph_to_standard_form_simple():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)

    G.add_edge(0, 1)
    G.add_edge(1, 0)

    A, b = get_graph_description(G, source=0, target=1)
    N, m = A.shape

    A, b = graph_to_standard_form(A, b)

    assert A.shape == (2 * N - 1, 2 * m)
    assert b.shape == (2 * N - 1,)

    f_feasible = np.array([1, 0, 0, 1])

    assert np.all(A @ f_feasible == b)

    f_infeasible = np.array([0, 1, 1, 0])
    assert not np.all(A @ f_infeasible == b)


def test_graph_to_standard_form_split():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    G.add_edge(0, 1)
    G.add_edge(1, 3)
    G.add_edge(0, 2)
    G.add_edge(2, 3)

    source = 0
    target = 3

    A, b = get_graph_description(G, source=source, target=target)
    # edges: (0, 1), (0, 2), (1, 3), (2, 3)
    # (after NetworkX reorders them)

    A, b = graph_to_standard_form(A, b)

    f_feasible_1 = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    assert np.all(A @ f_feasible_1 == b)

    f_feasible_2 = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    assert np.all(A @ f_feasible_2 == b)

    f_infeasible_1 = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert not np.all(A @ f_infeasible_1 == b)

    f_infeasible_2 = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    assert not np.all(A @ f_infeasible_2 == b)

    f_infeasible_3 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert not np.all(A @ f_infeasible_3 == b)

    f_infeasible_4 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert not np.all(A @ f_infeasible_4 == b)


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
test_fr_flow_split_bidirectional()
