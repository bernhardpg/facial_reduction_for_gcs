import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
import sympy
from networkx.classes import DiGraph
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve

from gcs_facial_reduction.facial_reduction import (
    draw_graph,
    get_graph_description,
    graph_to_standard_form,
    run_facial_reduction,
    simplify_graph_from_fr_result,
    solve_facial_reduction_auxiliary_prob,
)

# Implements Example 4.1.1 from
# D. Drusvyatskiy and H. Wolkowicz, “The many faces of degeneracy in conic
# optimization.” arXiv, Jun. 12, 2017. doi: 10.48550/arXiv.1706.03705.
# TODO(bernhardpg): This can be turned into a unit test


def test_example_4_1_1() -> None:
    A = np.array([[1, 1, 1, 1, 0], [1, -1, -1, 0, 1]])
    b = np.array([1, -1])
    strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(A, b)
    strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(
        A, b, x_zero_idxs
    )

    assert strictly_feasible
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
    _, x_zero_idxs = solve_facial_reduction_auxiliary_prob(A, b)

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

    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form(A, b)
    strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(A, b)
    strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(
        A, b, x_zero_idxs
    )
    strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(
        A, b, x_zero_idxs
    )

    assert strictly_feasible

    # edges: (0, 1), (1, 0), (1, 2), (2, 1)
    assert x_zero_idxs == [1, 3, 4, 6]

    # new_G = simplify_graph_from_fr_result(x_zero_idxs, G)
    # draw_graph(new_G)


def test_fr_simple_3():
    G = nx.DiGraph()

    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    G.add_node(5)

    G.add_edge(0, 1)
    G.add_edge(1, 0)
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    G.add_edge(2, 3)
    G.add_edge(3, 2)
    G.add_edge(2, 4)
    G.add_edge(4, 3)
    # G.add_edge(3, 5)
    # G.add_edge(4, 5)
    # G.add_edge(5, 4)
    # G.add_edge(5, 3)

    source = 0
    target = 3

    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form(A, b)
    strictly_feasible, x_zero_idxs = run_facial_reduction(A, b)

    assert strictly_feasible

    # edges: (0, 1), (1, 0), (1, 2), (2, 1)
    # assert x_zero_idxs == [1, 3, 4, 6]

    # draw_graph(G, source, target)
    new_G = simplify_graph_from_fr_result(x_zero_idxs, G)
    draw_graph(new_G, source, target)


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
        strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(
            A, b, x_zero_idxs
        )

    # No facial reduction should be possible
    assert len(x_zero_idxs) == 0


@pytest.mark.skip
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
        strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(
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
