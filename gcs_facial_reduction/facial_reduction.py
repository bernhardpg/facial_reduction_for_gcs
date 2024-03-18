from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import sympy
from matplotlib.axes import mpatches
from networkx.classes import DiGraph
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve


def simplify_graph_from_fr_result(zero_idxs: List[int], G: nx.Graph) -> nx.Graph:
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())

    G_new = nx.DiGraph()

    for node in G.nodes():
        G_new.add_node(node)

    for idx, (u, v) in enumerate(G.edges()):
        if idx not in zero_idxs:

            slack_var_tight = idx + num_edges in zero_idxs
            if slack_var_tight:
                G_new.add_edge(u, v, label="=1")
            else:
                G_new.add_edge(u, v)

    return G_new


def draw_graph(
    G: nx.Graph | nx.DiGraph,
    source: Optional[int] = None,
    target: Optional[int] = None,
    pos: Literal["random", "deterministic"] = "random",
) -> None:
    # Generate a fixed layout if not provided
    if pos == "deterministic":
        pos = nx.circular_layout(G)  # Example of using a circular layout
        # Alternatively, use any deterministic layout function or a custom layout dictionary
        # pos = {node: (node, len(G.nodes) - node) for node in G.nodes}  # Example of a custom layout
    else:  # random
        pos = nx.spring_layout(G)  # Calculate the layout for the nodes

    # Draw the graph using the specified/fixed positions
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=G.nodes(),
        node_size=500,
        node_color="lightblue",
    )

    if source is not None and source in G.nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[source],
            node_size=500,
            node_color="green",
        )

    if target is not None and target in G.nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[target],
            node_size=500,
            node_color="red",
        )

    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=16,
        font_weight="bold",
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Create a legend for the source and target nodes
    source_patch = mpatches.Patch(color="green", label="source")
    target_patch = mpatches.Patch(color="red", label="target")
    plt.legend(handles=[source_patch, target_patch])

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
    num_edges = len(list(G.edges()))
    num_vertices = len(list(G.nodes()))

    # Construct incidence matrix
    # Ax = b corresponds to incoming = outgoing
    # Note that this is different from the convention in
    # "Introduction to Linear Programming" where sum incoming + b = outgoing
    # (we have Ax = incoming - outgoing = b)
    A = nx.incidence_matrix(G, oriented=True).toarray()
    b = _construct_b(source, target, num_vertices)

    return A, b


# TODO(bernhardpg): Merge this with the function below
def graph_to_standard_form_with_flow_limits(
    A: npt.NDArray, b: npt.NDArray, G: nx.DiGraph, source: int, target: int
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

    # Add the constraint that sum incoming flows <= 1 for each vertex
    num_slacks = num_vertices
    A = np.hstack((A, np.zeros((A.shape[0], num_slacks))))
    edges = list(G.edges())
    for idx, vertex in enumerate(G.nodes()):
        incoming_edges = list(G.in_edges(vertex))
        constraint_row = np.zeros((1, A.shape[1]))
        # add the corresponding slack variable
        constraint_row[0, num_edges * 2 + idx] = 1
        b_row = 1
        for e in incoming_edges:
            e_idx = edges.index(e)
            constraint_row[0, e_idx] = 1

        if vertex == source:
            b_row -= 1

        A = np.vstack((A, constraint_row))
        b = np.concatenate((b, [b_row]))

    # Remove linearly independent rows in A
    sym_mat = sympy.Matrix(A.T)
    inds = list(sym_mat.rref()[1])

    A = A[inds, :]
    b = b[inds]

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


def run_facial_reduction(A: npt.NDArray, b: npt.NDArray) -> Tuple[bool, List[int]]:
    x_zero_idxs = []
    strictly_feasible = False
    while not strictly_feasible:
        strictly_feasible, x_zero_idxs = solve_facial_reduction_auxiliary_prob(
            A, b, x_zero_idxs
        )

    return strictly_feasible, x_zero_idxs


def run_fr_on_graph_and_visualize(
    G: nx.DiGraph,
    source: int,
    target: int,
    pos: Literal["random", "deterministic"] = "deterministic",
) -> None:
    draw_graph(G, source, target, pos)

    A, b = get_graph_description(G, source, target)
    A, b = graph_to_standard_form_with_flow_limits(A, b, G, source, target)
    strictly_feasible, x_zero_idxs = run_facial_reduction(A, b)
    assert strictly_feasible

    new_G = simplify_graph_from_fr_result(x_zero_idxs, G)
    draw_graph(new_G, source, target, pos)
