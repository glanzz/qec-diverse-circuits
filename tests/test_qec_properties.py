import numpy as np
import networkx as nx
import pytest
from qec_diverse_circuits.qec_properties import compute_qec_properties, CLIFFORD_GATES


class _FakeNode:
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or []


def _build_fake_dag(gate_chain_per_qubit):
    dag = nx.DiGraph()
    node_id = 0
    for gates in gate_chain_per_qubit:
        for name in gates:
            dag.add_node(node_id, node_data=_FakeNode(name))
            node_id += 1
    return dag


def _build_entangling_fake_dag():
    dag = nx.DiGraph()
    dag.add_node(0, node_data=_FakeNode("h"))
    dag.add_node(1, node_data=_FakeNode("h"))
    dag.add_node(2, node_data=_FakeNode("cx"))
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    return dag


def test_compute_qec_properties_returns_fixed_length_vector():
    dag = _build_fake_dag([["h", "x"]])
    vec = compute_qec_properties(dag, num_qubits=1)
    assert vec.shape == (7,)
    assert vec.dtype == np.float32


def test_pure_clifford_circuit_has_zero_non_clifford_ratio():
    dag = _build_fake_dag([["h", "x", "cx".lower()]])
    vec = compute_qec_properties(dag, num_qubits=1)
    assert vec[0] == pytest.approx(0.0)


def test_non_clifford_gate_raises_non_clifford_ratio():
    dag = _build_fake_dag([["h", "t", "x"]])
    vec = compute_qec_properties(dag, num_qubits=1)
    assert vec[0] == pytest.approx(1.0 / 3.0)


def test_t_gate_ratio_counts_t_and_tdg_only():
    dag = _build_fake_dag([["t", "tdg", "h", "x"]])
    vec = compute_qec_properties(dag, num_qubits=1)
    assert vec[1] == pytest.approx(2.0 / 4.0)


def test_empty_circuit_does_not_raise_and_reports_zero_properties():
    dag = nx.DiGraph()
    vec = compute_qec_properties(dag, num_qubits=3)
    assert np.all(np.isfinite(vec))
    assert vec[0] == pytest.approx(0.0)
    assert vec[1] == pytest.approx(0.0)
    assert vec[2] == pytest.approx(0.0)


def test_single_qubit_circuit_has_zero_connectivity_not_divide_by_zero():
    dag = _build_fake_dag([["h", "x"]])
    vec = compute_qec_properties(dag, num_qubits=1)
    assert vec[4] == pytest.approx(0.0)


def test_entangling_gate_raises_entangling_density_and_connectivity():
    dag = _build_entangling_fake_dag()
    vec = compute_qec_properties(dag, num_qubits=2)
    assert vec[2] == pytest.approx(1.0 / 3.0)
    assert vec[4] == pytest.approx(1.0)


def test_entangling_type_diversity_ratio_counts_distinct_entangling_gate_names():
    dag = nx.DiGraph()
    dag.add_node(0, node_data=_FakeNode("h"))
    dag.add_node(1, node_data=_FakeNode("h"))
    dag.add_node(2, node_data=_FakeNode("cx"))
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    dag.add_node(3, node_data=_FakeNode("cz"))
    dag.add_edge(2, 3)
    dag.add_edge(1, 3)
    vec = compute_qec_properties(dag, num_qubits=2)
    assert vec[3] == pytest.approx(1.0)


def test_depth_matches_longest_path_length():
    dag = _build_entangling_fake_dag()
    vec = compute_qec_properties(dag, num_qubits=2)
    assert vec[5] == pytest.approx(nx.dag_longest_path_length(dag))


def test_idle_fraction_is_clamped_to_unit_interval():
    dag = _build_entangling_fake_dag()
    vec = compute_qec_properties(dag, num_qubits=2)
    assert 0.0 <= vec[6] <= 1.0


def test_clifford_gates_constant_matches_validated_diagnostic_set():
    assert CLIFFORD_GATES == frozenset(
        {"id", "x", "y", "z", "h", "s", "sdg", "cx", "cy", "cz", "swap", "sx", "sxdg"}
    )


def test_handles_qubit_placeholder_nodes_with_no_node_data():
    dag = nx.DiGraph()
    dag.add_node(3, node_data=_FakeNode("h"), label="h")
    dag.add_node(4, node_data=_FakeNode("cx"), label="cx")
    dag.add_edge(1, 3)
    dag.add_edge(2, 4)
    dag.add_edge(3, 4)

    assert dag.nodes[1] == {}
    assert dag.nodes[2] == {}

    vec = compute_qec_properties(dag, num_qubits=2)
    assert np.all(np.isfinite(vec))
    assert vec[0] == pytest.approx(0.0)
