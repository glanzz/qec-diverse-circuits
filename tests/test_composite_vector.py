import numpy as np
import networkx as nx
import pytest
from qec_diverse_circuits.environment import build_composite_vector
from qec_diverse_circuits.constants import (
    COMPOSITE_FEATURES_SIZE,
    NETLSD_FEATURES_SIZE,
    GATE_HISTOGRAM_FEATURES_SIZE,
    QEC_PROPERTIES_FEATURES_SIZE,
    QEC_PROPERTY_WEIGHTS,
)


class _FakeNode:
    def __init__(self, name, params):
        self.name = name
        self.params = params


def _build_fake_dag(gate_names_with_params):
    dag = nx.DiGraph()
    for i, (name, params) in enumerate(gate_names_with_params):
        dag.add_node(i, node_data=_FakeNode(name, params), features={"params": params})
    return dag


def test_composite_vector_has_expected_length():
    dag = _build_fake_dag([("h", []), ("cx", [])])
    vec = build_composite_vector(dag, max_gate_index=49, num_qubits=2)
    assert vec.shape == (COMPOSITE_FEATURES_SIZE,)


def test_composite_vector_distinguishes_gate_type_on_identical_topology():
    dag_h = _build_fake_dag([("h", []), ("h", [])])
    dag_x = _build_fake_dag([("x", []), ("x", [])])

    vec_h = build_composite_vector(dag_h, max_gate_index=49, num_qubits=1)
    vec_x = build_composite_vector(dag_x, max_gate_index=49, num_qubits=1)

    assert not np.allclose(vec_h, vec_x)


def test_composite_vector_gate_histogram_counts_usage():
    dag = _build_fake_dag([("h", []), ("h", []), ("cx", [])])
    vec = build_composite_vector(dag, max_gate_index=3, num_qubits=2)
    histogram_section = vec[NETLSD_FEATURES_SIZE: NETLSD_FEATURES_SIZE + GATE_HISTOGRAM_FEATURES_SIZE]
    assert histogram_section.sum() > 0


def test_composite_vector_passes_undirected_graph_to_netlsd():
    import unittest.mock as mock

    dag = _build_fake_dag([("h", []), ("cx", [])])

    with mock.patch("qec_diverse_circuits.environment.netlsd.heat") as mock_heat:
        mock_heat.return_value = np.zeros(NETLSD_FEATURES_SIZE)
        build_composite_vector(dag, max_gate_index=49, num_qubits=2)

    assert mock_heat.call_count == 1
    graph_passed = mock_heat.call_args[0][0]
    assert not graph_passed.is_directed()


def test_composite_vector_includes_weighted_qec_property_block():
    dag = _build_fake_dag([("t", []), ("t", []), ("h", [])])
    vec = build_composite_vector(dag, max_gate_index=49, num_qubits=1)

    qec_block = vec[-QEC_PROPERTIES_FEATURES_SIZE:]
    expected_non_clifford_ratio = 2.0 / 3.0
    assert qec_block[0] == pytest.approx(expected_non_clifford_ratio * QEC_PROPERTY_WEIGHTS[0], abs=1e-5)


def test_composite_vector_qec_block_weighting_is_applied():
    dag = _build_fake_dag([("t", []), ("h", [])])
    vec = build_composite_vector(dag, max_gate_index=49, num_qubits=1)
    qec_block = vec[-QEC_PROPERTIES_FEATURES_SIZE:]

    from qec_diverse_circuits.qec_properties import compute_qec_properties

    raw_dag = nx.DiGraph()
    raw_dag.add_node(0, node_data=_FakeNode("t", []))
    raw_dag.add_node(1, node_data=_FakeNode("h", []))
    raw_properties = compute_qec_properties(raw_dag, num_qubits=1)

    np.testing.assert_allclose(
        qec_block, raw_properties * np.array(QEC_PROPERTY_WEIGHTS, dtype=np.float32), atol=1e-5
    )


def test_composite_vector_length_matches_declared_block_sizes():
    dag = _build_fake_dag([("h", []), ("cx", [])])
    vec = build_composite_vector(dag, max_gate_index=49, num_qubits=2)
    assert vec.shape == (COMPOSITE_FEATURES_SIZE,)
    assert COMPOSITE_FEATURES_SIZE == NETLSD_FEATURES_SIZE + GATE_HISTOGRAM_FEATURES_SIZE + 2 + QEC_PROPERTIES_FEATURES_SIZE
