import math
import pytest
from qec_diverse_circuits.gate_pool import QiskitGatePool


class _FakeGate:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_operation_rejects_zero_param_without_producing_inf():
    params = {"theta": lambda: None}
    action = {"params": [0, 5, 3, 7]}

    gate = QiskitGatePool.get_operation(params, action, _FakeGate)

    theta = gate.kwargs["theta"]
    assert math.isfinite(theta), f"expected finite theta, got {theta}"


def test_get_operation_nonzero_param_matches_pi_over_n_plus_one():
    params = {"theta": lambda: None}
    action = {"params": [3, 0, 0, 0]}

    gate = QiskitGatePool.get_operation(params, action, _FakeGate)

    assert math.isclose(gate.kwargs["theta"], math.pi / 4)


def test_get_operation_fixed_param_passes_through_unchanged():
    params = {"num_qubits": 3}
    action = {"params": []}

    gate = QiskitGatePool.get_operation(params, action, _FakeGate)

    assert gate.kwargs["num_qubits"] == 3


def test_get_operation_no_params_returns_bare_gate_class():
    gate_class = QiskitGatePool.get_operation({}, {"params": []}, _FakeGate)
    assert gate_class is _FakeGate


def test_all_gates_pool_is_non_empty_and_well_formed():
    gates = QiskitGatePool.all_gates()
    assert len(gates) > 0
    for gate_class, num_qubits, params in gates:
        assert isinstance(num_qubits, int)
        assert num_qubits >= 1
        assert isinstance(params, dict)
