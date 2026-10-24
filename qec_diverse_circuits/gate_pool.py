import math

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import standard_gates
from qiskit.circuit.library import MCPhaseGate, MCXGate, MCXRecursive, QFTGate, QuantumVolume
from qiskit.circuit.random import random_circuit

from qec_diverse_circuits.utility import Utility


class Node:
    def __init__(self, name, qubit, index, params=None):
        self.name = name
        self.params = params or []
        self.qubit = qubit
        self.index = index

    def __str__(self):
        return f"{self.name}{tuple(self.params) if self.params else ''}"


class QiskitGatePool:
    """Qiskit-only gate pool and circuit/DAG utilities for the diverse-circuit
    Gym environment. Deliberately has no multi-backend adapter layer -- this
    package targets Qiskit only."""

    NODES_DATABASE = {}
    current_node_index = 1

    non_parametric_gates = [
        (standard_gates.XGate, 1, {}),
        (standard_gates.YGate, 1, {}),
        (standard_gates.ZGate, 1, {}),
        (standard_gates.CXGate, 2, {}),
        (standard_gates.SwapGate, 2, {}),
        (standard_gates.SGate, 1, {}),
        (standard_gates.HGate, 1, {}),
        (standard_gates.CCXGate, 3, {}),
        (standard_gates.CZGate, 2, {}),
        (standard_gates.C3SXGate, 4, {}),
        (standard_gates.IGate, 1, {}),
        (standard_gates.SXGate, 1, {}),
        (standard_gates.SdgGate, 1, {}),
        (standard_gates.TdgGate, 1, {}),
        (standard_gates.TGate, 1, {}),
        (standard_gates.CSXGate, 2, {}),
        (standard_gates.CYGate, 2, {}),
        (standard_gates.ECRGate, 2, {}),
        (standard_gates.CSGate, 2, {}),
        (standard_gates.CSdgGate, 2, {}),
        (standard_gates.iSwapGate, 2, {}),
        (standard_gates.CSwapGate, 3, {}),
        (standard_gates.CCZGate, 3, {}),
        (standard_gates.RCCXGate, 3, {}),
        (standard_gates.SXdgGate, 1, {}),
        (MCXRecursive, 4, {"num_ctrl_qubits": 3}),
        (QFTGate, 1, {"num_qubits": 1}),
        (QuantumVolume, 4, {"num_qubits": 4}),
        (QuantumVolume, 3, {"num_qubits": 3}),
        (QuantumVolume, 2, {"num_qubits": 2}),
        (MCXGate, 4, {"num_ctrl_qubits": 3, "label": "MCX(4)"}),
        (MCXGate, 3, {"num_ctrl_qubits": 2, "label": "MCX(3)"}),
    ]

    parametric_gates = [
        (standard_gates.RZGate, 1, {"phi": Utility.get_random_pi_value}),
        (standard_gates.PhaseGate, 1, {"theta": Utility.get_random_pi_value}),
        (standard_gates.RXGate, 1, {"theta": Utility.get_random_pi_value}),
        (standard_gates.RYGate, 1, {"theta": Utility.get_random_pi_value}),
        (
            standard_gates.UGate,
            1,
            {
                "theta": Utility.get_random_pi_value,
                "phi": Utility.get_random_pi_value,
                "lam": Utility.get_random_pi_value,
            },
        ),
        (standard_gates.U1Gate, 1, {"theta": Utility.get_random_pi_value}),
        (
            standard_gates.U3Gate,
            1,
            {
                "theta": Utility.get_random_pi_value,
                "phi": Utility.get_random_pi_value,
                "lam": Utility.get_random_pi_value,
            },
        ),
        (standard_gates.CPhaseGate, 2, {"theta": Utility.get_random_pi_value}),
        (standard_gates.CRXGate, 2, {"theta": Utility.get_random_pi_value}),
        (standard_gates.CRYGate, 2, {"theta": Utility.get_random_pi_value}),
        (standard_gates.CRZGate, 2, {"theta": Utility.get_random_pi_value}),
        (
            standard_gates.CUGate,
            2,
            {
                "theta": Utility.get_random_pi_value,
                "phi": Utility.get_random_pi_value,
                "lam": Utility.get_random_pi_value,
                "gamma": Utility.get_random_pi_value,
            },
        ),
        (standard_gates.CU1Gate, 2, {"theta": Utility.get_random_pi_value}),
        (
            standard_gates.CU3Gate,
            2,
            {
                "theta": Utility.get_random_pi_value,
                "phi": Utility.get_random_pi_value,
                "lam": Utility.get_random_pi_value,
            },
        ),
        (standard_gates.RXXGate, 2, {"theta": Utility.get_random_pi_value}),
        (standard_gates.RYYGate, 2, {"theta": Utility.get_random_pi_value}),
        (standard_gates.RZZGate, 2, {"theta": Utility.get_random_pi_value}),
        (standard_gates.RZXGate, 2, {"theta": Utility.get_random_pi_value}),
        (
            MCPhaseGate,
            4,
            {"lam": Utility.get_random_pi_value, "num_ctrl_qubits": 3},
        ),
    ]
    max_gate_params = 4

    @classmethod
    def all_gates(cls):
        return [*cls.non_parametric_gates, *cls.parametric_gates]

    @classmethod
    def get_node_index(cls, name, force_update=False):
        if not force_update:
            existing = cls.NODES_DATABASE.get(name)
            if existing is not None:
                return existing
        cls.NODES_DATABASE[name] = cls.current_node_index
        cls.current_node_index += 1
        return cls.NODES_DATABASE[name]

    @classmethod
    def get_operation(cls, params, action, gate_class):
        """Construct a Gate instance from a gate-pool entry's parameter spec
        and the agent's sampled action. Callable param specs (e.g.
        Utility.get_random_pi_value) mark a parameter the agent controls
        directly via its raw sampled integer; that raw value (from a
        MultiDiscrete([1000, ...]) action component, i.e. an integer in
        [0, 999]) is mapped to an angle via pi / (raw_value + 1) rather than
        pi / raw_value, so a sampled 0 can never divide by zero."""
        param_index = 0
        suppliable_gate_parameters = {}
        for param_key in params.keys():
            if callable(params[param_key]):
                raw_value = action.get("params")[param_index]
                divisor = raw_value + 1
                suppliable_gate_parameters[param_key] = math.pi / divisor
                param_index += 1
            else:
                suppliable_gate_parameters[param_key] = params[param_key]
        return (
            gate_class
            if not suppliable_gate_parameters
            else gate_class(**suppliable_gate_parameters)
        )

    @classmethod
    def get_random_circuit(cls, qubits, operations):
        return random_circuit(qubits, operations)

    @classmethod
    def init_circuit(cls, qubits):
        return QuantumCircuit(qubits)

    @classmethod
    def get_dag(cls, circuit):
        circuit = circuit.copy()
        node_mapping = {}
        G = nx.DiGraph()
        last_known_nodes = []

        for i in range(circuit.num_qubits):
            node_name = f"q{i}"
            # force_update=True: each gate INSTANCE must get its own graph
            # node -- the default name-memoized index would collapse every
            # gate of the same type (and every qubit placeholder of the same
            # index, across circuits) onto a single shared node, corrupting
            # edges whenever a gate name repeats within a circuit.
            node_index = cls.get_node_index(node_name, force_update=True)
            qubit_node = Node(node_name, i, -1)
            last_known_nodes.append(node_index)
            node_mapping[node_index] = qubit_node

        positions_tracker = [0] * circuit.num_qubits
        last_pos = [-1, -1]
        for inst, qubits, clbits in circuit:
            if inst.name == "barrier":
                continue
            if inst.name == "measure":
                continue

            target_qubit = qubits[-1]
            node_name = f"{inst.name}"
            last_pos = [target_qubit._index, positions_tracker[target_qubit._index]]
            node_index = cls.get_node_index(node_name, force_update=True)
            current_node = Node(
                inst.name,
                target_qubit._index,
                positions_tracker[target_qubit._index],
                params=inst.params,
            )
            G.add_node(
                node_index,
                node_data=current_node,
                label=str(current_node),
                features={"params": inst.params},
            )
            node_mapping[node_index] = current_node
            positions_tracker[target_qubit._index] += 1
            G.add_edge(last_known_nodes[target_qubit._index], node_index)
            last_known_nodes[target_qubit._index] = node_index
            for remaining_qubit in qubits[:-1]:
                G.add_edge(last_known_nodes[remaining_qubit._index], node_index)

        return G, last_pos


class QiskitCircuit:
    """Thin wrapper around a QuantumCircuit used by the Gym environment while
    incrementally constructing a circuit, one gate per step."""

    def __init__(self, gate_pool, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = gate_pool.init_circuit(num_qubits)

    def append(self, operation, qubits_index):
        self.circuit.append(operation, qubits_index)

    def get_circuit(self):
        return self.circuit

    def copy(self):
        return self.circuit.copy()
