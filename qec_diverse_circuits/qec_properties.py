import numpy as np
import networkx as nx

CLIFFORD_GATES = frozenset(
    {"id", "x", "y", "z", "h", "s", "sdg", "cx", "cy", "cz", "swap", "sx", "sxdg"}
)

QEC_PROPERTIES_ORDER = (
    "non_clifford_ratio",
    "t_gate_ratio",
    "entangling_density",
    "entangling_type_diversity_ratio",
    "connectivity",
    "depth",
    "idle_fraction",
)


def compute_qec_properties(dag, num_qubits):
    """Extract the QEC/noise-characterization-relevant property vector from a
    circuit DAG, in priority order: non-Clifford/T-gate content first
    (determines which error channel class the circuit exercises), then
    entangling-gate type/density (different native 2Q gates stress different
    physical crosstalk channels), then connectivity (routing/decoherence
    exposure pattern), then depth/idle-time (T1/T2 decoherence proxy)."""
    # QiskitGatePool.get_dag never calls G.add_node for its per-qubit lineage
    # placeholder nodes (q{i} labels) -- they're auto-created, bare, by
    # G.add_edge the first time a gate references that qubit, so they carry
    # no 'node_data' key at all. Only nodes with node_data are real gates;
    # everything else is graph plumbing and must be excluded from every
    # ratio/count below, mirroring build_composite_vector's histogram loop.
    gate_nodes = [n for n, data in dag.nodes(data=True) if "node_data" in data]
    num_ops = len(gate_nodes)
    if num_ops == 0:
        return np.zeros(7, dtype=np.float32)

    gate_names = [dag.nodes[n]["node_data"].name.lower() for n in gate_nodes]

    non_clifford_count = sum(1 for name in gate_names if name not in CLIFFORD_GATES)
    non_clifford_ratio = non_clifford_count / num_ops

    t_gate_count = sum(1 for name in gate_names if name in ("t", "tdg"))
    t_gate_ratio = t_gate_count / num_ops

    entangling_nodes = [n for n in gate_nodes if dag.in_degree(n) >= 2]
    entangling_density = len(entangling_nodes) / num_ops

    if entangling_nodes:
        entangling_names = {
            dag.nodes[n]["node_data"].name.lower() for n in entangling_nodes
        }
        entangling_type_diversity_ratio = len(entangling_names) / len(entangling_nodes)
    else:
        entangling_type_diversity_ratio = 0.0

    if num_qubits <= 1:
        connectivity = 0.0
    else:
        touched_pairs = set()
        for n in entangling_nodes:
            preds = list(dag.predecessors(n))
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    touched_pairs.add((min(preds[i], preds[j]), max(preds[i], preds[j])))
        max_pairs = num_qubits * (num_qubits - 1) / 2
        connectivity = min(len(touched_pairs) / max_pairs, 1.0)

    depth = float(nx.dag_longest_path_length(dag)) if num_ops > 0 else 0.0

    if num_qubits > 0:
        idle_fraction = 1.0 - (num_ops / (num_qubits * max(depth, 1.0)))
        idle_fraction = float(np.clip(idle_fraction, 0.0, 1.0))
    else:
        idle_fraction = 0.0

    return np.array(
        [
            non_clifford_ratio,
            t_gate_ratio,
            entangling_density,
            entangling_type_diversity_ratio,
            connectivity,
            depth,
            idle_fraction,
        ],
        dtype=np.float32,
    )
