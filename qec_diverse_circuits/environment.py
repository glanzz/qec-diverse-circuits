import zlib
from gymnasium import Env
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
from time import time
import netlsd

from qec_diverse_circuits.knn_archive import KNNCircuitArchive
from qec_diverse_circuits.utility import Utility
from qec_diverse_circuits.exceptions import UnUniqueQubits, FilledQubits
from qec_diverse_circuits.gate_pool import QiskitGatePool, QiskitCircuit
from qec_diverse_circuits.qec_properties import compute_qec_properties, QEC_PROPERTIES_ORDER
from qec_diverse_circuits.constants import (
    NETLSD_FEATURES_SIZE,
    GATE_HISTOGRAM_FEATURES_SIZE,
    PARAM_STATS_FEATURES_SIZE,
    QEC_PROPERTIES_FEATURES_SIZE,
    QEC_PROPERTY_WEIGHTS,
    COMPOSITE_FEATURES_SIZE,
    ARCHIVE_WARMUP_SIZE,
    NEW_GATE_SHAPING_BONUS,
    NEW_POSITION_SHAPING_BONUS,
    GATE_DIVERSITY_BONUS,
    ENTANGLING_TYPE_DIVERSITY_BONUS,
    T_GATE_SHAPING_BONUS,
)


def build_composite_vector(dag, max_gate_index, num_qubits):
    """Build a vector combining NetLSD topology signature, a gate-type usage
    histogram, basic parameter statistics, and a priority-weighted
    QEC/noise-characterization property block (non-Clifford/T-gate content,
    entangling-gate type/density, connectivity, depth/idle-time), so the
    novelty archive's distance metric tracks properties that determine which
    physical error channels a circuit exercises -- not just generic
    structural/compositional difference. The QEC block is weighted before
    concatenation (not at search time) so the archive's native L2 search
    reflects the priority weighting with no changes to the storage/search
    layer."""
    # netlsd's normalized-Laplacian math assumes an undirected graph; on a
    # directed DAG the Laplacian can be asymmetric with negative eigenvalues,
    # which blows up the heat-kernel trace to astronomical magnitudes.
    topology_vec = Utility.safe_normalize(netlsd.heat(dag.to_undirected()))
    topology_vec = np.asarray(topology_vec, dtype=np.float32)
    if topology_vec.shape[0] != NETLSD_FEATURES_SIZE:
        topology_vec = np.resize(topology_vec, (NETLSD_FEATURES_SIZE,)).astype(np.float32)

    histogram = np.zeros((GATE_HISTOGRAM_FEATURES_SIZE,), dtype=np.float32)
    all_params = []
    for _, data in dag.nodes(data=True):
        node_data = data.get("node_data")
        if node_data is None:
            continue
        # Python's built-in hash() randomizes string hashing per process
        # (PYTHONHASHSEED), so two separate runs can bucket the same gate
        # name differently -- silently corrupting novelty-archive distance
        # comparisons between vectors computed in different processes (e.g.
        # a saved archive from training vs. a later generate() call). zlib's
        # crc32 is deterministic across processes and Python versions.
        bucket = zlib.crc32(node_data.name.encode("utf-8")) % GATE_HISTOGRAM_FEATURES_SIZE
        histogram[bucket] += 1.0
        for p in (node_data.params or []):
            if isinstance(p, (int, float)) and np.isfinite(p):
                all_params.append(float(p))

    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    if all_params:
        param_stats = np.array(
            [np.mean(all_params), np.var(all_params)], dtype=np.float32
        )
    else:
        param_stats = np.zeros((PARAM_STATS_FEATURES_SIZE,), dtype=np.float32)

    qec_block = compute_qec_properties(dag, num_qubits) * np.array(QEC_PROPERTY_WEIGHTS, dtype=np.float32)

    return np.concatenate([topology_vec, histogram, param_stats, qec_block]).astype(np.float32)


class DiverseCircuitGenerationEnv(Env):
    def __init__(self, max_qubits=7, max_gates=13, target_circuits=30, storage=None):
        super(DiverseCircuitGenerationEnv, self).__init__()
        self.GATE_POOL = QiskitGatePool
        self.CIRCUIT_CLASS = QiskitCircuit

        self.all_circuits = []
        self.target_circuits = target_circuits
        self.max_qubits = max_qubits  # Fixed number of qbits in the circuit
        self.max_gates = max_gates  # Maximum operations in the program
        self.max_gate_params = (
            self.GATE_POOL.max_gate_params
        )  # Maximum parameters across the gate pool

        # Global
        self.total_explored = 0
        self.episode = 0
        self.storage = storage if storage is not None else KNNCircuitArchive()
        self._owns_storage = storage is None
        self.storage_id = 1

        # Local
        self.step_count = 0
        self.seen_gates = []
        self.last_pos = [-1, -1]
        self.farpoints = 0
        self.start_time = None
        self.decomposed_action = {}
        self.episode_calls = 0
        self.operations = None
        self.qubits = None
        self.circuit = None
        self.circuit_vector = np.zeros((COMPOSITE_FEATURES_SIZE,), dtype=np.float32)
        self.max_gate_index = len(self.GATE_POOL.all_gates())

        """Defining Action space and observation space"""
        self.action_space = MultiDiscrete(
            [
                self.max_gate_index,
                self.max_qubits,
                self.max_qubits,
                self.max_qubits,
                self.max_qubits,
                1000,
                1000,
                1000,
                1000,
            ]
        )
        low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + ([-1] * 10),
            dtype=np.float32,
        )
        high = np.array(
            [
                110.0,
                1000.0,
                100000.0,
                float(self.max_gates),
                float(self.max_qubits),
                11.0,
                float(self.max_gates),
                float(self.max_qubits),
                float(self.max_gate_index),
            ]
            + ([2] * 10),
            dtype=np.float32,
        )
        self.observation_space = Box(low, high, (len(low),), np.float32)

        self.prev_10_gates = [-1] * self.max_gate_index
        self.collected_circuits = []
        self.episode_complete = False
        self.distances = [-1] * 10
        self.ids = [-1] * 10
        self.last10Pos = [-1] * 10
        self.prevexplored = 0
        self.stagnant_count = 0
        self.level = 0
        self.gate_types_used_this_episode = set()
        self.entangling_gate_types_used_this_episode = set()
        self.t_gate_used_this_episode = False

    def init_circuit(self, qubits, operations):
        self.qubits = qubits
        self.operations = operations
        self.circuit = self.CIRCUIT_CLASS(self.GATE_POOL, qubits)
        self.step_count = 0
        self.start_time = time()
        self.circuit_vector = np.zeros((COMPOSITE_FEATURES_SIZE,), dtype=np.float32)
        self.negative_direction = 0
        self.last_pos = [-1, -1]
        self.num_param = 0
        self.circuit_complete = False
        self.farpoints = 0
        self.distances = [-1] * 10
        self.ids = [-1] * 10
        self.episode_complete = False
        self.last_pos_10 = [-1] * self.qubits

    def step(self, action):
        self.episode_calls += 1
        params = action[5:]
        self.decomposed_action = {
            "params": params,
            "gate_index": action[0] % self.max_gate_index,
            "qubits": action[1: 1 + 4],
        }

        action = self.decomposed_action
        error = False
        applied = False

        if self.circuit is None:
            applied = True
            self.init_circuit(
                np.random.randint(1, self.max_qubits),
                np.random.randint(1, self.max_gates),
            )

        new_gate = action.get("gate_index") not in self.prev_10_gates[-10:]
        applied = True
        gate_class, number_gate_qubits, params = self.select_compatible_gate(
            self.GATE_POOL.all_gates(), action.get("gate_index")
        )
        operation = self.GATE_POOL.get_operation(params, action, gate_class)
        resolved_qubits = self.resolve_qubit_indices(
            list(action.get("qubits")), number_gate_qubits
        )
        try:
            self.circuit.append(operation, resolved_qubits)
            self.step_count += 1
        except FilledQubits:
            error = True
        except UnUniqueQubits:
            error = True
        except Exception:
            error = True

        if (not applied) or error:
            reward = -1
            observation = self.get_obs()
            self.reset_circuit()

            return observation, reward, False, False, {}

        if not self.circuit:
            reward = 0
            observation = self.get_obs()
            self.reset_circuit()

            return observation, reward, False, False, {}

        try:
            self.dag, lastPos = self.GATE_POOL.get_dag(self.circuit)
        except Exception:
            observation = self.get_obs()
            self.reset_circuit()
            return observation, -5, False, False, {}

        self.circuit_vector = build_composite_vector(self.dag, self.max_gate_index, self.qubits)

        reward = 0  # Valid action reward
        reward += self.compute_step_shaping_bonus(
            new_gate=new_gate, is_new_position=lastPos[0] not in self.last_pos_10
        )
        reward += self.compute_gate_diversity_bonus(action.get("gate_index"))
        reward += self.compute_entangling_diversity_bonus(gate_class.__name__, number_gate_qubits)
        reward += self.compute_t_gate_bonus(gate_class.__name__)

        self.prev_10_gates = self.prev_10_gates[1:] + [action.get("gate_index")]
        self.last_pos_10 = self.last_pos_10[1:] + [lastPos[0]]

        self.last_pos = lastPos

        self.circuit_complete = self.step_count >= self.operations
        score, farpoints, ids, distances, _ = self.storage.search_n_nearest(
            [self.circuit_vector]
        )

        self.ids = ids
        self.distances = distances
        self.farpoints = farpoints

        if self.circuit_complete:
            reward += self.compute_novelty_bonus()

        if self.circuit_complete and (self.farpoints >= 10):
            raw_qec_properties = compute_qec_properties(self.dag, self.qubits)
            self.append_circuit_with_metadata(self.circuit.get_circuit(), raw_qec_properties)
            try:
                self.storage.insert_vector([self.storage_id], [self.circuit_vector])
                self.storage_id += 1
                self.total_explored += 1
            except ValueError:
                pass  # degenerate vector -- don't count it as a genuinely explored unique circuit

        observation = self.get_obs()
        if self.circuit_complete:
            self.reset_circuit()
            if self.prevexplored == self.total_explored:
                self.stagnant_count += 1
                self.prevexplored = self.total_explored

            if self.total_explored == self.target_circuits:
                self.episode_complete = True
                return observation, reward, True, False, {}
            return observation, reward, False, False, {}
        return observation, reward, False, False, {}

    def compute_novelty_bonus(self):
        """Graded novelty bonus: continuous in mean distance to nearest
        neighbors, and gated on the archive having enough real (non-padding)
        entries that early-episode padding can't be farmed for free reward."""
        if self.storage_id <= ARCHIVE_WARMUP_SIZE:
            return 0.0
        mean_distance = sum(self.distances) / len(self.distances)
        # Smoothly saturating bonus: grows with mean distance, never cliffs.
        return 10.0 * np.tanh(mean_distance)

    def compute_step_shaping_bonus(self, new_gate, is_new_position):
        """Dense per-step tie-breaker reward. Kept an order of magnitude
        below the terminal novelty bonus so it guides action selection
        without giving the policy a cheap, always-available alternative to
        pursuing real circuit novelty."""
        bonus = 0.0
        bonus += NEW_GATE_SHAPING_BONUS if new_gate else 0.0
        bonus += NEW_POSITION_SHAPING_BONUS if is_new_position else 0.0
        return bonus

    def compute_gate_diversity_bonus(self, gate_index):
        """Awards credit the first time a gate type is used within the
        current episode, rewarding breadth of gate-type coverage across a
        whole episode rather than only short-window (last-10-step) novelty."""
        if gate_index in self.gate_types_used_this_episode:
            return 0.0
        self.gate_types_used_this_episode.add(gate_index)
        return GATE_DIVERSITY_BONUS

    def compute_entangling_diversity_bonus(self, gate_class_name, number_gate_qubits):
        """Awards credit the first time a distinct entangling (>=2-qubit)
        gate type is used within the current episode. Gives PPO per-step
        credit for which specific entangling gate type was chosen, since the
        terminal-only novelty bonus alone gives no such attribution."""
        if number_gate_qubits < 2:
            return 0.0
        if gate_class_name in self.entangling_gate_types_used_this_episode:
            return 0.0
        self.entangling_gate_types_used_this_episode.add(gate_class_name)
        return ENTANGLING_TYPE_DIVERSITY_BONUS

    def compute_t_gate_bonus(self, gate_class_name):
        """Awards credit once per episode on first T/Tdg use. Kept
        independent of compute_entangling_diversity_bonus: that bonus
        rewards any non-Clifford/entangling gate choice, which would
        otherwise crowd out T-gates specifically for "non-Clifford" credit,
        so T-gate usage needs its own independent per-step signal."""
        if gate_class_name not in ("TGate", "TdgGate"):
            return 0.0
        if self.t_gate_used_this_episode:
            return 0.0
        self.t_gate_used_this_episode = True
        return T_GATE_SHAPING_BONUS

    def append_circuit_with_metadata(self, circuit, raw_qec_properties):
        """Pairs a generated circuit with its unweighted QEC property values
        (not the weighted archive-search copy) so a downstream consumer can
        filter/select by property -- e.g. 'give me the high-T-count,
        low-connectivity ones' -- without recomputing from the raw circuit."""
        metadata = dict(zip(QEC_PROPERTIES_ORDER, (float(v) for v in raw_qec_properties)))
        self.all_circuits.append((circuit, metadata))

    def select_compatible_gate(self, all_gates, gate_index):
        """Remaps a raw sampled gate_index onto the subset of gates whose
        arity fits the current circuit's qubit count, so the policy can
        never select a gate requiring more qubits than the circuit has.
        Eliminates gate-arity mismatches as a source of invalid placements."""
        compatible_gates = [g for g in all_gates if g[1] <= self.qubits]
        if not compatible_gates:
            raise ValueError(
                f"no gate in the pool fits a circuit with {self.qubits} qubit(s)"
            )
        return compatible_gates[gate_index % len(compatible_gates)]

    def resolve_qubit_indices(self, raw_indices, num_needed):
        """Remaps raw sampled qubit-slot values into valid, distinct indices
        within [0, self.qubits), eliminating out-of-range and duplicate-qubit
        placements as failure modes. Collisions after modulo are resolved by
        advancing deterministically to the next unused index."""
        if num_needed > self.qubits:
            raise ValueError(
                f"circuit has {self.qubits} qubit(s), cannot supply {num_needed} distinct indices"
            )
        used = set()
        resolved = []
        for raw in raw_indices[:num_needed]:
            idx = raw % self.qubits
            while idx in used:
                idx = (idx + 1) % self.qubits
            used.add(idx)
            resolved.append(idx)
        return resolved

    def get_obs(self):
        prevobs = [
            self.target_circuits,
            self.step_count or 0,
            self.total_explored or 0,
            self.operations or 0,
            self.qubits or 0,
            self.farpoints or 0,
            self.last_pos[0],
            self.last_pos[1],
            self.decomposed_action.get("gate_index", 0),
            *self.distances,
        ]

        return np.array(prevobs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.episode_calls:
            print("Explore success rate", self.total_explored / self.episode_calls)
        self.reset_episode()

        return self.get_obs(), {}

    def reset_circuit(self):
        self.qubits = None
        self.operations = None
        self.circuit = None
        self.step_count = 0
        self.start_time = None
        self.circuit_vector = np.zeros((COMPOSITE_FEATURES_SIZE,), dtype=np.float32)
        self.negative_direction = 0
        self.num_param = 0

    def reset_episode(self):
        self.episode += 1
        if self._owns_storage:
            self.storage = KNNCircuitArchive()
            self.storage_id = 1
        self.episode_calls = 0
        self.collected_circuits = []
        self.total_explored = 0
        self.prev_10_gates = [-1] * self.max_gate_index
        self.last_pos = [-1, -1]
        self.gate_types_used_this_episode = set()
        self.entangling_gate_types_used_this_episode = set()
        self.t_gate_used_this_episode = False
