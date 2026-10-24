NETLSD_FEATURES_SIZE = 250
GATE_HISTOGRAM_FEATURES_SIZE = 64  # fixed-size slot for gate-type usage counts, hashed if gate pool exceeds this
PARAM_STATS_FEATURES_SIZE = 2  # mean, variance of numeric gate params used in the circuit
QEC_PROPERTIES_FEATURES_SIZE = 7  # non_clifford_ratio, t_gate_ratio, entangling_density, entangling_type_diversity_ratio, connectivity, depth, idle_fraction
QEC_PROPERTY_WEIGHTS = (2.0, 2.0, 2.0, 2.0, 1.0, 0.5, 0.5)  # priority weights, same order as QEC_PROPERTIES_ORDER; validated against an N=1000 agent-vs-random diagnostic (non-Clifford/entangling content weighted highest, connectivity next, depth/idle-time lowest)
COMPOSITE_FEATURES_SIZE = NETLSD_FEATURES_SIZE + GATE_HISTOGRAM_FEATURES_SIZE + PARAM_STATS_FEATURES_SIZE + QEC_PROPERTIES_FEATURES_SIZE

ARCHIVE_WARMUP_SIZE = 10  # minimum real archive entries before the novelty bonus pays out, so an empty archive's padding can't be farmed for free reward
NEW_GATE_SHAPING_BONUS = 0.02  # dense tie-breaker reward, kept small so it doesn't compete with the terminal novelty bonus
NEW_POSITION_SHAPING_BONUS = 0.01
GATE_DIVERSITY_BONUS = 0.05  # awarded once per gate type on its first use within an episode
ENTANGLING_TYPE_DIVERSITY_BONUS = 0.08  # awarded once per distinct entangling (>=2-qubit) gate type's first use within an episode
T_GATE_SHAPING_BONUS = 0.08  # awarded once per episode on first T/Tdg use; kept independent of the entangling-diversity bonus so T-gate usage isn't crowded out by "non-Clifford" credit earned from other gate types

MAX_QUBITS = 7
MAX_GATES = 13
TARGET_CIRCUITS = 30
