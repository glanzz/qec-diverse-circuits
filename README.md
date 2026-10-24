# qec-diverse-circuits

A [Gym](https://github.com/openai/gym)-compatible reinforcement learning environment that generates quantum circuits optimized for diversity along properties relevant to **quantum error correction (QEC) and noise characterization research** — not generic structural diversity.

Extracted and adapted from the [QARATE](https://github.com/glanzz/qarate) project.

## What this is

An RL agent (trained here with `RecurrentPPO` via `sb3-contrib`) incrementally builds Qiskit circuits, gate by gate. Each completed circuit is scored for novelty against an archive of previously generated circuits, using a composite vector that combines:

- **NetLSD graph heat-kernel signature** (250-dim) — circuit DAG topology
- **Gate-type usage histogram** (64-dim) — which gates were used, how often
- **Parameter statistics** (2-dim) — mean/variance of numeric gate parameters
- **Priority-weighted QEC property block** (7-dim) — non-Clifford ratio, T-gate ratio, entangling-gate density, entangling-gate-type diversity, connectivity, depth, idle-fraction

The QEC property block is weighted (non-Clifford/entangling content weighted highest, connectivity next, depth/idle-time lowest) before being folded into the archive-search vector, so the novelty signal specifically rewards circuits that differ along axes known to expose **different physical error channels** — not just circuits that look structurally different in a generic sense.

## Why these specific properties

QEC and noise-characterization researchers need circuits that, as a set, exercise genuinely different error behavior:

1. **Non-Clifford content** (ratio + T-gate count) — Clifford-only circuits are classically simulable and only exercise stabilizer-type error structure; non-Clifford content requires magic-state resources and exposes a categorically different error channel.
2. **Entangling-gate type and density** — different native two-qubit gates (CX/CZ/SWAP/iSwap-family) stress different physical crosstalk channels; density determines how much of a circuit's error budget comes from (typically highest-error) two-qubit operations.
3. **Connectivity / locality pattern** — nearest-neighbor-only vs. long-range interaction patterns change SWAP-routing overhead and which qubit pairs' physical crosstalk gets exercised.
4. **Depth and idle-qubit time** — a proxy for T1/T2 decoherence exposure.

## Validated results — read before relying on this

This environment's reward was validated empirically (N=1000 agent vs. random circuits, trained via 3 retrain iterations) against the properties above. **Report the result honestly, not optimistically:**

| Metric | Target | Actual result | Status |
|---|---|---|---|
| Priority-weighted property-distance correlation (agent) | r ≥ +0.35 | **r = +0.574** | Met |
| Agent beats random on priority-weighted correlation | margin ≥ 0.1 | Agent 0.574 vs. random 5-seed mean 0.554±0.023 — **margin ~0.02–0.03, below target** | Partially met |
| Per-property correlation floor (non-Clifford/entangling, ≥ +0.2 each) | all four ≥ 0.2 | non_clifford_ratio 0.403, t_gate_count 0.208, entangling_density 0.381, entangling_type_diversity 0.274 | Met |
| Property-space coverage (≥80% of a coarse grid) | — | Not formally re-measured against the final checkpoint | Not directly measured |
| No validity regression | 0% invalid placements | Full regression suite green | Met |
| No training-stability regression | healthy entropy/explained-variance/clip-fraction | Stable across all 3 retrain cycles | Met |

**The honest summary: this agent beats random by a small, real, reproducible margin on QEC-relevant property diversity — not by the large margin originally targeted.** If you need a large-margin improvement over random, this does not yet deliver that; what it delivers is a statistically real (not noise-level) but modest edge. Treat claims of "diverse circuit generation" accordingly.

This environment's predecessor (a NetLSD+gate-histogram-only novelty signal, with no QEC property block) was separately measured to be *worse than random* on these same properties (e.g., non-Clifford ratio: agent r=+0.041 vs. random r=+0.194) — the QEC property block above exists specifically because that signal didn't track error-relevant properties at all. If you're comparing against a generic-diversity-only version of this kind of environment, expect this one to do meaningfully better on QEC-relevant axes specifically, even though its margin over random is itself modest.

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

No external services required — novelty search uses an in-memory, brute-force k-NN archive (`qec_diverse_circuits.knn_archive.KNNCircuitArchive`), not a vector database.

## Usage

```python
from qec_diverse_circuits import DiverseCircuitGenerationEnv

env = DiverseCircuitGenerationEnv(max_qubits=7, max_gates=13, target_circuits=30)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # replace with a trained policy
    obs, reward, done, info = env.step(action)

# Generated circuits, each paired with its QEC property metadata:
for circuit, qec_properties in env.all_circuits:
    print(qec_properties)  # {'non_clifford_ratio': ..., 't_gate_ratio': ..., ...}
```

### Training a policy

```python
from qec_diverse_circuits.base_generator import BaseGenerator

generator = BaseGenerator(model_dir="./models")
generator.train()                 # trains and saves a versioned checkpoint
circuits = generator.generate()   # loads the latest checkpoint and generates circuits
```

## Gate pool

Qiskit-only. Includes standard single- and two-qubit gates (X/Y/Z/H/S/T/CX/CZ/SWAP/iSwap/ECR/...), the universal rotation gates (U, U1, U3), the two-qubit interaction family (RXX/RYY/RZZ/RZX), and several multi-controlled/Clifford+T-adjacent gates (MCX, MCPhase, QFT, QuantumVolume). See `qec_diverse_circuits/gate_pool.py` for the full list.

## Tests

```bash
pytest tests/
```

## License

Not yet specified by the maintainer — do not treat this as open for reuse until a LICENSE file is added.
