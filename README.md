# qec-diverse-circuits

A Gym-compatible RL environment that generates Qiskit circuits diverse in properties relevant to quantum error correction (QEC) and noise characterization: non-Clifford content, entangling-gate type and density, connectivity, depth, and idle time.

## Why these properties

Circuits that differ only in generic shape don't necessarily exercise different hardware error behavior. These specific properties do:

- **Non-Clifford content** (ratio, T-gate count) — determines whether a circuit needs magic-state resources, a different error regime than Clifford-only circuits.
- **Entangling-gate type and density** — different two-qubit gates stress different crosstalk channels.
- **Connectivity** — local vs. long-range interactions exercise different qubit pairs.
- **Depth and idle time** — proxy for decoherence exposure.

## Results

Validated on N=1000 generated circuits: the trained agent correlates with these properties at r=+0.574, beating a random baseline (r=0.554±0.023 across 5 seeds) by a small but real margin. It is not a large improvement — treat it as a modest, reproducible edge, not a dramatic one.

## Install

```bash
pip install -e .
```

## Usage

```python
from qec_diverse_circuits import DiverseCircuitGenerationEnv

env = DiverseCircuitGenerationEnv(max_qubits=7, max_gates=13, target_circuits=30)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # replace with a trained policy
    obs, reward, done, info = env.step(action)

for circuit, qec_properties in env.all_circuits:
    print(qec_properties)
```

### Training

```python
from qec_diverse_circuits.base_generator import BaseGenerator

generator = BaseGenerator(model_dir="./models")
generator.train()
circuits = generator.generate()
```

## Tests

```bash
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
