# qec-diverse-circuits

A Gymnasium-compatible RL environment that generates Qiskit circuits diverse in properties relevant to quantum error correction (QEC) and noise characterization: non-Clifford content, entangling-gate type and density, connectivity, depth, and idle time.

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
pip install qec-diverse-circuits
```

For local development:

```bash
pip install -e .
```

## Usage

```python
from qec_diverse_circuits import DiverseCircuitGenerationEnv

env = DiverseCircuitGenerationEnv(max_qubits=7, max_gates=13, target_circuits=30)
obs, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # replace with a trained policy
    obs, reward, terminated, truncated, info = env.step(action)

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

No pretrained checkpoint is published yet — `generate()` trains one from scratch (~7e5 timesteps) the first time it's called if `model_dir` is empty. `train()` and `generate()` use SB3's `VecNormalize` internally, so the observation/reward scaling a generated checkpoint is loaded with always matches what it was trained under.

## Tests

```bash
pytest tests/
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md). This project follows [Semantic Versioning](https://semver.org/).

## Citation

If you use this package in research, please cite it — see [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
