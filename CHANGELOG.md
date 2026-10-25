# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CITATION.cff` for academic citation.
- This changelog.
- PyPI trusted publishing via GitHub Actions, triggered on `v*` git tags.

## [0.1.0] - 2026-10-24

### Added
- Initial extraction of the QEC-property-diverse circuit generation environment: `DiverseCircuitGenerationEnv`, `build_composite_vector`, `compute_qec_properties`.
- In-memory `KNNCircuitArchive` novelty archive, replacing the original Milvus-backed vector database dependency.
- Qiskit-only gate pool (`QiskitGatePool`, `QiskitCircuit`), dropping the multi-backend adapter layer.
- `BaseGenerator` training/generation pipeline built on `sb3_contrib.RecurrentPPO`.
- Test suite (pytest) covering the QEC property extractor, composite vector construction, reward shaping, the gate pool, and the KNN archive.
- MIT `LICENSE`.

### Changed
- Migrated from the unmaintained legacy `gym` API to `gymnasium`: `step()` now returns the 5-tuple `(obs, reward, terminated, truncated, info)`, `reset()` accepts `seed`/`options` and returns `(obs, info)`.
- Replaced the hand-rolled `NormalizedRewardEnv` observation/reward normalizer with Stable-Baselines3's `VecNormalize`, so `generate()` loads the exact normalization statistics a checkpoint was trained with instead of starting from fresh, untrained statistics.
- Tightened all dependency version ranges (previously open-ended `>=` floors) to bounded ranges, and loosened the hard-pinned `qiskit==1.2.0` to a range so CI can test multiple Qiskit versions.

### Fixed
- Division-by-zero in parametric gate angle construction when an agent's sampled raw parameter value was `0`.
- Archive novelty-search padding vectors used a hardcoded dimension instead of the actual embedding size.
- Degenerate (all-zero or non-finite) vectors could be inserted into the novelty archive, corrupting later distance comparisons.
- All-or-nothing novelty reward cliff replaced with a graded, continuous bonus; added an archive warm-up gate so an empty archive's padding can't be farmed for free reward.
- Gate-histogram bucketing used Python's built-in `hash()`, which is randomized per process (`PYTHONHASHSEED`) — two different gate names could land in the same bucket depending on the process, silently corrupting novelty-archive comparisons across process boundaries (e.g. a checkpoint trained in one process and loaded by `generate()` in another). Replaced with a deterministic `zlib.crc32`-based hash.

[Unreleased]: https://github.com/glanzz/qec-diverse-circuits/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/glanzz/qec-diverse-circuits/releases/tag/v0.1.0
