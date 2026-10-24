import numpy as np
import pytest
from qec_diverse_circuits.environment import DiverseCircuitGenerationEnv
from qec_diverse_circuits.constants import (
    ARCHIVE_WARMUP_SIZE,
    NEW_GATE_SHAPING_BONUS,
    NEW_POSITION_SHAPING_BONUS,
    GATE_DIVERSITY_BONUS,
    ENTANGLING_TYPE_DIVERSITY_BONUS,
    T_GATE_SHAPING_BONUS,
)


def _make_env_with_mocks():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env.storage_id = 1
    env.distances = [2.0] * 10
    env.farpoints = 10
    return env


def test_no_novelty_bonus_awarded_during_archive_warmup():
    env = _make_env_with_mocks()
    env.storage_id = 1
    bonus = env.compute_novelty_bonus()
    assert bonus == 0.0


def test_novelty_bonus_is_graded_not_cliff_shaped():
    env = _make_env_with_mocks()
    env.storage_id = ARCHIVE_WARMUP_SIZE + 1

    env.distances = [0.05] * 9 + [0.005]
    bonus_near_cliff = env.compute_novelty_bonus()

    env.distances = [0.05] * 10
    bonus_at_cliff = env.compute_novelty_bonus()

    assert bonus_near_cliff > 0.0
    assert bonus_at_cliff > bonus_near_cliff


def test_dead_early_farpoints_reset_is_removed():
    import inspect
    source = inspect.getsource(DiverseCircuitGenerationEnv.step)
    before_search_call = source.split("search_n_nearest")[0]
    assert "self.farpoints = 0" not in before_search_call


def test_step_shaping_bonus_is_scaled_down_from_original_magnitudes():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)

    bonus_both = env.compute_step_shaping_bonus(new_gate=True, is_new_position=True)
    bonus_neither = env.compute_step_shaping_bonus(new_gate=False, is_new_position=False)

    assert bonus_both == pytest.approx(NEW_GATE_SHAPING_BONUS + NEW_POSITION_SHAPING_BONUS)
    assert bonus_neither == 0.0
    max_shaping_per_circuit = bonus_both * 13
    assert max_shaping_per_circuit < 1.0


def test_gate_diversity_bonus_awarded_once_per_gate_type_per_episode():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env.gate_types_used_this_episode = set()

    first_use = env.compute_gate_diversity_bonus(gate_index=5)
    repeat_use = env.compute_gate_diversity_bonus(gate_index=5)
    different_gate = env.compute_gate_diversity_bonus(gate_index=9)

    assert first_use == GATE_DIVERSITY_BONUS
    assert repeat_use == 0.0
    assert different_gate == GATE_DIVERSITY_BONUS
    assert env.gate_types_used_this_episode == {5, 9}


def test_step_uses_shaping_and_diversity_bonus_helpers():
    import inspect
    source = inspect.getsource(DiverseCircuitGenerationEnv.step)
    assert "compute_step_shaping_bonus" in source
    assert "compute_gate_diversity_bonus" in source
    assert "reward += 0.2" not in source
    assert "reward += 0.1" not in source


def _reset_episode_fixture_env():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env._owns_storage = False
    env.episode = 0
    env.episode_calls = 0
    env.collected_circuits = []
    env.total_explored = 0
    env.max_gate_index = 49
    env.prev_10_gates = [-1] * 49
    env.last_pos = [-1, -1]
    return env


def test_gate_diversity_tracking_resets_on_new_episode():
    env = _reset_episode_fixture_env()
    env.gate_types_used_this_episode = {5, 9}

    env.reset_episode()

    assert env.gate_types_used_this_episode == set()


def test_all_circuits_entries_carry_qec_property_metadata():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env.all_circuits = []

    fake_circuit = object()
    fake_qec_vector = np.array([0.5, 0.1, 0.3, 0.4, 0.2, 5.0, 0.1], dtype=np.float32)

    env.append_circuit_with_metadata(fake_circuit, fake_qec_vector)

    assert len(env.all_circuits) == 1
    stored_circuit, metadata = env.all_circuits[0]
    assert stored_circuit is fake_circuit
    assert metadata == {
        "non_clifford_ratio": pytest.approx(0.5),
        "t_gate_ratio": pytest.approx(0.1),
        "entangling_density": pytest.approx(0.3),
        "entangling_type_diversity_ratio": pytest.approx(0.4),
        "connectivity": pytest.approx(0.2),
        "depth": pytest.approx(5.0),
        "idle_fraction": pytest.approx(0.1),
    }


def test_entangling_diversity_bonus_awarded_once_per_entangling_gate_type_per_episode():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env.entangling_gate_types_used_this_episode = set()

    first_cx = env.compute_entangling_diversity_bonus("cx", number_gate_qubits=2)
    repeat_cx = env.compute_entangling_diversity_bonus("cx", number_gate_qubits=2)
    first_cz = env.compute_entangling_diversity_bonus("cz", number_gate_qubits=2)
    single_qubit = env.compute_entangling_diversity_bonus("h", number_gate_qubits=1)

    assert first_cx == ENTANGLING_TYPE_DIVERSITY_BONUS
    assert repeat_cx == 0.0
    assert first_cz == ENTANGLING_TYPE_DIVERSITY_BONUS
    assert single_qubit == 0.0
    assert env.entangling_gate_types_used_this_episode == {"cx", "cz"}


def test_entangling_diversity_tracking_resets_on_new_episode():
    env = _reset_episode_fixture_env()
    env.entangling_gate_types_used_this_episode = {"cx", "cz"}
    env.gate_types_used_this_episode = {5, 9}

    env.reset_episode()

    assert env.entangling_gate_types_used_this_episode == set()


def test_step_uses_entangling_diversity_bonus_helper():
    import inspect
    source = inspect.getsource(DiverseCircuitGenerationEnv.step)
    assert "compute_entangling_diversity_bonus" in source


def test_t_gate_bonus_awarded_once_per_episode_on_first_t_gate_use():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env.t_gate_used_this_episode = False

    first_t = env.compute_t_gate_bonus("TGate")
    repeat_t = env.compute_t_gate_bonus("TGate")
    other_gate = env.compute_t_gate_bonus("HGate")

    assert first_t == T_GATE_SHAPING_BONUS
    assert repeat_t == 0.0
    assert other_gate == 0.0
    assert env.t_gate_used_this_episode is True


def test_t_gate_bonus_recognizes_tdg_too():
    env = DiverseCircuitGenerationEnv.__new__(DiverseCircuitGenerationEnv)
    env.t_gate_used_this_episode = False

    bonus = env.compute_t_gate_bonus("TdgGate")

    assert bonus == T_GATE_SHAPING_BONUS


def test_t_gate_tracking_resets_on_new_episode():
    env = _reset_episode_fixture_env()
    env.t_gate_used_this_episode = True
    env.entangling_gate_types_used_this_episode = {"cx"}
    env.gate_types_used_this_episode = {5}

    env.reset_episode()

    assert env.t_gate_used_this_episode is False


def test_step_uses_t_gate_bonus_helper():
    import inspect
    source = inspect.getsource(DiverseCircuitGenerationEnv.step)
    assert "compute_t_gate_bonus" in source


def test_env_constructs_with_default_args():
    env = DiverseCircuitGenerationEnv(max_qubits=3, max_gates=5, target_circuits=2)
    assert env.max_qubits == 3
    assert env.max_gates == 5
    assert env.target_circuits == 2
    assert env.action_space.shape == (9,)
