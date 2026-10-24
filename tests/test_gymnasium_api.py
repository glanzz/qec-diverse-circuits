from qec_diverse_circuits import DiverseCircuitGenerationEnv


def test_reset_returns_obs_info_tuple():
    env = DiverseCircuitGenerationEnv(max_qubits=4, max_gates=6, target_circuits=2)
    result = env.reset()
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert isinstance(info, dict)


def test_step_returns_five_tuple():
    env = DiverseCircuitGenerationEnv(max_qubits=4, max_gates=6, target_circuits=2)
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_full_episode_terminates_via_gymnasium_contract():
    env = DiverseCircuitGenerationEnv(max_qubits=4, max_gates=6, target_circuits=3)
    obs, info = env.reset()

    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 500:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert terminated is True
    assert truncated is False
    assert len(env.all_circuits) == env.target_circuits
