import os
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from qec_diverse_circuits.base_generator import BaseGenerator, VECNORMALIZE_FILENAME


def _tiny_generator(tmp_path):
    return BaseGenerator(
        model_dir=str(tmp_path / "models"),
        hyperparameters=dict(
            n_steps=8,
            learning_rate=0.001,
            gamma=0.99,
            ent_coef=0.001,
            n_epochs=1,
            verbose=0,
            device="cpu",
        ),
        total_timesteps=16,
        max_qubits=3,
        max_gates=4,
        target_circuits=1,
    )


def test_set_env_without_path_builds_fresh_vecnormalize(tmp_path):
    generator = _tiny_generator(tmp_path)
    generator._set_env(training=True)
    assert isinstance(generator.env, VecNormalize)
    assert generator.env.training is True


def test_set_env_defaults_to_training_mode_on_fresh_env(tmp_path):
    generator = _tiny_generator(tmp_path)
    generator._set_env()
    assert generator.env.training is True


def test_train_saves_model_and_vecnormalize_stats(tmp_path):
    generator = _tiny_generator(tmp_path)
    versioned_directory = generator.train()

    assert os.path.exists(os.path.join(versioned_directory, "model.zip"))
    assert os.path.exists(os.path.join(versioned_directory, VECNORMALIZE_FILENAME))
    assert os.path.exists(os.path.join(versioned_directory, "hyperparameters.json"))


def test_generate_loads_saved_vecnormalize_stats_not_fresh_ones(tmp_path):
    generator = _tiny_generator(tmp_path)
    versioned_directory = generator.train()

    # Capture the trained observation-normalization statistics directly from
    # the saved file, independent of BaseGenerator, as the ground truth.
    trained_stats = VecNormalize.load(
        os.path.join(versioned_directory, VECNORMALIZE_FILENAME),
        generator.env.venv.__class__([generator._make_env]),
    )

    circuits = generator.generate(model_dir=versioned_directory)

    # generate() must have loaded the same statistics train() produced, not
    # a fresh VecNormalize's untrained defaults (mean=0, var=1).
    assert np.allclose(generator.env.obs_rms.mean, trained_stats.obs_rms.mean)
    assert np.allclose(generator.env.obs_rms.var, trained_stats.obs_rms.var)
    assert isinstance(circuits, list)


def test_generate_sets_training_false_so_stats_stop_updating(tmp_path):
    generator = _tiny_generator(tmp_path)
    versioned_directory = generator.train()

    generator.generate(model_dir=versioned_directory)

    assert generator.env.training is False


def test_generate_returns_circuit_metadata_tuples(tmp_path):
    generator = _tiny_generator(tmp_path)
    versioned_directory = generator.train()

    circuits = generator.generate(model_dir=versioned_directory)

    for circuit, metadata in circuits:
        assert isinstance(metadata, dict)
