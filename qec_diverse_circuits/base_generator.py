import os
from typing import List

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qec_diverse_circuits.environment import DiverseCircuitGenerationEnv
from qec_diverse_circuits.utility import Utility
from qec_diverse_circuits.constants import MAX_QUBITS, MAX_GATES, TARGET_CIRCUITS

VECNORMALIZE_FILENAME = "vecnormalize.pkl"

DEFAULT_HYPERPARAMETERS = dict(
    n_steps=1280 * 3,
    learning_rate=0.000008,
    gamma=0.99,
    ent_coef=0.001,
    n_epochs=100,
    verbose=1,
    device="cpu",
)
DEFAULT_TOTAL_TIME_STEPS = 7e5


class BaseGenerator:
    """Trains a RecurrentPPO policy against DiverseCircuitGenerationEnv and
    generates circuits from a trained checkpoint. Models and hyperparameters
    are versioned under `model_dir` (v0, v1, ... subdirectories)."""

    def __init__(
        self,
        model_dir: str = "./models",
        hyperparameters: dict = None,
        total_timesteps: float = DEFAULT_TOTAL_TIME_STEPS,
        max_qubits: int = MAX_QUBITS,
        max_gates: int = MAX_GATES,
        target_circuits: int = TARGET_CIRCUITS,
    ):
        self.model_dir = model_dir
        self.hyperparameters = hyperparameters or DEFAULT_HYPERPARAMETERS
        self.total_timesteps = total_timesteps
        self.max_qubits = max_qubits
        self.max_gates = max_gates
        self.target_circuits = target_circuits

    def _make_env(self) -> DiverseCircuitGenerationEnv:
        return DiverseCircuitGenerationEnv(
            max_qubits=self.max_qubits,
            max_gates=self.max_gates,
            target_circuits=self.target_circuits,
        )

    def _set_env(self, vecnormalize_path: str = None, training: bool = True):
        """Builds the DummyVecEnv + VecNormalize pipeline used for both
        training and generation. Passing `vecnormalize_path` loads
        previously saved observation/reward normalization statistics
        instead of starting from fresh (untrained) ones -- this is what
        makes generate() see observations scaled the same way the policy
        was actually trained on, instead of the near-untrained statistics a
        freshly constructed VecNormalize would start with."""
        vec_env = DummyVecEnv([self._make_env])
        if vecnormalize_path is not None:
            self.env = VecNormalize.load(vecnormalize_path, vec_env)
            self.env.training = training
        else:
            self.env = VecNormalize(vec_env, training=training)

    def train(self) -> str:
        """Trains the model and saves it, along with the VecNormalize
        statistics accumulated during training, in a freshly versioned
        subdirectory of model_dir. Returns the path to that directory."""
        print("Agent: Training model....")
        versioned_directory = Utility.create_next_versioned_directory(self.model_dir)
        self._set_env(training=True)

        Utility.save_json(
            os.path.join(versioned_directory, "hyperparameters.json"),
            {
                "total_timesteps": self.total_timesteps,
                "max_qubits": self.max_qubits,
                "max_gates": self.max_gates,
                "target_circuits": self.target_circuits,
                **self.hyperparameters,
            },
        )

        model = RecurrentPPO(
            "MlpLstmPolicy",
            self.env,
            tensorboard_log=versioned_directory,
            **self.hyperparameters,
        )

        from qec_diverse_circuits.callbacks import TensorboardCallback

        callback = TensorboardCallback(self.env, log_dir=versioned_directory)
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback,
            tb_log_name="basic_metrics",
        )
        model_path = os.path.join(versioned_directory, "model")
        model.save(model_path)
        self.env.save(os.path.join(versioned_directory, VECNORMALIZE_FILENAME))
        self.env.close()
        return versioned_directory

    def generate(self, model_dir: str = None) -> List:
        """Generates circuits from a trained checkpoint. Pass model_dir
        (a specific versioned directory containing model.zip and
        vecnormalize.pkl) to load a specific checkpoint; defaults to the
        latest versioned directory under self.model_dir. Trains a fresh
        checkpoint first if none exists yet.

        Loads the VecNormalize statistics saved alongside the model (rather
        than starting from fresh ones) and sets training=False, so the
        observations the policy sees here are scaled the same way they were
        during training -- generation no longer has to approximate the
        normalization the policy actually learned against.

        Returns a list of (circuit, qec_property_metadata) tuples -- see
        DiverseCircuitGenerationEnv.append_circuit_with_metadata."""
        if model_dir is None:
            model_dir = Utility.get_latest_versioned_directory(self.model_dir)
            if model_dir is None:
                print("No trained checkpoint found, training a new one.")
                model_dir = self.train()

        print("Agent: Generating circuits....")
        vecnormalize_path = os.path.join(model_dir, VECNORMALIZE_FILENAME)
        self._set_env(vecnormalize_path=vecnormalize_path, training=False)
        model = RecurrentPPO.load(os.path.join(model_dir, "model"))

        obs = self.env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, dones, infos = self.env.step(action)
            if dones[0]:
                return self.env.get_attr("all_circuits")[0]
