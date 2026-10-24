from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter


class TensorboardCallback(BaseCallback):
    """Reads DiverseCircuitGenerationEnv's per-step bookkeeping attributes
    (circuit_complete, farpoints, episode_complete, level) through the
    wrapping VecEnv's get_attr/set_attr, rather than directly off `env`, so
    this callback works whether `env` is a bare DiverseCircuitGenerationEnv
    wrapped only enough to satisfy SB3's VecEnv requirement, or sits behind
    VecNormalize/DummyVecEnv -- neither of which forwards arbitrary
    attribute access the way a plain gymnasium.Wrapper does."""

    def __init__(self, env: VecEnv, log_dir: str, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix="_custom_metrics")
        self.total_circuits = 0
        self.total_unique_circuits = 0
        self.timesteps = 0
        self.epsiodes = 0
        self.env = env

    def _get_env_attr(self, name):
        return self.env.get_attr(name)[0]

    def _on_step(self) -> bool:
        self.timesteps += 1

        if self._get_env_attr("circuit_complete"):
            self.total_circuits += 1
            if self._get_env_attr("farpoints") == 10:
                self.total_unique_circuits += 1
            self.writer.add_scalar(
                "Completely Unique circuits v/s total circuits (All distant)-- Episode"
                + str(self.epsiodes),
                self.total_unique_circuits,
                self.total_circuits,
            )
        self.writer.add_scalar(
            "Circuits v/s timesteps", self.total_circuits, self.timesteps
        )
        self.writer.add_scalar(
            "Unique circuit v/s timesteps", self.total_unique_circuits, self.timesteps
        )
        if self._get_env_attr("episode_complete"):
            self.total_circuits = 0
            self.total_unique_circuits = 0
            self.epsiodes += 1
            self.env.set_attr("level", self._get_env_attr("level") + 1)

        return True

    def _on_training_end(self) -> None:
        self.writer.close()
