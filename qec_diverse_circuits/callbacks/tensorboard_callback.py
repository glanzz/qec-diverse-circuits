from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from qec_diverse_circuits.environment import DiverseCircuitGenerationEnv


class TensorboardCallback(BaseCallback):
    def __init__(self, env: DiverseCircuitGenerationEnv, log_dir: str, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix="_custom_metrics")
        self.total_circuits = 0
        self.total_unique_circuits = 0
        self.timesteps = 0
        self.epsiodes = 0
        self.env = env

    def _on_step(self) -> bool:
        self.timesteps += 1

        if self.env.circuit_complete:
            self.total_circuits += 1
            if self.env.farpoints == 10:
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
        if self.env.episode_complete:
            self.total_circuits = 0
            self.total_unique_circuits = 0
            self.epsiodes += 1
            self.env.level += 1

        return True

    def _on_training_end(self) -> None:
        self.writer.close()
