import gym
import numpy as np
from qec_diverse_circuits.env_wrappers.running_std_mean import RunningMeanStd


class NormalizedRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.running_mean = 0
        self.running_var = 1
        self.count = 1e-4
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = 1e-8

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obfilt(obs)

        self.count += 1
        alpha = 0.001  # decay rate
        self.running_mean = (1 - alpha) * self.running_mean + alpha * reward
        self.running_var = (1 - alpha) * self.running_var + alpha * (
            reward - self.running_mean
        ) ** 2

        std = (self.running_var + 1e-8) ** 0.5
        normalized_reward = (reward - self.running_mean) / std

        return obs, normalized_reward, done, info

    def _obfilt(self, obs):
        self.ob_rms.update(obs)
        obs = (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon)
        return obs

    def reset(self):
        obs = self.env.reset()
        return self._obfilt(obs)
