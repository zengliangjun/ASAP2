import torch
from typing import Union
import numpy as np
from isaac_utils import rotations

class MVStatistics:

    def __init__(self, shape: Union[tuple, torch.Size], device, episode_truncation = -1):
        self.episode_truncation = episode_truncation

        self.current_step = torch.zeros((shape[0], ), device=device, dtype=torch.long)
        self.episode_mean_buf = torch.zeros(shape, device=device, dtype=torch.float)
        self.episode_variance_buf = torch.zeros_like(self.episode_mean_buf)

        self.zero_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


    def clean(self):
        self.current_step[...] = 0
        self.episode_mean_buf[...] = 0
        self.episode_variance_buf[...] = 0

    def _calcute_step(self):
        self.current_step += 1
        if -1 == self.episode_truncation:
            return self.current_step
        else:
            return torch.clamp_max(self.current_step, self.episode_truncation)

    def update(self, input):
        step = self._calcute_step()
        self.zero_flag[:] = step <= 1

        step_e1 = step[:, None]
        while len(step_e1.shape) != len(self.episode_mean_buf.shape):
            step_e1 = step_e1[..., None]

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = input - self.episode_mean_buf
        self.episode_mean_buf += delta0 / step_e1

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = input - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (step_e1 - 2)
            + delta0 * delta1
        ) / (step_e1 - 1)

        # 当episode刚开始时重置方差，防止数值异常
        self.episode_variance_buf[self.zero_flag] = 0

    def reset(self, env_ids):
        self.zero_flag[env_ids] = 0
        self.current_step[env_ids] = 0
        self.episode_mean_buf[env_ids] =0
        self.episode_variance_buf[env_ids] = 0
