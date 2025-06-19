import torch
from typing import Union

class MVStatistics:

    def __init__(self, shape: Union[tuple, torch.Size], device, episode_truncation = -1):
        self.episode_truncation = episode_truncation

        self.current_step = torch.zeros((shape[0], ), device=device, dtype=torch.long)
        self.episode_mean_buf = torch.zeros(shape, device=device, dtype=torch.float)

        self.episode_variance_buf = torch.zeros_like(self.episode_mean_buf)


    def clean(self):
        self.current_step[:] = 0

        self.episode_mean_buf[:] = 0
        self.episode_variance_buf[:] = 0

    def _calcute_step(self):
        self.current_step += 1
        if -1 == self.episode_truncation:
            return self.current_step
        else:
            return torch.clamp_max(self.current_step, self.episode_truncation)

    def update(self, input):
        step = self._calcute_step()
        step_e1 = step[:, None]
        while len(step_e1.shape) != len(self.episode_mean_buf.shape):
            step_e1 = step_e1[:, None]

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
        mask = step <= 1
        self.episode_variance_buf[mask] = 0

    def calculate(self, episode_data): # b len dim
        mean_bufs = []
        variance_bufs = []
        for id in range(episode_data.shape[1]):
            input = episode_data[:, id, :]
            self.update(input)

            mean_bufs.append(self.episode_mean_buf.clone())
            variance_bufs.append(self.episode_variance_buf.clone())

        return torch.stack(mean_bufs, dim = 1), \
               torch.stack(variance_bufs, dim = 1)

    def reset(self, env_ids, steps, means, variances):
        self.current_step[env_ids] = steps[env_ids]

        self.episode_mean_buf[env_ids] = means[env_ids]
        self.episode_variance_buf[env_ids] =  variances[env_ids]

    def reset2(self, env_ids):
        self.current_step[env_ids] = 0

        self.episode_mean_buf[env_ids] =0
        self.episode_variance_buf[env_ids] = 0

class MVCStatistics(MVStatistics):

    def __init__(self, shape: tuple, device, episode_truncation = -1):
        super(MVCStatistics, self).__init__(shape, device, episode_truncation)

        self.episode_covariance_buf = torch.zeros((shape[0], shape[1], shape[1]),
                                                    device=device, dtype=torch.float)

    def clean(self):
        super(MVCStatistics, self).clean()
        self.episode_covariance_buf[:] = 0

    def update(self, input):

        step = self._calcute_step()
        step_e1 = step[:, None]
        step_e2 = step_e1[..., None]

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = input - self.episode_mean_buf
        self.episode_mean_buf += delta0 / step_e1

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = input - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (step_e1 - 2)
            + delta0 * delta1
        ) / (step_e1 - 1)

        covariance = torch.einsum("bi,bj->bij", delta1, delta0)
        self.episode_covariance_buf = (\
            self.episode_covariance_buf * (step_e2 - 2) + \
            covariance) / (step_e2 - 1)

        # 当episode刚开始时重置方差，防止数值异常
        mask = step <= 1
        self.episode_variance_buf[mask] = 0
        self.episode_covariance_buf[mask] = 0


    def calculate(self, episode_data): # b len dim
        mean_bufs = []
        variance_bufs = []
        covariance_bufs = []
        for id in range(episode_data.shape[1]):
            input = episode_data[:, id, :]
            self.update(input)

            mean_bufs.append(self.episode_mean_buf.clone())
            variance_bufs.append(self.episode_variance_buf.clone())
            covariance_bufs.append(self.episode_covariance_buf.clone())

        return torch.stack(mean_bufs, dim = 1), \
               torch.stack(variance_bufs, dim = 1), \
               torch.stack(covariance_bufs, dim = 1)

    def reset(self, env_ids, steps, means, variances, covariances):
        super(MVCStatistics, self).reset(env_ids, steps, means, variances)
        self.episode_covariance_buf[env_ids] = covariances[env_ids]

    def reset2(self, env_ids):
        super(MVCStatistics, self).reset2(env_ids)
        self.episode_covariance_buf[env_ids] = 0

