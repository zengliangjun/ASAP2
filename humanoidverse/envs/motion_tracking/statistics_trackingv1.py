import torch
from . import statistics_tracking
from humanoidverse.utils import statistics


class Tracking(statistics_tracking.Tracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        shape = (self.num_envs, 3)
        self.statistics_base_pos = statistics.MVStatistics(shape, device)

    def _init_tracking_config(self):
        super()._init_tracking_config()
        assert "base_link" in self.config.robot.motion
        self.base_id = self.simulator._body_list.index(self.config.robot.motion.base_link)

    def _pre_compute_observations_callback(self):
        super(statistics_tracking.Tracking, self)._pre_compute_observations_callback()
        #

        dif_base = self.dif_global_body_pos[:, self.base_id, :]

        self.statistics_base_pos.update(dif_base)
        dif = self.dif_global_body_pos - self.dif_global_body_pos[:, self.base_id: self.base_id + 1, :]
        self.statistics_body_pos.update(dif)

    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        self.statistics_base_pos.reset(env_ids)

    def _reward_lower_mean_rbpos(self):
        return self._rew_mean_rbpos_diff(self.lower_body_id[: -1], self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

    def _reward_lower_variance_rbpos(self):
        return self._rew_variance_rbpos_diff(self.lower_body_id[: -1], self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

    ## base pos
    def _reward_base_mean_rbpos(self):
        r = torch.exp(- torch.square(self.statistics_base_pos.episode_mean_buf) / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r = torch.mean(r, dim=-1)
        return r

    def _reward_base_variance_rbpos(self):
        r = torch.exp(- self.statistics_base_pos.episode_variance_buf / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r = torch.mean(r, dim=-1)
        return r
