import torch
from . import motion_tracking
from humanoidverse.utils import statistics


class Tracking(motion_tracking.LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        shape = (self.num_envs, self.dim_actions)
        shape = (self.num_envs, self.num_bodies + self.num_extend_bodies, 3)
        self.statistics_body_pos = statistics.MVStatistics(shape, device)

    def _init_tracking_config(self):
        super()._init_tracking_config()

    # _post_physics_step
    ## _pre_compute_observations_callback
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        #
        self.statistics_body_pos.update(self.dif_global_body_pos)

    ## reset_envs_idx
    ### _reset_robot_states_callback
    #### _reset_dofs
    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        self.statistics_body_pos.reset(env_ids)

    #### _reset_root_states
    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

    ###############################################################
    def _rew_mean_rbpos_diff(self, body_ids, std: float = 0.25) -> torch.Tensor:
        episode_mean = self.statistics_body_pos.episode_mean_buf[:, body_ids]
        episode_mean = torch.reshape(episode_mean, (self.num_envs, -1))

        r = torch.exp(- torch.square(episode_mean) / std)
        r = torch.mean(r, dim=-1)

        r[self.statistics_body_pos.zero_flag] = 0
        return r

    def _rew_variance_rbpos_diff(self, body_ids, std: float = 0.25) -> torch.Tensor:
        episode_variance = self.statistics_body_pos.episode_variance_buf[:, body_ids]
        episode_variance = torch.reshape(episode_variance, (self.num_envs, -1))

        r = torch.exp(- episode_variance / std)
        r = torch.mean(r, dim=-1)

        r[self.statistics_body_pos.zero_flag] = 0
        return r

    ## body pos
    def _reward_upper_mean_rbpos(self):
        return self._rew_mean_rbpos_diff(self.upper_body_id, self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

    def _reward_lower_mean_rbpos(self):
        return self._rew_mean_rbpos_diff(self.lower_body_id, self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

    def _reward_vr_mean_rbpos(self):
        return self._rew_mean_rbpos_diff(self.motion_tracking_id, self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

    def _reward_upper_variance_rbpos(self):
        return self._rew_variance_rbpos_diff(self.upper_body_id, self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

    def _reward_lower_variance_rbpos(self):
        return self._rew_variance_rbpos_diff(self.lower_body_id, self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

    def _reward_vr_variance_rbpos(self):
        return self._rew_variance_rbpos_diff(self.motion_tracking_id, self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
