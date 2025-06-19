import torch
from . import motion_tracking
from humanoidverse.utils import statistics


class Tracking(motion_tracking.LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)

        shape = (self.num_envs, self.dim_actions)
        self.statistics_joint_angles = statistics.MVStatistics(shape, device)
        self.statistics_joint_velocities = statistics.MVStatistics(shape, device)

        shape = (self.num_envs, self.num_bodies + self.num_extend_bodies, 3)
        self.statistics_body_pos = statistics.MVStatistics(shape, device)
        self.statistics_body_vel = statistics.MVStatistics(shape, device)
        self.statistics_body_ang_vel = statistics.MVStatistics(shape, device) # dif_global_body_ang_vel


    def _init_tracking_config(self):
        super()._init_tracking_config()

        if "lower_body_link" in self.config.robot.motion:
            left_lower_body_id = []
            right_lower_body_id = []
            for link in self.config.robot.motion.lower_body_link:
                if link.startswith("left"):
                    left_lower_body_id.append(self.simulator._body_list.index(link))
                elif link.startswith("right"):
                    right_lower_body_id.append(self.simulator._body_list.index(link))

            self.left_lower_body_id = left_lower_body_id
            self.right_lower_body_id = right_lower_body_id

        if "upper_body_link" in self.config.robot.motion:
            left_upper_body_id = []
            right_upper_body_id = []
            for link in self.config.robot.motion.upper_body_link:
                if link.startswith("left"):
                    left_upper_body_id.append(self.simulator._body_list.index(link))
                elif link.startswith("right"):
                    right_upper_body_id.append(self.simulator._body_list.index(link))

            self.left_upper_body_id = left_upper_body_id
            self.right_upper_body_id = right_upper_body_id

    # _post_physics_step
    ## _pre_compute_observations_callback
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        #
        self.statistics_joint_angles.update(self.dif_joint_angles)
        self.statistics_joint_velocities.update(self.dif_joint_velocities)
        self.statistics_body_pos.update(self.dif_global_body_pos)
        self.statistics_body_vel.update(self.dif_global_body_vel)
        self.statistics_body_ang_vel.update(self.dif_global_body_ang_vel)

    ## reset_envs_idx
    ### _reset_robot_states_callback
    #### _reset_dofs
    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)

        self.statistics_joint_angles.reset2(env_ids)
        self.statistics_joint_velocities.reset2(env_ids)
        self.statistics_body_pos.reset2(env_ids)
        self.statistics_body_vel.reset2(env_ids)
        self.statistics_body_ang_vel.reset2(env_ids)

    #### _reset_root_states
    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

    ###############################################################
    ## body pos
    def _body_pos_symmetry(self, _leftids, _rightids, std):
        left_mean = self.statistics_body_pos.episode_mean_buf[:, _leftids, :]
        right_mean = self.statistics_body_pos.episode_mean_buf[:, _rightids, :]

        diff = left_mean - right_mean
        diff = (diff**2).mean(dim=-1).mean(dim=-1)

        r_mean = torch.exp(-diff / std)

        left_variance = self.statistics_body_pos.episode_variance_buf[:, _leftids, :]
        right_variance = self.statistics_body_pos.episode_variance_buf[:, _rightids, :]

        diff = left_variance - right_variance
        diff = torch.abs(diff).mean(dim=-1).mean(dim=-1)

        r_variance = torch.exp(-diff / std)

        return (r_mean + r_variance) / 2

    def _reward_S_upper_symmetry(self):
        return self._body_pos_symmetry(self.left_upper_body_id, self.right_upper_body_id, \
                                       self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

    def _reward_S_lower_symmetry(self):
        return self._body_pos_symmetry(self.left_lower_body_id, self.right_lower_body_id, \
                                       self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

    # self
    def _diff(self, values):
        count = 0
        total = None
        for id0 in range(values.shape[1]):
            for id1 in range(1, values.shape[1]):
                if id0 == id1:
                    continue
                diff = torch.abs(values[:, id0] - values[:, id1])
                if None == total:
                    total = diff
                else:
                    total += diff

                count += 1

        return total / count

    def _body_pos_self(self, bodyids, std):
        mean = self.statistics_body_pos.episode_mean_buf[:, bodyids, :]
        mean = self._diff(mean)

        variance = self.statistics_body_pos.episode_variance_buf[:, bodyids, :]
        variance = self._diff(variance)

        mean = (mean**2).mean(dim=-1)
        r_mean = torch.exp(-mean / std)


        variance = variance.mean(dim=-1)
        r_variance = torch.exp(-variance / std)

        return (r_mean + r_variance) / 2

    def _reward_S_tracking(self):
        return self._body_pos_self(self.motion_tracking_id, self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

    def _reward_S_feet(self):
        return self._body_pos_self(self.feet_indices, self.config.rewards.reward_tracking_sigma.teleop_feet_pos)


    ## body ang velocity
    def _reward_S_angular_mean(self):
        mean_diff = self.statistics_body_ang_vel.episode_mean_buf
        mean_diff = (mean_diff**2).mean(dim=-1).mean(dim=-1)
        r_mean = torch.exp(-mean_diff / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)

        flags = self.episode_length_buf <= 1
        r_mean[flags] = 0
        return r_mean

    def _reward_S_angular_var(self):
        var_diff = self.statistics_body_ang_vel.episode_variance_buf
        var_diff = (var_diff).mean(dim=-1).mean(dim=-1)
        r_var = torch.exp(-var_diff / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)

        flags = self.episode_length_buf <= 1
        r_var[flags] = 0
        return r_var

    ## body velocity
    def _reward_S_velocity_mean(self):
        mean_diff = self.statistics_body_vel.episode_mean_buf
        mean_diff = (mean_diff**2).mean(dim=-1).mean(dim=-1)
        r_mean = torch.exp(-mean_diff / self.config.rewards.reward_tracking_sigma.teleop_body_vel)

        flags = self.episode_length_buf <= 1
        r_mean[flags] = 0
        return r_mean

    def _reward_S_velocity_var(self):
        var_diff = self.statistics_body_vel.episode_variance_buf
        var_diff = (var_diff).mean(dim=-1).mean(dim=-1)
        r_var = torch.exp(-var_diff / self.config.rewards.reward_tracking_sigma.teleop_body_vel)

        flags = self.episode_length_buf <= 1
        r_var[flags] = 0
        return r_var

    ## dof pos
    def _reward_S_dof_pos_mean(self):
        mean_diff = self.statistics_joint_angles.episode_mean_buf
        mean_diff = (mean_diff**2).mean(dim=-1)
        r_mean = torch.exp(-mean_diff / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)

        flags = self.episode_length_buf <= 1
        r_mean[flags] = 0
        return r_mean

    def _reward_S_dof_pos_var(self):
        variance_diff = self.statistics_joint_angles.episode_variance_buf
        variance_diff = torch.abs(variance_diff).mean(dim=-1)
        r_variance = torch.exp(-variance_diff / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)

        flags = self.episode_length_buf <= 1
        r_variance[flags] = 0
        return r_variance

    def _reward_S_dof_vels_mean(self):
        mean_diff = self.statistics_joint_velocities.episode_mean_buf
        mean_diff = (mean_diff**2).mean(dim=-1)
        r_mean = torch.exp(-mean_diff / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)

        flags = self.episode_length_buf <= 1
        r_mean[flags] = 0
        return r_mean

    def _reward_S_dof_vels_var(self):
        variance_diff = self.statistics_joint_velocities.episode_variance_buf
        variance_diff = torch.abs(variance_diff).mean(dim=-1)
        r_variance = torch.exp(-variance_diff / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)

        flags = self.episode_length_buf <= 1
        r_variance[flags] = 0
        return r_variance

