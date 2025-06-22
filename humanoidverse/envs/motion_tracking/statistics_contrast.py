import torch
import torch.nn.functional as F
from . import motion_tracking
from humanoidverse.utils import statistics

from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)


class Tracking(motion_tracking.LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)

        joint_shape = (self.num_envs, self.dim_actions)
        body_shape = (self.num_envs, self.num_bodies + self.num_extend_bodies, 3)
        body_rot_shape = (self.num_envs, self.num_bodies + self.num_extend_bodies, 4)

        upper_body_shape = (self.num_envs, len(self.upper_body_id), 3)
        lower_body_shape = (self.num_envs, len(self.lower_body_id), 3)
        tracking_body_shape = (self.num_envs, len(self.motion_tracking_id), 3)
        feet_body_shape = (self.num_envs, len(self.feet_indices), 3)


        self.target_joint_angles = statistics.MVCStatistics2(joint_shape, device, 5)
        self.policy_joint_angles = statistics.MVCStatistics2(joint_shape, device, 5)

        # rot
        self.target_body_rot = statistics.MVQuat2(body_rot_shape, device, 5)
        self.policy_body_rot = statistics.MVQuat2(body_rot_shape, device, 5)

        # body_position
        self.target_upper_body_pos = statistics.MVCStatistics2(upper_body_shape, device, 5)
        self.policy_upper_body_pos = statistics.MVCStatistics2(upper_body_shape, device, 5)
        self.target_lower_body_pos = statistics.MVCStatistics2(lower_body_shape, device, 5)
        self.policy_lower_body_pos = statistics.MVCStatistics2(lower_body_shape, device, 5)
        self.target_tracking_body_pos = statistics.MVCStatistics2(tracking_body_shape, device, 5)
        self.policy_tracking_body_pos = statistics.MVCStatistics2(tracking_body_shape, device, 5)
        self.target_feet_body_pos = statistics.MVCStatistics2(feet_body_shape, device, 5)
        self.policy_feet_body_pos = statistics.MVCStatistics2(feet_body_shape, device, 5)

        self.target_body_pos = statistics.MVCStatistics2(body_shape, device, 5)
        self.policy_body_pos = statistics.MVCStatistics2(body_shape, device, 5)

        ##################################################
        self.pre_ref_body_pos_extend = torch.zeros(body_shape, device = device)
        #self.pre_ref_body_vel_extend = torch.zeros(body_shape, device = device)
        #self.pre_ref_body_ang_vel_extend = torch.zeros(body_shape, device = device)
        self.pre_ref_joint_pos = torch.zeros(joint_shape, device = device)
        #self.pre_ref_joint_vel = torch.zeros(joint_shape, device = device)

        # rot
        self.pre_ref_rigid_body_rot_extend = torch.zeros(body_rot_shape, device = device)


        # policy
        self.pre_rigid_body_pos_extend = torch.zeros(body_shape, device = device)
        #self.pre_rigid_body_vel_extend = torch.zeros(body_shape, device = device)
        #self.pre_rigid_body_ang_vel_extend = torch.zeros(body_shape, device = device)
        self.pre_joint_pos = torch.zeros(joint_shape, device = device)
        #self.pre_joint_vel = torch.zeros(joint_shape, device = device)

        # rot
        self.pre_rigid_body_rot_extend = torch.zeros(body_rot_shape, device = device)

        self.DEBUG_PLOT_REWARD = False
        if self.DEBUG_PLOT_REWARD:
            self.reward_collect = {}


    def _collect(self, key, value):
        if not self.DEBUG_PLOT_REWARD:
            return

        if key in self.reward_collect:
            self.reward_collect[key].append(value)
        else:
            self.reward_collect[key] = [value]


    # _post_physics_step
    ## _pre_compute_observations_callback
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()

        flags = self.episode_length_buf <= 1
        #
        # target
        target_diff_joint_angles = self.ref_joint_pos - self.pre_ref_joint_pos
        target_diff_body_pos = self.ref_body_pos_extend - self.pre_ref_body_pos_extend

        target_diff_rot = quat_mul(self.ref_body_rot_extend, quat_conjugate(self.pre_ref_rigid_body_rot_extend, w_last=True), w_last=True)


        target_diff_joint_angles[flags] = 0
        target_diff_body_pos[flags] = 0
        target_diff_rot[flags, ..., : -1] = 0
        target_diff_rot[flags, ..., -1] = 1

        # policy
        policy_diff_joint_angles = self.simulator.dof_pos - self.pre_joint_pos
        policy_diff_body_pos = self._rigid_body_pos_extend - self.pre_rigid_body_pos_extend

        policy_diff_rot = quat_mul(self._rigid_body_rot_extend, quat_conjugate(self.pre_rigid_body_rot_extend, w_last=True), w_last=True)

        policy_diff_joint_angles[flags] = 0
        policy_diff_body_pos[flags] = 0
        policy_diff_rot[flags, ..., : -1] = 0
        policy_diff_rot[flags, ..., -1] = 1

        ## statistics
        self.target_joint_angles.update(target_diff_joint_angles)
        self.policy_joint_angles.update(policy_diff_joint_angles)


        self.target_upper_body_pos.update(target_diff_body_pos[:, self.upper_body_id, :])
        self.policy_upper_body_pos.update(policy_diff_body_pos[:, self.upper_body_id, :])
        self.target_lower_body_pos.update(target_diff_body_pos[:, self.lower_body_id, :])
        self.policy_lower_body_pos.update(policy_diff_body_pos[:, self.lower_body_id, :])
        self.target_tracking_body_pos.update(target_diff_body_pos[:, self.motion_tracking_id, :])
        self.policy_tracking_body_pos.update(policy_diff_body_pos[:, self.motion_tracking_id, :])
        self.target_feet_body_pos.update(target_diff_body_pos[:, self.feet_indices, :])
        self.policy_feet_body_pos.update(policy_diff_body_pos[:, self.feet_indices, :])

        self.target_body_pos.update(target_diff_body_pos)
        self.policy_body_pos.update(policy_diff_body_pos)

        self.target_body_rot.update(target_diff_rot)
        self.policy_body_rot.update(policy_diff_rot)
        ## update
        # target
        self.pre_ref_body_pos_extend[...] = self.ref_body_pos_extend
        #self.pre_ref_body_vel_extend[...] = self.ref_body_vel_extend
        #self.pre_ref_body_ang_vel_extend[...] = self.ref_body_ang_vel_extend
        self.pre_ref_joint_pos[...] = self.ref_joint_pos
        #self.pre_ref_joint_vel[...] = self.ref_joint_vel

        self.pre_ref_rigid_body_rot_extend[...] = self.ref_body_rot_extend

        # policy
        self.pre_rigid_body_pos_extend[...] = self._rigid_body_pos_extend
        #self.pre_rigid_body_vel_extend[...] = self._rigid_body_vel_extend
        #self.pre_rigid_body_ang_vel_extend[...] = self._rigid_body_ang_vel_extend
        self.pre_joint_pos[...] = self.simulator.dof_pos
        #self.pre_joint_vel[...] = self.simulator.dof_vel

        self.pre_rigid_body_rot_extend[...] = self._rigid_body_rot_extend


    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)

        self.target_joint_angles.reset2(env_ids)
        self.policy_joint_angles.reset2(env_ids)

        self.target_upper_body_pos.reset2(env_ids)
        self.policy_upper_body_pos.reset2(env_ids)
        self.target_lower_body_pos.reset2(env_ids)
        self.policy_lower_body_pos.reset2(env_ids)
        self.target_tracking_body_pos.reset2(env_ids)
        self.policy_tracking_body_pos.reset2(env_ids)
        self.target_feet_body_pos.reset2(env_ids)
        self.policy_feet_body_pos.reset2(env_ids)

        self.target_body_pos.reset2(env_ids)
        self.policy_body_pos.reset2(env_ids)


        ## update
        # target
        self.pre_ref_body_pos_extend[env_ids] = 0
        self.pre_ref_joint_pos[env_ids] = 0
        # policy
        self.pre_rigid_body_pos_extend[env_ids] = 0
        self.pre_joint_pos[env_ids] = 0


        if self.DEBUG_PLOT_REWARD:
            import matplotlib.pyplot as plt
            import numpy as np

            if 0 != len(self.reward_collect):
                rows = len(self.reward_collect)
                fig, axs = plt.subplots(2, 1)
                for key, value in self.reward_collect.items():
                    time = np.linspace(0, len(value), len(value))
                    break

                id = 0
                for key, value in self.reward_collect.items():
                    if key.startswith("S_"):
                        a = axs[0]
                    else:
                        a = axs[1]
                    a.plot(time, np.array(value), label=key)
                    a.legend()
                    id += 1

                plt.show()
                self.reward_collect.clear()

    ###############################################################
    def _similarity(self, target, policy, mse_weight=0.5, cos_weight=0.5, eps=1e-6):
        """
        计算关节角对称性的奖励，结合均方误差和余弦相似度。
        Args:
            mse_weight (float): 均方误差部分的权重
            cos_weight (float): 余弦相似度部分的权重
            eps (float): 防止除零的小常数
        Returns:
            torch.Tensor: 奖励值
        """

        # 均方误差部分
        if len(target.shape) == 3:
            target = torch.norm(target, dim=-1)
            policy = torch.norm(policy, dim=-1)

        diff = target - policy
        norm_target = torch.norm(target, dim=-1)
        norm_policy = torch.norm(policy, dim=-1)
        norm_mean = (norm_target + norm_policy) / 2.0
        reward_mse = - torch.norm(diff, dim=-1) / (norm_mean + eps)

        # 余弦相似度部分
        reward_cos_sim = F.cosine_similarity(target, policy, dim=-1)

        # 奖励加权组合
        reward = mse_weight * torch.exp(reward_mse) + cos_weight * reward_cos_sim

        return reward

    def _diff(self, diff, std):
        if len(diff.shape) == 3:
            diff = torch.mean(torch.norm(diff**2, dim=-1), dim=-1)
        else:
            diff = torch.mean(diff**2, dim=-1)

        reward = torch.exp(- diff / std)

        return reward

    ## bodypos
    def _reward_S_bodypos_mean(self):
        reward = self._similarity(self.target_body_pos.episode_mean_buf.flatten(1),
                         self.policy_body_pos.episode_mean_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos, self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_bodypos_variance(self):
        reward = self._similarity(self.target_body_pos.episode_variance_buf.flatten(1),
                         self.policy_body_pos.episode_variance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos, self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_bodypos_covariance(self):
        reward = self._similarity(self.target_body_pos.episode_covariance_buf.flatten(1),
                         self.policy_body_pos.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos, self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    ##
    ## upper_body
    def _reward_S_upperpos_mean(self):
        reward = self._similarity(self.target_upper_body_pos.episode_mean_buf.flatten(1),
                         self.policy_upper_body_pos.episode_mean_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.upper_body_id, :], self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_upperpos_variance(self):
        reward = self._similarity(self.target_upper_body_pos.episode_variance_buf.flatten(1),
                         self.policy_upper_body_pos.episode_variance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.upper_body_id, :], self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_upperpos_covariance(self):
        reward = self._similarity(self.target_upper_body_pos.episode_covariance_buf.flatten(1),
                         self.policy_upper_body_pos.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.upper_body_id, :], self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    ##
    ## lower_body
    def _reward_S_lowerpos_mean(self):
        reward = self._similarity(self.target_lower_body_pos.episode_mean_buf.flatten(1),
                         self.policy_lower_body_pos.episode_mean_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.lower_body_id, :], self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_lowerpos_variance(self):
        reward = self._similarity(self.target_lower_body_pos.episode_variance_buf.flatten(1),
                         self.policy_lower_body_pos.episode_variance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.lower_body_id, :], self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_lowerpos_covariance(self):
        reward = self._similarity(self.target_lower_body_pos.episode_covariance_buf.flatten(1),
                         self.policy_lower_body_pos.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.lower_body_id, :], self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward


    ##
    ## tracking_body
    def _reward_S_trackingpos_mean(self):
        reward = self._similarity(self.target_tracking_body_pos.episode_mean_buf.flatten(1),
                         self.policy_tracking_body_pos.episode_mean_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.motion_tracking_id, :], self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_trackingpos_variance(self):
        reward = self._similarity(self.target_tracking_body_pos.episode_variance_buf.flatten(1),
                         self.policy_tracking_body_pos.episode_variance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.motion_tracking_id, :], self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_trackingpos_covariance(self):
        reward = self._similarity(self.target_tracking_body_pos.episode_covariance_buf.flatten(1),
                         self.policy_tracking_body_pos.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.motion_tracking_id, :], self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    ##
    ## feet_body
    def _reward_S_feetpos_mean(self):
        reward = self._similarity(self.target_feet_body_pos.episode_mean_buf.flatten(1),
                         self.policy_feet_body_pos.episode_mean_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.feet_indices, :], self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_feetpos_variance(self):
        reward = self._similarity(self.target_feet_body_pos.episode_variance_buf.flatten(1),
                         self.policy_feet_body_pos.episode_variance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.feet_indices, :], self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_feetpos_covariance(self):
        reward = self._similarity(self.target_feet_body_pos.episode_covariance_buf.flatten(1),
                         self.policy_feet_body_pos.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos[:, self.feet_indices, :], self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward


    ##
    ## jointangle
    def _reward_S_jointangle_mean(self):
        reward = self._similarity(self.target_joint_angles.episode_mean_buf,
                         self.policy_joint_angles.episode_mean_buf)
        diff = self._diff(self.dif_joint_angles, self.config.rewards.reward_tracking_sigma.teleop_joint_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_jointangle_variance(self):
        reward = self._similarity(self.target_joint_angles.episode_variance_buf,
                         self.policy_joint_angles.episode_variance_buf)
        diff = self._diff(self.dif_joint_angles, self.config.rewards.reward_tracking_sigma.teleop_joint_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_jointangle_covariance(self):
        reward = self._similarity(self.target_joint_angles.episode_covariance_buf.flatten(1),
                         self.policy_joint_angles.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_joint_angles, self.config.rewards.reward_tracking_sigma.teleop_joint_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _normal2pi(self, quat):
        import numpy as np
        angle, _ = quat_to_angle_axis(quat)
        angle = angle % (2 * np.pi)  # 归一化到[0, 2π]
        angle = torch.where(angle > np.pi,
                               2 * np.pi - angle,
                               angle)  # 取最小角度[0, π]
        return angle
    ##
    ## rot
    def _reward_S_rot_mean(self, eps=1e-6):
        dif_mean = quat_mul(self.target_body_rot.episode_mean_buf,
                            quat_conjugate(self.policy_body_rot.episode_mean_buf, w_last=True), w_last=True)
        dif_mean = self._normal2pi(dif_mean)

        diff_dist = (dif_mean**2)#.mean(dim=-1)

        norm_target = self._normal2pi(self.target_body_rot.episode_mean_buf)
        norm_policy = self._normal2pi(self.policy_body_rot.episode_mean_buf)
        norm_mean = ((norm_target + norm_policy) / 2.0 + eps)

        reward = torch.exp(-diff_dist / norm_mean)
        reward = torch.mean(reward, dim = -1)
        flags = self.episode_length_buf <= 2
        reward[flags] = 0
        return reward

    def _reward_S_rot_variance(self, eps=1e-6):
        dif_mean = self.target_body_rot.episode_variance_buf - self.policy_body_rot.episode_variance_buf

        diff_dist = dif_mean[..., 0]**2

        norm_mean = ((self.target_body_rot.episode_variance_buf + self.policy_body_rot.episode_variance_buf) / 2.0 + eps)

        reward = torch.exp(-diff_dist / norm_mean[..., 0])
        reward = torch.mean(reward, dim = -1)
        flags = self.episode_length_buf <= 2

        reward[flags] = 0

        return reward
