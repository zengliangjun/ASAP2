import torch
import torch.nn.functional as F
from . import motion_tracking
from humanoidverse.utils import statistics


class Tracking(motion_tracking.LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)

        joint_shape = (self.num_envs, self.dim_actions)
        self.target_joint_angles = statistics.MVCStatistics2(joint_shape, device, 5)
        self.policy_joint_angles = statistics.MVCStatistics2(joint_shape, device, 5)


        body_shape = (self.num_envs, self.num_bodies + self.num_extend_bodies, 3)
        self.target_body_pos = statistics.MVCStatistics2(body_shape, device, 5)
        #self.target_body_vel = statistics.MVStatistics(shape, device)
        #self.target_body_ang_vel = statistics.MVStatistics(shape, device) # dif_global_body_ang_vel

        self.policy_body_pos = statistics.MVCStatistics2(body_shape, device, 5)


        ##################################################
        self.pre_ref_body_pos_extend = torch.zeros(body_shape, device = device)
        #self.pre_ref_body_vel_extend = torch.zeros(body_shape, device = device)
        #self.pre_ref_body_ang_vel_extend = torch.zeros(body_shape, device = device)
        self.pre_ref_joint_pos = torch.zeros(joint_shape, device = device)
        #self.pre_ref_joint_vel = torch.zeros(joint_shape, device = device)

        # policy
        self.pre_rigid_body_pos_extend = torch.zeros(body_shape, device = device)
        #self.pre_rigid_body_vel_extend = torch.zeros(body_shape, device = device)
        #self.pre_rigid_body_ang_vel_extend = torch.zeros(body_shape, device = device)
        self.pre_joint_pos = torch.zeros(joint_shape, device = device)
        #self.pre_joint_vel = torch.zeros(joint_shape, device = device)

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

        target_diff_joint_angles[flags] = 0
        target_diff_body_pos[flags] = 0

        ## statistics
        self.target_joint_angles.update(target_diff_joint_angles)
        self.target_body_pos.update(target_diff_body_pos)

        # policy
        policy_diff_joint_angles = self.simulator.dof_pos - self.pre_joint_pos
        policy_diff_body_pos = self._rigid_body_pos_extend - self.pre_rigid_body_pos_extend

        policy_diff_joint_angles[flags] = 0
        policy_diff_body_pos[flags] = 0

        ## statistics
        self.policy_joint_angles.update(policy_diff_joint_angles)
        self.policy_body_pos.update(policy_diff_body_pos)


        ## update
        # target
        self.pre_ref_body_pos_extend[...] = self.ref_body_pos_extend
        #self.pre_ref_body_vel_extend[...] = self.ref_body_vel_extend
        #self.pre_ref_body_ang_vel_extend[...] = self.ref_body_ang_vel_extend
        self.pre_ref_joint_pos[...] = self.ref_joint_pos
        #self.pre_ref_joint_vel[...] = self.ref_joint_vel

        # policy
        self.pre_rigid_body_pos_extend[...] = self._rigid_body_pos_extend
        #self.pre_rigid_body_vel_extend[...] = self._rigid_body_vel_extend
        #self.pre_rigid_body_ang_vel_extend[...] = self._rigid_body_ang_vel_extend
        self.pre_joint_pos[...] = self.simulator.dof_pos
        #self.pre_joint_vel[...] = self.simulator.dof_vel


    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)

        self.target_joint_angles.reset2(env_ids)
        self.target_body_pos.reset2(env_ids)
        self.policy_joint_angles.reset2(env_ids)
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

    ##
    def _reward_S_bodypos_mean(self):
        reward = self._similarity(self.target_body_pos.episode_mean_buf.flatten(1),
                         self.policy_body_pos.episode_mean_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos, self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

        flags = self.episode_length_buf <= 2

        reward[flags] = diff[flags]

        return reward

    def _reward_S_bodypos_variance(self):
        reward = self._similarity(self.target_body_pos.episode_variance_buf.flatten(1),
                         self.policy_body_pos.episode_variance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos, self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

    def _reward_S_bodypos_covariance(self):
        reward = self._similarity(self.target_body_pos.episode_covariance_buf.flatten(1),
                         self.policy_body_pos.episode_covariance_buf.flatten(1))
        diff = self._diff(self.dif_global_body_pos, self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)

        flags = self.episode_length_buf <= 3

        reward[flags] = diff[flags]

        return reward

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
