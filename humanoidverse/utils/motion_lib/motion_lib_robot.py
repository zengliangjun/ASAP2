from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

class MotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        return

"""
motion data info:

global_velocity_extend torch.Size([119, 27, 3])
global_angular_velocity_extend torch.Size([119, 27, 3])
global_translation_extend torch.Size([119, 27, 3])
global_rotation_mat_extend torch.Size([119, 27, 3, 3])
global_rotation_extend torch.Size([119, 27, 4])
global_translation torch.Size([119, 24, 3])
global_rotation_mat torch.Size([119, 24, 3, 3])
global_rotation torch.Size([119, 24, 4])
local_rotation torch.Size([119, 27, 4])
global_root_velocity torch.Size([119, 3])
global_root_angular_velocity torch.Size([119, 3])
global_angular_velocity torch.Size([119, 24, 3])
global_velocity torch.Size([119, 24, 3])
dof_pos torch.Size([119, 23])
dof_vels torch.Size([119, 23])
"""
from .. import statistics
import easydict
import torch
from loguru import logger

class MotionStatistics(MotionLibRobot):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super(MotionStatistics, self).__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)


    def load_motions(self,
                     random_sample=True,
                     start_idx=0,
                     max_len=-1,
                     target_heading = None):

        motions = super(MotionStatistics, self).load_motions(random_sample=random_sample,
                     start_idx=start_idx,
                     max_len=max_len,
                     target_heading = target_heading)

        dof_pos_means = []
        dof_pos_variances = []
        dof_pos_covariances = []
        dof_vels_means = []
        dof_vels_variances = []
        dof_vels_covariances = []

        for id, m in enumerate(motions):
            edict: easydict.EasyDict = m

            if 0 == id:
                logger.info("motion infos")
                for key, value in edict.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"motion {key}:  {value.shape}")
                    else:
                        logger.info(f"motion {key}:  {value}")


            if not hasattr(self, "dof_pos_statistics"):
                self.dof_pos_statistics = statistics.Statistics((1, edict.dof_pos.shape[-1]), edict.dof_pos.device)

            if not hasattr(self, "dof_vels_statistics"):
                self.dof_vels_statistics = statistics.Statistics((1, edict.dof_vels.shape[-1]), edict.dof_vels.device)

            self.dof_pos_statistics.reset()
            self.dof_vels_statistics.reset()

            pos_mean, pos_variance, pos_covariance = \
                self.dof_pos_statistics.calculate(edict.dof_pos[None, ...])

            vels_mean, vels_variance, vels_covariance = \
            self.dof_vels_statistics.calculate(edict.dof_vels[None, ...])

            dof_pos_means.append(pos_mean[0])
            dof_pos_variances.append(pos_variance[0])
            dof_pos_covariances.append(pos_covariance[0])
            dof_vels_means.append(vels_mean[0])
            dof_vels_variances.append(vels_variance[0])
            dof_vels_covariances.append(vels_covariance[0])

        self.pos_means = torch.cat(dof_pos_means, dim=0).float().to(self._device)
        self.pos_variances = torch.cat(dof_pos_variances, dim=0).float().to(self._device)
        self.pos_covariances = torch.cat(dof_pos_covariances, dim=0).float().to(self._device)
        self.vels_means = torch.cat(dof_vels_means, dim=0).float().to(self._device)
        self.vels_variances = torch.cat(dof_vels_variances, dim=0).float().to(self._device)
        self.vels_covariances = torch.cat(dof_vels_covariances, dim=0).float().to(self._device)
        return motions

    def sample_time(self, motion_ids, truncate_time=None):
        return super(MotionStatistics, self).sample_time(motion_ids, truncate_time)

    def get_motion_length(self, motion_ids=None):
        return super(MotionStatistics, self).get_motion_length(motion_ids)

    def _blend_data(self, buffer, f0l, f1l, blend):
        b0 = buffer[f0l]
        b1 = buffer[f1l]
        return (1.0 - blend) * b0 + blend * b1

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        # dof_pos
        motion_res = super(MotionStatistics, self).get_motion_state(motion_ids, motion_times, offset)

        # copy from super(MotionStatistics, self).get_motion_state
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        motion_res["pos_mean"] = self._blend_data(self.pos_means, f0l, f1l, blend)
        motion_res["pos_variances"] = self._blend_data(self.pos_variances, f0l, f1l, blend)
        motion_res["pos_covariances"] = self._blend_data(self.pos_covariances, f0l, f1l, blend_exp)
        motion_res["vels_mean"] = self._blend_data(self.vels_means, f0l, f1l, blend)
        motion_res["vels_variances"] = self._blend_data(self.vels_variances, f0l, f1l, blend)
        motion_res["vels_covariances"] = self._blend_data(self.vels_covariances, f0l, f1l, blend_exp)

        return motion_res

