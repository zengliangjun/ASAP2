import torch
from . import motion_tracking
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionStatistics
from humanoidverse.utils import statistics


class Tracking(motion_tracking.LeggedRobotMotionTracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.dof_pos_statistics = statistics.Statistics(self.last_dof_pos.shape, device)
        self.dof_vels_statistics = statistics.Statistics(self.last_dof_vel.shape, device)


    def _init_motion_lib(self):
        """
        sample logic with LeggedRobotMotionTracking, replace MotionStatistics
        """
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionStatistics(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        else:
            self._motion_lib.load_motions(random_sample=True)

        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions


    # _post_physics_step
    ## _pre_compute_observations_callback
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        #
        # motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times


    ## reset_envs_idx
    ### _reset_robot_states_callback
    #### _reset_dofs
    def _reset_dofs(self, env_ids):
        super()._reset_dofs(env_ids)
        # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times


    #### _reset_root_states
    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times


