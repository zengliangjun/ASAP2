from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
class MotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        return


"""
Motion info

dof_vel : torch.Size([1, 23])                         ***
dof_pos : torch.Size([1, 23])                         ***


root_rot : torch.Size([1, 4])
root_pos : torch.Size([1, 3])
root_vel : torch.Size([1, 3])
root_ang_vel : torch.Size([1, 3])

motion_aa : torch.Size([1, 72])
motion_bodies : torch.Size([1, 17])

body_vel : torch.Size([1, 24, 3])
body_ang_vel : torch.Size([1, 24, 3])

rg_pos : torch.Size([1, 24, 3])
rg_pos_t : torch.Size([1, 27, 3])                        ***    gts_t

body_vel_t : torch.Size([1, 27, 3])                      ***    gvs_t
body_ang_vel_t : torch.Size([1, 27, 3])                  ***    gavs_t

rb_rot : torch.Size([1, 24, 4])
rg_rot_t : torch.Size([1, 27, 4])

"""




"""
motion data info:

global_velocity_extend torch.Size([119, 27, 3])           ***    gvs_t
global_angular_velocity_extend torch.Size([119, 27, 3])   ***    gavs_t
global_translation_extend torch.Size([119, 27, 3])        ***    gts_t
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
