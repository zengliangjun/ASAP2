import torch
from typing import Union

from isaac_utils import rotations

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


##
class MVQuat(MVStatistics):

    def __init__(self, shape: Union[tuple, torch.Size], device, episode_truncation = -1):
        super(MVQuat, self).__init__(shape, device, episode_truncation)
        vshape = (*shape[:-1], 1)
        self.episode_variance_buf = torch.zeros(vshape, device=device)
        self.episode_mean_buf[..., -1] = 1


    def update(self, input):
        step = self._calcute_step()
        step_e1 = step[:, None]
        while len(step_e1.shape) != len(self.episode_mean_buf.shape):
            step_e1 = step_e1[:, None]

        # 将当前均值四元数转换为切空间向量
        current_mean_quat = rotations.quat_normalize(self.episode_mean_buf)
        current_mean_log = rotations.quat_to_exp_map(current_mean_quat)

        # 将输入四元数转换到均值切空间
        input_quat = rotations.quat_normalize(input)
        relative_quat = rotations.quat_mul_norm(
            rotations.quat_inverse(current_mean_quat, w_last=True),
            input_quat,
            w_last=True
        )
        input_log = rotations.quat_to_exp_map(relative_quat)

        # 在切空间进行增量式均值更新
        delta0 = input_log - current_mean_log
        new_mean_log = current_mean_log + delta0 / step_e1

        # 计算方差(在切空间)
        diff_norm_sq = torch.sum(delta0**2, dim=-1, keepdim=True)

        delta = diff_norm_sq - self.episode_variance_buf
        self.episode_variance_buf += delta / step_e1

        # step == 2
        mask2 = step == 2
        self.episode_variance_buf[mask2] = diff_norm_sq[mask2]

        # 将新均值转换回四元数空间
        angle = torch.norm(new_mean_log, dim=-1, keepdim=True)
        axis = new_mean_log / (angle + 1e-8)
        delta_quat = rotations.quat_from_angle_axis(angle.squeeze(-1), axis, w_last=True)
        self.episode_mean_buf = rotations.quat_mul_norm(
            current_mean_quat,
            delta_quat,
            w_last=True
        )

        # 初始情况处理
        mask = step <= 1
        self.episode_mean_buf[mask] = input[mask]
        self.episode_variance_buf[mask] = 0

    def reset2(self, env_ids):
        self.current_step[env_ids] = 0

        self.episode_mean_buf[env_ids, ..., : -1] =0
        self.episode_mean_buf[env_ids, ..., -1] =1
        self.episode_variance_buf[env_ids] = 0



class MVCStatistics(MVStatistics):

    def __init__(self, shape: tuple, device, episode_truncation = -1):
        super(MVCStatistics, self).__init__(shape, device, episode_truncation)

        covshape = (shape[0], shape[1], shape[1], *shape[2:])

        self.episode_covariance_buf = torch.zeros(covshape,
                                                    device=device, dtype=torch.float)

    def clean(self):
        super(MVCStatistics, self).clean()
        self.episode_covariance_buf[:] = 0

    def update(self, input):

        step = self._calcute_step()
        step_e1 = step[:, None]

        while len(step_e1.shape) != len(self.episode_variance_buf.shape):
            step_e1 = step_e1[..., None]

        step_e2 = step_e1[..., None]
        while len(step_e2.shape) != len(self.episode_covariance_buf.shape):
            step_e2 = step_e2[..., None]

        # 计算均值：根据新差值delta0更新均值缓冲区
        delta0 = input - self.episode_mean_buf
        self.episode_mean_buf += delta0 / step_e1

        # 计算方差：利用delta0和新均值计算更新方差缓冲区
        delta1 = input - self.episode_mean_buf
        self.episode_variance_buf = (
            self.episode_variance_buf * (step_e1 - 2)
            + delta0 * delta1
        ) / (step_e1 - 1)

        if len(delta1.shape) > 2:
            orgshape = delta1.shape
            delta0 = torch.reshape(delta0, (*orgshape[:2], -1))
            delta1 = torch.reshape(delta1, (*orgshape[:2], -1))
            covariance = torch.einsum("bik,bjk->bijk", delta1, delta0)
            covariance = torch.reshape(covariance, (*orgshape[:2], orgshape[1], *orgshape[2:]))

        else:
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

class MVStatistics2(MVStatistics):

    def __init__(self, shape: Union[tuple, torch.Size], device, episode_truncation = -1):
        super(MVStatistics2, self).__init__(shape, device, episode_truncation)

    def update(self, input):
        super(MVStatistics2, self).update(input)

        mask = self.current_step <= 1
        self.episode_mean_buf[mask] = 0

        mask = self.current_step <= 2
        self.episode_variance_buf[mask] = 0

##
class MVQuat2(MVQuat):

    def __init__(self, shape: Union[tuple, torch.Size], device, episode_truncation = -1):
        super(MVQuat2, self).__init__(shape, device, episode_truncation)

    def update(self, input):
        super(MVQuat2, self).update(input)

        mask = self.current_step <= 2
        self.episode_mean_buf[mask] = input[mask]
        self.episode_variance_buf[mask] = 0

        mask = self.current_step <= 1
        self.episode_mean_buf[mask, ..., : -1] = 0
        self.episode_mean_buf[mask, ..., -1] = 1


class MVCStatistics2(MVCStatistics):

    def __init__(self, shape: tuple, device, episode_truncation = -1):
        super(MVCStatistics2, self).__init__(shape, device, episode_truncation)

    def update(self, input):
        super(MVCStatistics2, self).update(input)

        mask = self.current_step <= 1
        self.episode_mean_buf[mask] = 0

        mask = self.current_step <= 2
        self.episode_variance_buf[mask] = 0
        self.episode_covariance_buf[mask] = 0
