import torch
import genesis as gs
import gstaichi as ti
import genesis.utils.geom as gu

from genesis.engine.entities import RodEntity
from genesis.utils.misc import to_gs_tensor

class RobotController:
    def __init__(
        self,
        scene,
        robot,
        ef,
        configs,
        initial_pos=(0., 0., 0.),
        initial_quat=(0., 1., 0., 0.),
        initial_gripper_gap=0.03,
        n_motors_dofs=7,
        n_fingers_dofs=2,
        debug=False,
    ):
        self.scene = scene
        self.robot = robot
        self.ef = ef
        self.configs = configs
        self.motors_dof = torch.arange(n_motors_dofs)
        self.fingers_dof = torch.arange(n_motors_dofs, n_motors_dofs + n_fingers_dofs)
        self.debug = debug
        self.debug_point_nodes = list()
        self.initial_pos = initial_pos
        self.initial_quat = initial_quat
        self.init_gap = initial_gripper_gap

    def set_initial_position(self):
        pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
        quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)

        is_batched = self.configs.n_envs > 0
        self.pos_abs = torch.stack([pos_abs] * self.configs.n_envs) if is_batched else pos_abs
        self.quat_abs = torch.stack([quat_abs] * self.configs.n_envs) if is_batched else quat_abs
        qpos = self.robot.inverse_kinematics(
            link=self.ef,
            pos=self.pos_abs,
            quat=self.quat_abs,
        )
        qpos[..., -2:] = self.init_gap  # initial gripper open
        self.robot.set_dofs_position(qpos)

    def control_robot(
        self, g_dof1, g_dof2,
        dx=0., dy=0., dz=0., di=0., dj=0., dk=0.,
        g_dof_use_force=False, degrees=True, **kwargs
    ):
        """
        Controls the robot's end-effector to move by specified deltas in position and orientation.
        """
        if isinstance(dx, (float, int)) and isinstance(dy, (float, int)) and isinstance(dz, (float, int)):
            delta_pos = torch.tensor([dx, dy, dz], dtype=gs.tc_float)
        else:
            delta_pos = torch.stack([dx, dy, dz], dim=-1)
        target_pos = self.pos_abs + delta_pos
        if isinstance(di, (float, int)) and isinstance(dj, (float, int)) and isinstance(dk, (float, int)):
            delta_orient = torch.tensor([di, dj, dk], dtype=gs.tc_float)
        else:
            delta_orient = torch.stack([di, dj, dk], dim=-1)
        delta_quat = gu.xyz_to_quat(delta_orient, rpy=True, degrees=degrees)
        if delta_quat.ndim == 1 and self.quat_abs.ndim == 2:
            delta_quat = torch.stack([delta_quat] * self.configs.n_envs) if self.configs.n_envs > 0 else delta_quat
        else:
            raise ValueError("`delta_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(delta_quat, self.quat_abs)

        self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

    def rotate_around_point(
        self, g_dof1, g_dof2, center, axis, angle, pos_angle=None,
        g_dof_use_force=False, degrees=True, **kwargs
    ):
        """
        Rotates the robot's end-effector around a specified world-space point.
        TODO: this func currently only supports single env.
        """
        center_tensor = torch.as_tensor(center, dtype=gs.tc_float)
        axis_tensor = torch.as_tensor(axis, dtype=gs.tc_float)

        position_angle = angle if pos_angle is None else pos_angle

        angle_tensor = torch.tensor(angle, dtype=gs.tc_float)
        pos_angle_tensor = torch.tensor(position_angle, dtype=gs.tc_float)

        orient_angle_rad = torch.deg2rad(angle_tensor) if degrees else angle_tensor
        pos_angle_rad = torch.deg2rad(pos_angle_tensor) if degrees else pos_angle_tensor

        orient_rotation_quat = gu.axis_angle_to_quat(orient_angle_rad, axis_tensor)
        pos_rotation_quat = gu.axis_angle_to_quat(pos_angle_rad, axis_tensor)

        vec_to_pos = self.pos_abs - center_tensor
        rotated_vec = gu.transform_by_quat(vec_to_pos, pos_rotation_quat)
        target_pos = center_tensor + rotated_vec

        if orient_rotation_quat.ndim == 1 and self.quat_abs.ndim == 2:
            orient_rotation_quat = torch.stack([orient_rotation_quat] * self.configs.n_envs) if self.configs.n_envs > 0 else orient_rotation_quat
        else:
            raise ValueError("`orient_rotation_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(orient_rotation_quat, self.quat_abs)

        self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

    def _execute_ik_control(self, target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs):
        """
        Run inverse kinematics and send control commands.
        """
        is_batched = self.configs.n_envs > 0
        pos_arg = target_pos
        quat_arg = target_quat
        gripper_arg = torch.tensor([[g_dof1, g_dof2]] * self.configs.n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])

        if self.debug:
            for i in self.debug_point_nodes:
                self.scene.clear_debug_object(i)
            self.debug_point_nodes = list()
            for batch_idx in range(self.configs.n_envs if is_batched else 1):
                if is_batched:
                    offset = self.scene.envs_offset[batch_idx]
                    offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
                self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                    pos=target_pos[batch_idx] + offset if is_batched else target_pos,
                    radius=0.01
                ))

        qpos = self.robot.inverse_kinematics(
            link=self.ef,
            pos=pos_arg,
            quat=quat_arg,
            **kwargs
        )

        self.robot.control_dofs_position(qpos[..., :-2], self.motors_dof)

        if g_dof_use_force:
            self.robot.control_dofs_force(gripper_arg, self.fingers_dof)
        else:
            self.robot.control_dofs_position(gripper_arg, self.fingers_dof)

        self.pos_abs = target_pos
        self.quat_abs = target_quat


class RobotControllerOptim(RobotController):
    def __init__(
        self,
        scene,
        robot,
        ef,
        configs,
        initial_pos=(0., 0., 0.),
        initial_quat=(0., 1., 0., 0.),
        initial_q_dof=0.03,
        n_motors_dofs=7,
        n_fingers_dofs=2,
        n_stages=10,
        n_optim_dofs=6,
        max_d_pos=0.05,
        max_d_angle=10.,
        debug=False,
    ):
        super().__init__(
            scene, robot, ef, configs,
            initial_pos, initial_quat, initial_q_dof,
            n_motors_dofs, n_fingers_dofs, debug
        )

        self.traj = torch.zeros(
            size=(self.scene.n_envs, n_stages, n_optim_dofs), dtype=gs.tc_float
        )
        self.n_stages = n_stages
        self.n_optim_dofs = n_optim_dofs
        self.max_d_pos = max_d_pos
        self.max_d_angle = max_d_angle

    def apply_grad(self, g_dof1, g_dof2, g_dof_use_force=False, stage_idx=None):
        # print(f'stage_idx: {stage_idx}, dx: {self.traj[:, stage_idx, 0].shape}')
        self.control_robot(
            g_dof1, g_dof2, g_dof_use_force=g_dof_use_force,
            dx=self.traj[:, stage_idx, 0],
            dy=self.traj[:, stage_idx, 1],
            dz=self.traj[:, stage_idx, 2],
            di=self.traj[:, stage_idx, 3],
            dj=self.traj[:, stage_idx, 4],
            dk=self.traj[:, stage_idx, 5],
        )

    def gather_grad(self, grad, pos, active_grad_ids, stage_idx, lr=0.01):
        # [n_envs, 3]
        contact_grad = grad[:, active_grad_ids, :].sum(dim=1)

        # [n_envs, 3]
        contact_pos = pos[:, active_grad_ids, :].mean(dim=1)

        d_pos = -lr * contact_grad
        d_pos = torch.clamp(d_pos, -self.max_d_pos, self.max_d_pos)
        # print(f'stage_idx: {stage_idx}, dpos x: {d_pos[0]}')

        # TODO: not well tested for torque control
        d_torque = torch.linalg.cross(
            pos[:, active_grad_ids, :] - contact_pos[:, None, :],
            grad[:, active_grad_ids, :], dim=-1
        )
        t_torque = d_torque.sum(dim=1)
        d_rad = -lr * t_torque
        d_angle = torch.rad2deg(d_rad)
        d_angle = torch.clamp(d_angle, -self.max_d_angle, self.max_d_angle)

        d_dof = torch.cat([d_pos, d_angle], dim=-1)
        self.traj[:, stage_idx, :] += d_dof


class DLOControllerOptim:
    def __init__(
        self,
        scene,
        rod: RodEntity,
        grasp_point_ids,
        n_stages=10,
        n_optim_dofs=3,
        max_ddist=0.05,
        use_adam=False,
        adam_config=None,
        debug=False
    ):
        self.scene = scene
        self.rod = rod
        self.grasp_point_ids = grasp_point_ids
        self.n_grasp_points = len(grasp_point_ids)
        
        self.traj = torch.zeros(
            size=(self.scene.n_envs, n_stages, self.n_grasp_points, n_optim_dofs), dtype=gs.tc_float
        )

        # for Adam optimizer
        self.use_adam = use_adam
        if self.use_adam:
            self.m_buffer = torch.zeros_like(self.traj)
            self.v_buffer = torch.zeros_like(self.traj)
            self.iter = 0
            if adam_config is None:
                adam_config = {
                    "beta1": 0.9,
                    "beta2": 0.99,
                    "eps": 1e-8,
                }
            else:
                if "beta1" not in adam_config:
                    adam_config["beta1"] = 0.9
                if "beta2" not in adam_config:
                    adam_config["beta2"] = 0.99
                if "eps" not in adam_config:
                    adam_config["eps"] = 1e-8
            self.adam_config = adam_config

        self.n_stages = n_stages
        self.n_optim_dofs = n_optim_dofs
        self.max_ddist = max_ddist

        self.debug = debug
        self.debug_point_nodes = list()

    def apply_grad(self, stage_idx):
        expected_pos = self.rod.get_all_verts()
        expected_pos = torch.tensor(expected_pos, dtype=self.traj.dtype, device=self.traj.device)
        dpos = self.traj[:, stage_idx, :, :]
        ddist = torch.linalg.norm(dpos, dim=-1)
        weight = self.max_ddist / (ddist + gs.EPS)
        dpos_ = dpos * torch.minimum(weight, torch.ones_like(weight))[:, :, None]
        # dpos_ = torch.clamp(dpos, -self.max_ddist, self.max_ddist)
        expected_pos[:, self.grasp_point_ids, :] = expected_pos[:, self.grasp_point_ids, :] + dpos_

        for batch_idx in range(self.scene.n_envs):
            offset = self.scene.envs_offset[batch_idx]
            offset = torch.as_tensor(offset, dtype=self.traj.dtype, device=self.traj.device)
            if self.debug:
                for i in self.debug_point_nodes:
                    self.scene.clear_debug_object(i)
                self.debug_point_nodes = list()
                for i in self.grasp_point_ids:
                    self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                        pos=expected_pos[batch_idx, i] + offset,
                        radius=0.01
                    ))

        # if stage_idx == self.n_stages - 1:
        #     print(f'stage_idx: {stage_idx}, expected_pos {expected_pos.shape}')
        #     print('original pos')
        #     print(self.rod.get_all_verts()[0, self.grasp_point_ids, :])
        #     print(f'expected pos after adding dpos:')
        #     print(expected_pos[batch_idx, self.grasp_point_ids, :])
        #     print('dpos before clamp:')
        #     print(dpos)
        #     print('dpos after clamp:')
        #     print(dpos_)
        self.rod.set_position(expected_pos)

    def pre_apply_grad(self, stage_idx, num_horizons=1):
        # compute delta pos for this stage, and then distribute to each horizon step within this stage
        dpos = self.traj[:, stage_idx, :, :]
        ddist = torch.linalg.norm(dpos, dim=-1)
        weight = self.max_ddist / (ddist + gs.EPS)
        dpos_ = dpos * torch.minimum(weight, torch.ones_like(weight))[:, :, None]
        # print(f'dpos before clamp: {dpos}')
        # print(f'dpos after clamp: {dpos_}')

        expected_pos = self.rod.get_all_verts_tc()

        if self.debug:
            debug_pos = expected_pos[:, self.grasp_point_ids, :] + dpos_
            debug_pos = debug_pos.clone().detach().cpu().numpy()
            for batch_idx in range(self.scene.n_envs):
                offset = self.scene.envs_offset[batch_idx]
                for i in self.debug_point_nodes:
                    self.scene.clear_debug_object(i)
                self.debug_point_nodes = list()
                for i in range(self.n_grasp_points):
                    self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                        pos=debug_pos[batch_idx, i] + offset,
                        radius=0.016,
                        color=(1.0, 0.0, 0.0, 0.6)
                    ))

        target_pos_list = list()
        current_pos = expected_pos.clone()
        for i in range(num_horizons):
            alpha = (i + 1) / num_horizons
            expected_pos[:, self.grasp_point_ids, :] = current_pos[:, self.grasp_point_ids, :] + alpha * dpos_
            target_pos_list.append(expected_pos)

        return target_pos_list, dpos_

    def on_apply_grad(self, target_pos):
        self.rod.set_position(target_pos)

    def gather_grad(self, stage_idx, horizon_idx, lr=0.01):
        grad = self.rod._queried_states[horizon_idx][0].pos.grad

        # [n_envs, n_grasp_points, 3]
        contact_grad = grad[:, self.grasp_point_ids, :]
        # replace NaN or Inf with 0
        contact_grad = torch.where(torch.isnan(contact_grad), torch.zeros_like(contact_grad), contact_grad)
        contact_grad = torch.where(torch.isinf(contact_grad), torch.zeros_like(contact_grad), contact_grad)

        if self.use_adam:
            # Adam
            beta1 = self.adam_config["beta1"]
            beta2 = self.adam_config["beta2"]
            eps = self.adam_config["eps"]

            m_t = beta1 * self.m_buffer[:, stage_idx, :, :] + (1 - beta1) * contact_grad
            v_t = beta2 * self.v_buffer[:, stage_idx, :, :] + (1 - beta2) * (contact_grad ** 2)
            self.m_buffer[:, stage_idx, :, :] = m_t
            self.v_buffer[:, stage_idx, :, :] = v_t

            m_cap = m_t / (1 - beta1 ** (self.iter + 1))
            v_cap = v_t / (1 - beta2 ** (self.iter + 1))
            
            d_pos = -lr * m_cap / (torch.sqrt(v_cap) + eps)
            self.iter += 1
        else:
            # SGD
            d_pos = -lr * contact_grad

        self.traj[:, stage_idx, :, :] += d_pos
