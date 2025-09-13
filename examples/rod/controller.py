import torch
import genesis as gs
import genesis.utils.geom as gu


class RobotController:
    def __init__(
        self,
        robot,
        ef,
        configs,
        initial_pos=(0., 0., 0.),
        initial_quat=(0., 1., 0., 0.),
        initial_q_dof=0.03,
        n_motors_dofs=7,
        n_fingers_dofs=2
    ):
        self.robot = robot
        self.ef = ef
        self.configs = configs
        self.motors_dof = torch.arange(n_motors_dofs)
        self.fingers_dof = torch.arange(n_motors_dofs, n_motors_dofs + n_fingers_dofs)

        self.pos_abs = torch.tensor(initial_pos, dtype=gs.tc_float)
        self.quat_abs = torch.tensor(initial_quat, dtype=gs.tc_float)

        is_batched = self.configs.n_envs > 0
        pos_arg = torch.stack([self.pos_abs] * self.configs.n_envs) if is_batched else self.pos_abs
        quat_arg = torch.stack([self.quat_abs] * self.configs.n_envs) if is_batched else self.quat_abs
        qpos = self.robot.inverse_kinematics(
            link=self.ef,
            pos=pos_arg,
            quat=quat_arg,
        )
        qpos[..., -2:] = initial_q_dof  # initial gripper open
        self.robot.set_dofs_position(qpos)

    def control_robot(
        self, g_dof1, g_dof2,
        dx=0., dy=0., dz=0., di=0., dj=0., dk=0.,
        g_dof_use_force=False, degrees=True
    ):
        """
        Controls the robot's end-effector to move by specified deltas in position and orientation.
        """
        target_pos = self.pos_abs + torch.tensor([dx, dy, dz], dtype=gs.tc_float)
        delta_orient = torch.tensor([di, dj, dk], dtype=gs.tc_float)
        delta_quat = gu.xyz_to_quat(delta_orient, rpy=True, degrees=degrees)
        target_quat = gu.transform_quat_by_quat(delta_quat, self.quat_abs)

        self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force)

    def rotate_around_point(
        self, g_dof1, g_dof2, center, axis, angle, pos_angle=None,
        g_dof_use_force=False, degrees=True
    ):
        """
        Rotates the robot's end-effector around a specified world-space point.
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

        target_quat = gu.transform_quat_by_quat(orient_rotation_quat, self.quat_abs)

        self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force)

    def _execute_ik_control(self, target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force):
        """
        Run inverse kinematics and send control commands.
        """
        is_batched = self.configs.n_envs > 0
        pos_arg = torch.stack([target_pos] * self.configs.n_envs) if is_batched else target_pos
        quat_arg = torch.stack([target_quat] * self.configs.n_envs) if is_batched else target_quat
        gripper_arg = torch.tensor([[g_dof1, g_dof2]] * self.configs.n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])

        qpos = self.robot.inverse_kinematics(
            link=self.ef,
            pos=pos_arg,
            quat=quat_arg,
        )

        self.robot.control_dofs_position(qpos[..., :-2], self.motors_dof)

        if g_dof_use_force:
            self.robot.control_dofs_force(gripper_arg, self.fingers_dof)
        else:
            self.robot.control_dofs_position(gripper_arg, self.fingers_dof)

        self.pos_abs = target_pos
        self.quat_abs = target_quat
