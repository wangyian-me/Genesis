import torch
import genesis as gs
import gstaichi as ti
import genesis.utils.geom as gu

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
import qpsolvers

from genesis.engine.entities import RodEntity


def rod_attached_to_gripper(scene, contact_info, rod, ef, geom_indices, envs_idx=0):
    info = scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=envs_idx)
    contact_info.update(info)
    attach_verts = list(contact_info.keys())
    for v in attach_verts:
        if contact_info[v] in geom_indices:
            rod.attach_to_rigid_link_with_envs_idx(ef, [v], envs_idx=envs_idx)

def rod_vertex_attached_to_gripper(rod, vert_idx, ef, envs_idx=0):
    rod.attach_to_rigid_link_with_envs_idx(ef, [vert_idx], envs_idx=envs_idx)

def rod_detached_from_gripper(contact_info, rod, geom_indices, envs_idx=0):
    vertices_to_detach = list()
    for i_v, i_g in contact_info.items():
        if i_g in geom_indices:
            rod.detach_from_rigid_link_with_envs_idx([i_v], envs_idx=envs_idx)
            vertices_to_detach.append(i_v)
    for i_v in vertices_to_detach:
        contact_info.pop(i_v)

def rod_vertex_detached_from_gripper(rod, vert_idx, envs_idx=0):
    rod.detach_from_rigid_link_with_envs_idx([vert_idx], envs_idx=envs_idx)

class RobotController:
    def __init__(
        self,
        scene,
        robot,
        ef,
        configs=None,
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

        is_batched = self.scene.n_envs > 0
        self.pos_abs = torch.stack([pos_abs] * self.scene.n_envs) if is_batched else pos_abs
        self.quat_abs = torch.stack([quat_abs] * self.scene.n_envs) if is_batched else quat_abs
        qpos = self.robot.inverse_kinematics(
            link=self.ef,
            pos=self.pos_abs,
            quat=self.quat_abs,
        )
        qpos[..., -2:] = self.init_gap  # initial gripper open
        self.robot.set_dofs_position(qpos)

        return qpos

    def set_initial_dofs_position(self, qpos, use_initial_pq=True):
        is_batched = self.scene.n_envs > 0
        if is_batched:
            qpos = torch.stack([qpos] * self.scene.n_envs)
        self.robot.set_dofs_position(qpos)
        if use_initial_pq:
            pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
            quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)
            self.pos_abs = torch.stack([pos_abs] * self.scene.n_envs) if is_batched else pos_abs
            self.quat_abs = torch.stack([quat_abs] * self.scene.n_envs) if is_batched else quat_abs
        else:
            self.pos_abs = self.ef.get_pos()
            self.quat_abs = self.ef.get_quat()

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
        if kwargs.get('min_z', None) is not None:
            min_z = torch.zeros_like(target_pos[..., 2])
            min_z.fill_(kwargs.pop('min_z'))
            kwargs.update({'underground': (target_pos[..., 2] < min_z).any()})
            target_pos[..., 2] = torch.maximum(target_pos[..., 2], min_z)
        if isinstance(di, (float, int)) and isinstance(dj, (float, int)) and isinstance(dk, (float, int)):
            delta_orient = torch.tensor([di, dj, dk], dtype=gs.tc_float)
        else:
            delta_orient = torch.stack([di, dj, dk], dim=-1)
        delta_quat = gu.xyz_to_quat(delta_orient, rpy=True, degrees=degrees)
        if delta_quat.ndim == 1 and self.quat_abs.ndim == 2:
            delta_quat = torch.stack([delta_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else delta_quat
        elif delta_quat.ndim != self.quat_abs.ndim:
            raise ValueError("`delta_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(delta_quat, self.quat_abs)

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

        return qpos

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
            orient_rotation_quat = torch.stack([orient_rotation_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else orient_rotation_quat
        elif orient_rotation_quat.ndim != self.quat_abs.ndim:
            raise ValueError("`orient_rotation_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(orient_rotation_quat, self.quat_abs)

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

        return qpos

    def _execute_ik_control(self, target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs):
        """
        Run inverse kinematics and send control commands.
        """
        is_batched = self.scene.n_envs > 0
        pos_arg = target_pos
        quat_arg = target_quat
        gripper_arg = torch.tensor([[g_dof1, g_dof2]] * self.scene.n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])
        underground = kwargs.pop('underground', False)

        if self.debug:
            for i in self.debug_point_nodes:
                self.scene.clear_debug_object(i)
            self.debug_point_nodes = list()
            for batch_idx in range(self.scene.n_envs if is_batched else 1):
                if is_batched:
                    offset = self.scene.envs_offset[batch_idx]
                    offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
                color = (1.0, 1.0, 0.0, 0.6) if underground else (0.0, 1.0, 0.0, 0.6)
                self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                    pos=target_pos[batch_idx] + offset if is_batched else target_pos,
                    radius=0.01,
                    color=color
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

        return qpos

class RobotControllerPink:
    """
    Robot controller using Pink-based inverse kinematics.

    This is a drop-in replacement for RobotController that uses Pink IK
    instead of Genesis's built-in IK solver.
    """

    def __init__(
        self,
        scene,
        robot,
        ef,
        configs=None,
        initial_pos=(0., 0., 0.),
        initial_quat=(0., 1., 0., 0.),
        initial_gripper_gap=0.03,
        n_motors_dofs=7,
        n_fingers_dofs=2,
        urdf_path=None,
        ee_frame_name=None,
        debug=False,
        # Pink-specific parameters
        pink_max_iterations=100,
        pink_dt=0.04,
        pink_solver="proxqp",
        pink_position_cost=10.0,
        pink_orientation_cost=10.0,
        pink_posture_cost=1e-3,
    ):
        """
        Initialize Pink-based robot controller.

        Parameters
        ----------
        scene : gs.Scene
            Genesis scene object
        robot : RigidEntity
            Robot entity (e.g., Franka Panda)
        ef : RigidLink
            End-effector link
        configs : optional
            Configuration dictionary (not used currently)
        initial_pos : tuple, optional
            Initial end-effector position (x, y, z)
        initial_quat : tuple, optional
            Initial end-effector quaternion (w, x, y, z)
        initial_gripper_gap : float, optional
            Initial gripper opening width
        n_motors_dofs : int, optional
            Number of motor DOFs (default: 7 for Franka)
        n_fingers_dofs : int, optional
            Number of finger DOFs (default: 2 for Franka gripper)
        urdf_path : str, optional
            Path to robot URDF file. If None, defaults to `robot.morph.file`
        ee_frame_name : str, optional
            Name of end-effector frame in Pinocchio model. If None, defaults to `ef.name`
        debug : bool, optional
            Enable debug visualization
        pink_max_iterations : int, optional
            Maximum iterations for Pink IK solver (default: 100)
        pink_dt : float, optional
            Integration timestep for Pink IK (default: 0.04)
        pink_solver : str, optional
            QP solver to use: "proxqp", "quadprog", etc. (default: "proxqp")
        pink_position_cost : float, optional
            Position task cost weight (default: 10.0)
        pink_orientation_cost : float, optional
            Orientation task cost weight (default: 10.0)
        pink_posture_cost : float, optional
            Posture regularization cost weight (default: 1e-3)
        """

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

        # Pink-specific attributes
        if urdf_path is None:
            urdf_path = robot.morph.file
        self.urdf_path = urdf_path
        if ee_frame_name is None:
            ee_frame_name = ef.name
        self.ee_frame_name = ee_frame_name

        # Get robot base position in world frame (Pinocchio assumes robot at origin)
        # We need this to convert between world and robot-local coordinates
        self.robot_base_pos = np.array(robot.morph.pos, dtype=gs.np_float) if hasattr(robot.morph, 'pos') else np.zeros(3, dtype=gs.np_float)
        self.robot_base_quat = np.array(robot.morph.quat, dtype=gs.np_float) if hasattr(robot.morph, 'quat') else np.array([1, 0, 0, 0], dtype=gs.np_float)
        self.pink_max_iterations = pink_max_iterations
        self.pink_dt = pink_dt
        self.pink_solver = pink_solver
        self.pink_position_cost = pink_position_cost
        self.pink_orientation_cost = pink_orientation_cost
        self.pink_posture_cost = pink_posture_cost

        # Load Pinocchio model and create Configuration
        self._load_pinocchio_model()

    def _load_pinocchio_model(self):
        """Load the robot model with Pinocchio and create Configuration."""
        try:
            # Build model from URDF using RobotWrapper for better path handling
            import pathlib
            urdf_path_obj = pathlib.Path(self.urdf_path)

            # Use pin_robot to avoid overwriting self.robot (Genesis entity)
            self.pin_robot = pin.RobotWrapper.BuildFromURDF(
                filename=urdf_path_obj.as_posix(),
                package_dirs=[".", urdf_path_obj.parent.as_posix()],
                root_joint=None,
            )

            self.robot_model = self.pin_robot.model
            self.robot_data = self.pin_robot.data

            # Verify end-effector frame exists
            if not self.robot_model.existFrame(self.ee_frame_name):
                available_frames = [
                    self.robot_model.frames[i].name
                    for i in range(len(self.robot_model.frames))
                ]
                raise ValueError(
                    f"End-effector frame '{self.ee_frame_name}' not found in model. "
                    f"Available frames: {available_frames}"
                )

            # Create Pink Configuration object (will be updated with current qpos before each IK call)
            initial_q = pin.neutral(self.robot_model)
            self.configuration = pink.Configuration(self.robot_model, self.robot_data, initial_q)

            if self.debug:
                gs.logger.debug(f"Pink IK: Loaded Pinocchio model from {self.urdf_path}")
                gs.logger.debug(f"  Model has {self.robot_model.nq} position DOFs, {self.robot_model.nv} velocity DOFs")
                gs.logger.debug(f"  End-effector frame: {self.ee_frame_name}")
                gs.logger.debug(f"  Genesis EF link name: {self.ef.name}")
                gs.logger.debug(f"  Robot base position (world): {self.robot_base_pos}")
                gs.logger.debug(f"  Robot base quaternion (world): {self.robot_base_quat}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Pinocchio model from {self.urdf_path}: {e}\n"
                "Make sure the URDF path is correct and accessible."
            )

    def set_initial_position(self):
        """
        Set robot to initial end-effector position using Pink IK.

        Returns
        -------
        qpos : torch.Tensor
            Joint positions including gripper
        """
        pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
        quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)

        is_batched = self.scene.n_envs > 0
        self.pos_abs = torch.stack([pos_abs] * self.scene.n_envs) if is_batched else pos_abs
        self.quat_abs = torch.stack([quat_abs] * self.scene.n_envs) if is_batched else quat_abs

        # Get current qpos as initial guess
        current_qpos = self.robot.get_qpos()

        if is_batched:
            # For batched environments, solve IK for the first environment
            # and replicate to all environments
            # Pass full qpos (including gripper) to Pink
            qpos_full = self._solve_pink_ik(
                pos=self.pos_abs[0].cpu().numpy(),
                quat=self.quat_abs[0].cpu().numpy(),
                init_qpos=current_qpos[0].cpu().numpy(),  # Full 9 DOFs
            )
            qpos_full = torch.tensor(qpos_full, dtype=gs.tc_float)
            qpos_full = torch.stack([qpos_full] * self.scene.n_envs)

            # Set gripper to initial gap
            qpos_full[..., -2:] = self.init_gap
            qpos = qpos_full
        else:
            # Single environment
            # Pass full qpos (including gripper) to Pink
            qpos_full = self._solve_pink_ik(
                pos=self.pos_abs.cpu().numpy(),
                quat=self.quat_abs.cpu().numpy(),
                init_qpos=current_qpos.cpu().numpy(),  # Full 9 DOFs
            )
            qpos = torch.tensor(qpos_full, dtype=gs.tc_float)

            # Set gripper to initial gap
            qpos[-2:] = self.init_gap

        self.robot.set_dofs_position(qpos)

        # # Verify the solution - step scene to update Genesis forward kinematics
        # self.scene.step()
        # actual_pos = self.ef.get_pos()

        # if is_batched:
        #     target_pos = self.pos_abs[0].cpu().numpy()
        #     achieved_pos = actual_pos[0].cpu().numpy()
        #     pos_error = np.linalg.norm(achieved_pos - target_pos)
        # else:
        #     target_pos = self.pos_abs.cpu().numpy()
        #     achieved_pos = actual_pos.cpu().numpy()
        #     pos_error = np.linalg.norm(achieved_pos - target_pos)

        # if self.debug:
        #     gs.logger.debug(f"\nset_initial_position verification:")
        #     gs.logger.debug(f"  Target pos: {target_pos}")
        #     gs.logger.debug(f"  Genesis EF pos: {achieved_pos}")
        #     gs.logger.debug(f"  Genesis error: {pos_error:.6f} m")

        return qpos

    def set_initial_dofs_position(self, qpos, use_initial_pq=True):
        """
        Set initial joint positions directly.

        Parameters
        ----------
        qpos : array_like
            Joint positions
        use_initial_pq : bool, optional
            If True, use predefined initial position/quaternion.
            If False, compute from current end-effector pose.
        """
        is_batched = self.scene.n_envs > 0
        if is_batched:
            qpos = torch.stack([qpos] * self.scene.n_envs)
        self.robot.set_dofs_position(qpos)

        if use_initial_pq:
            pos_abs = torch.tensor(self.initial_pos, dtype=gs.tc_float)
            quat_abs = torch.tensor(self.initial_quat, dtype=gs.tc_float)
            self.pos_abs = torch.stack([pos_abs] * self.scene.n_envs) if is_batched else pos_abs
            self.quat_abs = torch.stack([quat_abs] * self.scene.n_envs) if is_batched else quat_abs
        else:
            self.pos_abs = self.ef.get_pos()
            self.quat_abs = self.ef.get_quat()

    def _solve_pink_ik(self, pos, quat, init_qpos, stop_threshold=1e-5):
        """
        Solve IK using Pink for a single configuration.

        Parameters
        ----------
        pos : np.ndarray, shape (3,)
            Target position in WORLD coordinates
        quat : np.ndarray, shape (4,)
            Target quaternion [w, x, y, z] (Genesis convention) in WORLD frame
        init_qpos : np.ndarray, shape (robot_model.nq,)
            Initial joint configuration (full, including gripper)
        stop_threshold : float, optional
            Convergence threshold for stopping the IK solver

        Returns
        -------
        qpos : np.ndarray, shape (robot_model.nq,)
            Solved joint configuration (full, including gripper)
        """
        # Update configuration with current qpos
        current_q = np.clip(
            init_qpos,
            self.robot_model.lowerPositionLimit,
            self.robot_model.upperPositionLimit,
        )
        self.configuration.update(current_q)

        # Transform target from world coordinates to robot-local coordinates
        # Pinocchio assumes robot base is at origin, but Genesis robot is at robot_base_pos
        robot_base_pos_np = self.robot_base_pos
        robot_base_quat_np = self.robot_base_quat

        # Transform position from world frame to robot-local frame
        # pos_local = R_base^T @ (pos_world - base_pos)
        pos_local = gu.inv_transform_by_trans_quat(pos, robot_base_pos_np, robot_base_quat_np)

        # Transform orientation from world frame to robot-local frame
        # quat_local = quat_world * quat_base^(-1)
        quat_local = gu.transform_quat_by_quat(quat, gu.inv_quat(robot_base_quat_np))

        # Build target pose in robot-local frame
        quat_local_xyzw = np.array([quat_local[1], quat_local[2], quat_local[3], quat_local[0]])  # [w,x,y,z] -> [x,y,z,w]
        rotation = pin.Quaternion(quat_local_xyzw[3], quat_local_xyzw[0], quat_local_xyzw[1], quat_local_xyzw[2]).toRotationMatrix()
        target_pose = pin.SE3(rotation, pos_local)

        # Create frame task
        frame_task = FrameTask(
            self.ee_frame_name,
            position_cost=self.pink_position_cost,
            orientation_cost=self.pink_orientation_cost,
        )
        frame_task.set_target(target_pose)

        # Create posture task
        posture_task = PostureTask(cost=self.pink_posture_cost)
        posture_task.set_target_from_configuration(self.configuration)

        tasks = [frame_task, posture_task]

        # Check solver availability
        solver = self.pink_solver
        if solver not in qpsolvers.available_solvers:
            if "quadprog" in qpsolvers.available_solvers:
                solver = "quadprog"
            elif len(qpsolvers.available_solvers) > 0:
                solver = qpsolvers.available_solvers[0]
            else:
                print("Warning: No QP solvers available")
                return init_qpos

        # Solve IK iteratively
        converged = False

        for iteration in range(self.pink_max_iterations):
            try:
                # Solve for velocity
                velocity = pink.solve_ik(
                    self.configuration,
                    tasks,
                    self.pink_dt,
                    solver=solver,
                )

                # Integrate and clip to limits
                new_q = self.configuration.integrate(velocity, self.pink_dt)
                new_q = np.clip(
                    new_q,
                    self.robot_model.lowerPositionLimit,
                    self.robot_model.upperPositionLimit,
                )
                self.configuration.update(new_q)

                # Check convergence
                error_norm = np.linalg.norm(frame_task.compute_error(self.configuration))
                if error_norm < stop_threshold:
                    converged = True
                    if self.debug:
                        gs.logger.debug(f"  Pink IK converged in {iteration+1} iterations, error norm: {error_norm:.6f}")
                    break

            except pink.exceptions.NoSolutionFound:
                continue
            except Exception as e:
                gs.logger.warning(f"Warning: Pink IK error at iteration {iteration}: {e}")
                break

        # # Verify solution
        # achieved_transform = self.configuration.get_transform_frame_to_world(self.ee_frame_name)
        # achieved_pos_local = achieved_transform.translation

        # # Convert achieved position and orientation back to world coordinates for comparison
        # achieved_pos_world = gu.transform_by_trans_quat(achieved_pos_local, robot_base_pos_np, robot_base_quat_np)

        # # Extract achieved orientation and convert to world frame
        # achieved_rot_local = achieved_transform.rotation
        # achieved_quat_local_xyzw = pin.Quaternion(achieved_rot_local).coeffs()  # [x, y, z, w]
        # achieved_quat_local = np.array([achieved_quat_local_xyzw[3], achieved_quat_local_xyzw[0],
        #                                achieved_quat_local_xyzw[1], achieved_quat_local_xyzw[2]])  # [w, x, y, z]
        # achieved_quat_world = gu.transform_quat_by_quat(robot_base_quat_np, achieved_quat_local)

        # pos_error_local = np.linalg.norm(achieved_pos_local - pos_local)

        # if self.debug or pos_error_local > 0.01:  # Print if debug or large error
        #     gs.logger.debug(f"  Pink IK solution:")
        #     gs.logger.debug(f"    Target pos (world): {pos}")
        #     gs.logger.debug(f"    Target pos (robot-local): {pos_local}")
        #     gs.logger.debug(f"    Target quat (world): {quat}")
        #     gs.logger.debug(f"    Target quat (robot-local): {quat_local}")
        #     gs.logger.debug(f"    Achieved pos (robot-local): {achieved_pos_local}")
        #     gs.logger.debug(f"    Achieved pos (world): {achieved_pos_world}")
        #     gs.logger.debug(f"    Achieved quat (robot-local): {achieved_quat_local}")
        #     gs.logger.debug(f"    Achieved quat (world): {achieved_quat_world}")
        #     gs.logger.debug(f"    Position error: {pos_error_local:.6f} m")
        #     gs.logger.debug(f"    Converged: {converged}")
        #     if not converged:
        #         gs.logger.warning(f"    Final error norm: {error_norm:.6f}")

        return self.configuration.q.copy()

    def control_robot(
        self, g_dof1, g_dof2,
        dx=0., dy=0., dz=0., di=0., dj=0., dk=0.,
        g_dof_use_force=False, degrees=True, **kwargs
    ):
        """
        Control robot end-effector to move by specified deltas using Pink IK.

        Parameters
        ----------
        g_dof1, g_dof2 : float
            Gripper DOF commands
        dx, dy, dz : float or torch.Tensor
            Position delta in meters
        di, dj, dk : float or torch.Tensor
            Orientation delta (roll, pitch, yaw) in degrees or radians
        g_dof_use_force : bool, optional
            Use force control for gripper (default: False)
        degrees : bool, optional
            Interpret di, dj, dk as degrees (default: True)
        **kwargs : optional
            Additional arguments (e.g., min_z for ground constraint)

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
        """
        # Compute target position
        if isinstance(dx, (float, int)) and isinstance(dy, (float, int)) and isinstance(dz, (float, int)):
            delta_pos = torch.tensor([dx, dy, dz], dtype=gs.tc_float)
        else:
            delta_pos = torch.stack([dx, dy, dz], dim=-1)
        target_pos = self.pos_abs + delta_pos

        # Handle minimum Z constraint
        if kwargs.get('min_z', None) is not None:
            min_z = torch.zeros_like(target_pos[..., 2])
            min_z.fill_(kwargs.pop('min_z'))
            kwargs.update({'underground': (target_pos[..., 2] < min_z).any()})
            target_pos[..., 2] = torch.maximum(target_pos[..., 2], min_z)

        # Compute target orientation
        if isinstance(di, (float, int)) and isinstance(dj, (float, int)) and isinstance(dk, (float, int)):
            delta_orient = torch.tensor([di, dj, dk], dtype=gs.tc_float)
        else:
            delta_orient = torch.stack([di, dj, dk], dim=-1)
        delta_quat = gu.xyz_to_quat(delta_orient, rpy=True, degrees=degrees)

        if delta_quat.ndim == 1 and self.quat_abs.ndim == 2:
            delta_quat = torch.stack([delta_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else delta_quat
        elif delta_quat.ndim != self.quat_abs.ndim:
            raise ValueError("`delta_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(delta_quat, self.quat_abs)

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

        return qpos

    def rotate_around_point(
        self, g_dof1, g_dof2, center, axis, angle, pos_angle=None,
        g_dof_use_force=False, degrees=True, **kwargs
    ):
        """
        Rotate robot end-effector around a specified world-space point using Pink IK.

        Parameters
        ----------
        g_dof1, g_dof2 : float
            Gripper DOF commands
        center : array_like, shape (3,)
            Center point for rotation
        axis : array_like, shape (3,)
            Rotation axis (will be normalized)
        angle : float
            Rotation angle for orientation
        pos_angle : float, optional
            Rotation angle for position (default: same as angle)
        g_dof_use_force : bool, optional
            Use force control for gripper
        degrees : bool, optional
            Interpret angles as degrees
        **kwargs : optional
            Additional arguments

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
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
            orient_rotation_quat = torch.stack([orient_rotation_quat] * self.scene.n_envs) if self.scene.n_envs > 0 else orient_rotation_quat
        elif orient_rotation_quat.ndim != self.quat_abs.ndim:
            raise ValueError("`orient_rotation_quat` and `quat_abs` must have the same number of dimensions.")
        target_quat = gu.transform_quat_by_quat(orient_rotation_quat, self.quat_abs)

        qpos = self._execute_ik_control(target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs)

        return qpos

    def _execute_ik_control(self, target_pos, target_quat, g_dof1, g_dof2, g_dof_use_force, **kwargs):
        """
        Execute inverse kinematics and send control commands using Pink.

        Parameters
        ----------
        target_pos : torch.Tensor
            Target end-effector position
        target_quat : torch.Tensor
            Target end-effector quaternion
        g_dof1, g_dof2 : float
            Gripper DOF commands
        g_dof_use_force : bool
            Use force control for gripper
        **kwargs : optional
            Additional arguments (e.g., underground flag for debug)

        Returns
        -------
        qpos : torch.Tensor
            Computed joint positions
        """
        is_batched = self.scene.n_envs > 0
        underground = kwargs.pop('underground', False)

        # Debug visualization
        if self.debug:
            for i in self.debug_point_nodes:
                self.scene.clear_debug_object(i)
            self.debug_point_nodes = list()
            for batch_idx in range(self.scene.n_envs if is_batched else 1):
                if is_batched:
                    offset = self.scene.envs_offset[batch_idx]
                    offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
                else:
                    offset = torch.zeros(3, dtype=target_pos.dtype, device=target_pos.device)
                color = (1.0, 1.0, 0.0, 0.6) if underground else (0.0, 1.0, 0.0, 0.6)
                self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                    pos=target_pos[batch_idx] + offset if is_batched else target_pos,
                    radius=0.01,
                    color=color
                ))

        # Get current joint positions
        current_qpos = self.robot.get_qpos()

        # Solve IK using Pink
        if is_batched:
            # Solve for each environment separately
            qpos_list = []
            for batch_idx in range(self.scene.n_envs):
                pos_np = target_pos[batch_idx].cpu().numpy()
                quat_np = target_quat[batch_idx].cpu().numpy()
                init_qpos_np = current_qpos[batch_idx].cpu().numpy()  # Full qpos including gripper

                qpos_full = self._solve_pink_ik(pos_np, quat_np, init_qpos_np)
                qpos_list.append(torch.tensor(qpos_full, dtype=gs.tc_float))

            qpos = torch.stack(qpos_list)

            # Override gripper DOFs with commanded values
            qpos[:, -2] = g_dof1
            qpos[:, -1] = g_dof2
        else:
            # Single environment
            pos_np = target_pos.cpu().numpy()
            quat_np = target_quat.cpu().numpy()
            init_qpos_np = current_qpos.cpu().numpy()  # Full qpos including gripper

            qpos_full = self._solve_pink_ik(pos_np, quat_np, init_qpos_np)
            qpos = torch.tensor(qpos_full, dtype=gs.tc_float)

            # Override gripper DOFs with commanded values
            qpos[-2] = g_dof1
            qpos[-1] = g_dof2

        # Send control commands
        self.robot.control_dofs_position(qpos[..., :-2], self.motors_dof)

        if g_dof_use_force:
            gripper_arg = torch.tensor([[g_dof1, g_dof2]] * self.scene.n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])
            self.robot.control_dofs_force(gripper_arg, self.fingers_dof)
        else:
            gripper_arg = torch.tensor([[g_dof1, g_dof2]] * self.scene.n_envs) if is_batched else torch.tensor([g_dof1, g_dof2])
            self.robot.control_dofs_position(gripper_arg, self.fingers_dof)

        # Update internal state
        self.pos_abs = target_pos
        self.quat_abs = target_quat

        return qpos

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
