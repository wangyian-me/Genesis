import argparse
import mediapy
import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from collections import defaultdict

def control_robot(
    args, robot, ef, g_dof1, g_dof2,
    motors_dof=np.arange(7), fingers_dof=np.arange(7, 9),
    dpx=0, dpy=0, dpz=0, dex=0, dey=0, dez=0
):
    target_pos = ef.get_pos().cpu().numpy().reshape(-1) + np.array([dpx, dpy, dpz])
    if dex == 0 and dey == 0 and dez == 0:
        target_quat = ef.get_quat().cpu().numpy().reshape(-1)
    else:
        delta_orientation = np.array([dex, dey, dez])
        delta_quat = gu.xyz_to_quat(
            delta_orientation, rpy=True, degrees=True
        )
        target_quat = gu.transform_quat_by_quat(
            delta_quat, ef.get_quat().cpu().numpy().reshape(-1)
        )
    print("target pos", target_pos, "target quat", target_quat)
    qpos = robot.inverse_kinematics(
        link=ef,
        pos=target_pos if args.n_envs == 0 else np.array([target_pos] * args.n_envs),
        quat=target_quat if args.n_envs == 0 else np.array([target_quat] * args.n_envs),
    )
    robot.control_dofs_position(qpos[..., :-2], motors_dof)
    robot.control_dofs_position(
        np.array([g_dof1, g_dof2]) if args.n_envs == 0 else np.array([g_dof1, g_dof2] * args.n_envs), fingers_dof
    )  # you can use position control

def control_robot_abs(
    args, robot, ef, g_dof1, g_dof2,
    motors_dof=np.arange(7), fingers_dof=np.arange(7, 9),
    x=0, y=0, z=0, quat=np.array([0, 1, 0, 0])
):
    target_pos = np.array([x, y, z])
    target_quat = quat
    qpos = robot.inverse_kinematics(
        link=ef,
        pos=target_pos if args.n_envs == 0 else np.array([target_pos] * args.n_envs),
        quat=target_quat if args.n_envs == 0 else np.array([target_quat] * args.n_envs),
    )
    robot.control_dofs_position(qpos[..., :-2], motors_dof)
    robot.control_dofs_position(
        np.array([g_dof1, g_dof2]) if args.n_envs == 0 else np.array([g_dof1, g_dof2] * args.n_envs), fingers_dof
    )  # you can use position control

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug",backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=25,
            # gravity=(0.,0.,0.)
        ),
        rod_options=gs.options.RodOptions(
            damping=15.0,
            angular_damping=10.0,
        ),
        show_viewer=args.vis,
    )

    cameras = list()
    if args.path is not None:
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-1.8, 1.2, 1.4), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=24, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(600, 450), pos=(-1, -0.8, 1.4), up=(0, 0, 1),
            lookat=(0.2, 0., 0), fov=20, GUI=False
        ))

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.01,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    segment_radius = 0.01
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            # K=1e6,
            E=1e4,
            G=0,
            plastic_yield=np.inf,
            # use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=60,
            interval=0.01,
            axis="x",
            pos=(0.3, 0.0, 0.02),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        )
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.7
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.45, -0.6, 0),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )

    franka2 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.8, 0.6, 0),
            # euler=(0., 0., -90.),
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
        # vis_mode='collision',
    )

    gripper_geom_indices = list()
    lf = franka1.get_link("panda_leftfinger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    lf = franka2.get_link("panda_leftfinger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka1.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka2.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)

    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    for f in [franka1, franka2]:
        if args.n_envs == 0:
            f.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
        else:
            f.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
        f.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 80, 80]),
        )
        f.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
        )
        f.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
            np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
        )

    # end_effector = franka.get_link("hand")
    ef1 = franka1.get_link("panda_grasptarget")
    ef2 = franka2.get_link("panda_grasptarget")