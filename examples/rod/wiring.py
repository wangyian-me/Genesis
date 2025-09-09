import argparse
import mediapy
import numpy as np
import genesis as gs


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
            damping=10.0,
            angular_damping=10.0,
        ),
        show_viewer=args.vis,
    )

    if args.path is not None:
        camera = scene.add_camera(
            res=(600, 450), pos=(-1.8, 1.2, 1.4), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=24, GUI=False
        )

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
            G=1e4,
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

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.008,
            static_friction=0.1,
            kinetic_friction=0.08,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="half_circle",
            n_vertices=24,
            radius=0.04,
            axis="y",
            pos=(0.2, 0.0, 0.008),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
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
            pos=(0.3, -0.6, 0),
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
            pos=(0.6, 0.6, 0),
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

    b1.set_fixed_states(
        fixed_ids=np.arange(24),
    )

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

    x1 = 0.42
    x2 = 0.6
    x_delta = -0.17
    x1_move_forward = -0.09
    y_delta = 0.1
    z = 0.014
    z_delta = 0.01
    force = -3

    # move to pre-grasp pose
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1, 0.0, z]) if args.n_envs == 0 else np.array([[x1, 0.0, z]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    qpos1[..., -2:] = 0.02

    franka1.set_dofs_position(
        qpos1
    )

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2, 0.0, z]) if args.n_envs == 0 else np.array([[x2, 0.0, z]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    qpos2[..., -2:] = 0.02

    franka2.set_dofs_position(
        qpos2
    )

    frames = list()

    # grasp
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    gs.logger.info("grasped")

    # lift
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x1, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    # move
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x1+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x1+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0.05, 0.05]) if args.n_envs == 0 else np.array([[0.05, 0.05]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(30):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta, 0.0, z+0.1]) if args.n_envs == 0 else np.array([[x1+x_delta, 0.0, z+0.1]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0.05, 0.05]) if args.n_envs == 0 else np.array([[0.05, 0.05]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    # f1 open, f2 close

    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta+x1_move_forward, 0.0, z+0.1]) if args.n_envs == 0 else np.array([[x1+x_delta+x1_move_forward, 0.0, z+0.1]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0.05, 0.05]) if args.n_envs == 0 else np.array([[0.05, 0.05]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(80):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)
    
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta+x1_move_forward, 0.0, z]) if args.n_envs == 0 else np.array([[x1+x_delta+x1_move_forward, 0.0, z]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0.05, 0.05]) if args.n_envs == 0 else np.array([[0.05, 0.05]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(80):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)
    
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta+x1_move_forward, 0.0, z]) if args.n_envs == 0 else np.array([[x1+x_delta+x1_move_forward, 0.0, z]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(80):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)
    
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta+x1_move_forward, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x1+x_delta+x1_move_forward, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)
    
    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta+x1_move_forward+0.3*x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x1+x_delta+x1_move_forward+0.3*x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+1.3*x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+1.3*x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    qpos1 = franka1.inverse_kinematics(
        link=ef1,
        pos=np.array([x1+x_delta+x1_move_forward+0.3*x_delta, 0.0+y_delta, z+z_delta]) if args.n_envs == 0 else np.array([[x1+x_delta+x1_move_forward+0.3*x_delta, 0.0+y_delta, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka1.control_dofs_position(qpos1[..., :-2], motors_dof)
    franka1.control_dofs_position(
        np.array([0, 0]) if args.n_envs == 0 else np.array([[0, 0]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka1.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    qpos2 = franka2.inverse_kinematics(
        link=ef2,
        pos=np.array([x2+1.3*x_delta, 0.0, z+z_delta]) if args.n_envs == 0 else np.array([[x2+1.3*x_delta, 0.0, z+z_delta]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka2.control_dofs_position(qpos2[..., :-2], motors_dof)
    franka2.control_dofs_position(
        np.array([0.05, 0.05]) if args.n_envs == 0 else np.array([[0.05, 0.05]] * args.n_envs), fingers_dof
    )  # you can use position control
    # franka2.control_dofs_force(
    #     np.array([force, force]) if args.n_envs == 0 else np.array([[force, force]] * args.n_envs), fingers_dof
    # )  # can also use force control

    for i in range(50):
        scene.step()
        if args.path is not None:
            img = camera.render()[0]
            frames.append(img)

    if args.path is not None:
        mediapy.write_video(args.path, np.array(frames), fps=30)


if __name__ == "__main__":
    main()
