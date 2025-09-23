import argparse
import mediapy
import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('./examples/rod')
from controller import RobotControllerOptim, DLOControllerOptim
from collections import defaultdict


def create_linear_array(N):
    base_seq = torch.arange(1, N + 1, dtype=gs.tc_float)
    out = base_seq / base_seq.sum()
    return out

def create_exp_array(N, base=1.1):
    exponents = torch.arange(N, dtype=gs.tc_float)
    base_seq = base ** exponents
    out = base_seq / base_seq.sum()
    return out

def create_custom_array(N):
    base_seq = torch.ones(N, dtype=gs.tc_float)
    base_seq[:N-1] = 0.5 / (N - 1)
    base_seq[N-1] = 0.5
    assert base_seq.sum() == 1.0
    return base_seq

def test_v1(args, scene, cameras, frames):
    """
    10 vertices fall down
    try to optimize the initial pos to reach the target
    """
    segment_radius = 0.01
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            # K=1e4,
            E=5e3,
            G=1e3,
            # use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=10,
            interval=0.02,
            axis="x",
            pos=(0, 0.0, 0.04),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        )
    )

    gripper_geom_indices = list()
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    # NOTE: set init pos will change the COM of the rod
    init_pos = gs.tensor([0., 0.05, 0.02], requires_grad=True)
    target_pos = gs.tensor([0., 0.1, 0.])

    lr = 0.01

    for it in range(20):
        scene.reset()
        r1.set_position(init_pos)
        loss = 0

        horizon = 20
        for i in range(horizon):
            scene.step()
            for cid, cam in enumerate(cameras):
                img = cam.render()[0]
                frames[cid].append(img)
            state = r1.get_state()
            if i == horizon - 1:
                loss += torch.pow(state.pos - target_pos, 2).sum()

        loss.backward()
        print(f"it: {it}, loss: {loss.item():.4f} grad: {init_pos.grad.detach().cpu().numpy()}")
        with torch.no_grad():
            init_pos -= lr * init_pos.grad
            init_pos.zero_grad()
        print(f"new init pos: {init_pos.detach().cpu().numpy()}")
        print("------------------------------")


def test_v2(args, scene, cameras, frames):
    """
    25 vertices fall down
    try to optimize the initial pos to reach the target
    """
    segment_radius = 0.005
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            # K=1e4,
            E=1e4,
            G=1e3,
            # use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=25,
            interval=0.01,
            axis="x",
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope03.png",
            ),
            vis_mode='recon',
        )
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=5,
            interval=0.1,
            axis="x",
            euler=(0.0, 0.0, -90.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    gripper_geom_indices = list()
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    init_pos = gs.tensor([0., 0., 0.2], requires_grad=True)
    # init_pos = gs.tensor([0., 0., 0.1], requires_grad=True)
    target_pos = gs.tensor([0., 0.2, 0.02])

    scene.draw_debug_sphere(
        pos=target_pos,
        radius=0.01
    )

    lr = 0.005

    for it in range(20):
        scene.reset()
        r1.set_position(init_pos)
        b1.set_position(gs.tensor([0., 0., 0.14]))
        loss = 0

        horizon = 100
        for i in range(horizon):
            scene.step()
            for cid, cam in enumerate(cameras):
                img = cam.render()[0]
                frames[cid].append(img)
                if i == horizon - 1:
                    frames[cid].extend([img]*15)
            state = r1.get_state()
            if i == horizon - 1:
                loss += torch.pow(state.pos - target_pos, 2).sum()
                # ensure z is above the table, pos is [env, n_vertices, 3]
                loss += 10*torch.relu(0.02 - state.pos[:, :, 2]).sum()
                print("final pos z:", state.pos.detach().cpu().numpy())

        # print(r1._queried_states.states.keys())
        # print(scene.rod_solver._ckpt.keys())

        r1._queried_states.states[99][0].pos.grad
        r1._queried_states.states[100][0].pos.grad
        loss.backward()
        print(f"it: {it}, loss: {loss.item()} grad: {init_pos.grad.detach().cpu().numpy()}")
        # torch.nn.utils.clip_grad_norm_([init_pos], 1.0)
        # print(f"clipped grad: {init_pos.grad.detach().cpu().numpy()}")
        with torch.no_grad():
            init_pos -= lr * init_pos.grad
            init_pos.zero_grad()
        print(f"new init pos: {init_pos.detach().cpu().numpy()}")
        print("------------------------------")


def test_v3(args, scene, cameras, frames):
    """
    25 vertices fall down
    try to optimize the initial pos to reach the target
    """
    segment_radius = 0.005
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.0005,
            # K=1e4,
            E=1e3,
            G=1e3,
            # use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=25,
            interval=0.01,
            axis="x",
            pos=(0., 0.0, 0.005),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope03.png",
            ),
            vis_mode='recon',
        )
    )

    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.9
    )

    franka1 = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            pos=(0.2, -0.5, 0),
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
    rf = franka1.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    # Optional: set control gains
    for f in [franka1]:
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

    ef1 = franka1.get_link("panda_grasptarget")

    init_pos = (0.23, 0.0, 0.007)
    n_stages = 5
    open_gap = 0.04

    # initial grasp pose
    c1 = RobotControllerOptim(
        scene, franka1, ef1, args, init_pos,
        initial_q_dof=open_gap,
        n_stages=n_stages,
        n_optim_dofs=6,
        max_d_pos=0.1,
        max_d_angle=10.,
        debug=True
    )

    target_pos = gs.tensor([0.4, 0., 0.005])

    for env_id in range(args.n_envs):
        offset = scene.envs_offset[env_id]
        offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
        scene.draw_debug_sphere(
            pos=target_pos + offset,
            radius=0.01,
            color=(0.8, 0.8, 0., 0.5)
        )

    lr = 0.05

    for it in range(10):
        total_horizon = 0
        horizon_ids = list()
        scene.reset()
        c1.set_initial_position()
        loss = 0

        c1.control_robot(0, 0)
        for i in range(50):
            scene.step()
            for cid, cam in enumerate(cameras):
                img = cam.render()[0]
                frames[cid].append(img)
        total_horizon += 50

        c1.control_robot(0, 0, dz=0.005)
        for i in range(100):
            scene.step()
            for cid, cam in enumerate(cameras):
                img = cam.render()[0]
                frames[cid].append(img)
        total_horizon += 100

        kids = r1.get_kinematic_indices()

        for j in range(n_stages):
            c1.apply_grad(0, 0, stage_idx=j)
            horizon = 50
            for i in range(horizon):
                scene.step()
                for cid, cam in enumerate(cameras):
                    img = cam.render()[0]
                    frames[cid].append(img)
                    if i == horizon - 1:
                        frames[cid].extend([img]*15)

            state = r1.get_state()
            loss += torch.pow(state.pos - target_pos, 2).sum()
            # ensure z is above the table, pos is [env, n_vertices, 3]
            loss += torch.relu(0.005 - state.pos[:, :, 2]).sum()
            # if i == horizon - 1:
            # print("final pos z:", state.pos.detach().cpu().numpy())
            total_horizon += horizon
            horizon_ids.append(total_horizon)

        # print(r1._queried_states.states.keys())
        # print(scene.rod_solver._ckpt.keys())
        print("backprop ...")
        loss.backward()
        # print(f"it: {it}, loss: {loss.item()} grad: {init_pos.grad.detach().cpu().numpy()}")
        print(f"it: {it}, loss: {loss.item()}")
        for stage_idx, hor_id in enumerate(horizon_ids):
            final_state = r1._queried_states.states[hor_id][0].pos
            final_state_grad = r1._queried_states.states[hor_id][0].pos.grad

            # print(final_state_grad[:, kids, :])

            c1.gather_grad(
                final_state_grad,
                final_state,
                kids, stage_idx, lr=lr
            )

        print(f"new traj: {c1.traj.detach().cpu().numpy()}")

        print("------------------------------")


def test_v4(args, scene, cameras, frames):
    segment_radius = 0.01
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.0005,
            # K=1e4,
            E=1e3,
            G=1e3,
            # use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=13,
            interval=0.02,
            axis="x",
            pos=(0., 0.0, segment_radius),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        )
    )

    gripper_geom_indices = list()
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    n_stages = 200
    grasp_point_ids = [11]

    # initial controller
    c1 = DLOControllerOptim(
        scene, r1, grasp_point_ids,
        n_stages=n_stages, 
        n_optim_dofs=3, 
        max_ddist=0.002,
        debug=True
    )

    target_pos = gs.tensor([0.4, 0., segment_radius])

    for env_id in range(args.n_envs):
        offset = scene.envs_offset[env_id]
        offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
        scene.draw_debug_sphere(
            pos=target_pos + offset,
            radius=0.01,
            color=(0.8, 0.8, 0., 0.5)
        )

    lr = 0.1
    loss_list = list()

    for it in range(40):
        total_horizon = 0
        horizon_ids = list()
        scene.reset()
        r1.set_fixed_states(
            fixed_ids=grasp_point_ids,
        )
        loss = 0

        for j in range(n_stages):
            c1.apply_grad(stage_idx=j)
            horizon = 1
            for i in range(horizon):
                scene.step()
                for cid, cam in enumerate(cameras):
                    img = cam.render()[0]
                    frames[cid].append(img)
                    # if i == horizon - 1:
                    #     frames[cid].extend([img]*15)

            state = r1.get_state()
            total_horizon += horizon
            horizon_ids.append(total_horizon)
            # scale = 0.01
            scale = 1 / n_stages
            if j == n_stages - 1:
                print(f'stage {j} end pos:\n{state.pos[0, :, :]}')
                # scale = 1.0
            loss += scale * torch.pow(state.pos - target_pos, 2).sum()
            # ensure z is above the table, pos is [env, n_vertices, 3]
            loss += torch.relu(segment_radius - state.pos[:, :, 2]).sum()
            # if i == horizon - 1:
            # print("final pos z:", state.pos.detach().cpu().numpy())

        # print(r1._queried_states.states.keys())
        print(scene.rod_solver._ckpt.keys())
        print("backprop ...")
        loss.backward()
        # print(f"it: {it}, loss: {loss.item()} grad: {init_pos.grad.detach().cpu().numpy()}")
        print(f"it: {it}, loss: {loss.item()}")
        loss_list.append(loss.item())
        for stage_idx, hor_idx in enumerate(horizon_ids):
            # print(final_state_grad[:, kids, :])

            c1.gather_grad(
                stage_idx, hor_idx, lr=lr
            )

        # print(f"new traj: {c1.traj.detach().cpu().numpy()}")

        print("------------------------------")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.grid()
    plt.savefig(args.path.replace('.mp4', '_loss.png'))
    plt.close()


def test_v5(args, scene, cameras, frames):
    segment_radius = 0.01
    r1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.0005,
            # K=1e4,
            E=1e3,
            G=1e3,
            # use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=13,
            interval=0.02,
            axis="x",
            pos=(0., 0.0, segment_radius),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        )
    )

    gripper_geom_indices = list()
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(1, 1))

    n_stages = 100
    grasp_point_ids = [11]

    # initial controller
    c1 = DLOControllerOptim(
        scene, r1, grasp_point_ids,
        n_stages=n_stages,
        n_optim_dofs=3,
        max_ddist=0.001,
        use_adam=True,
        debug=True
    )

    target_pos = gs.tensor([0.4, 0., segment_radius])

    for env_id in range(args.n_envs):
        offset = scene.envs_offset[env_id]
        offset = torch.as_tensor(offset, dtype=target_pos.dtype, device=target_pos.device)
        scene.draw_debug_sphere(
            pos=target_pos + offset,
            radius=0.01,
            color=(0.8, 0.8, 0., 0.5)
        )

    lr = 0.0001
    loss_list = list()

    for it in range(10):
        total_horizon = 0
        horizon_ids = list()
        scene.reset()
        r1.set_fixed_states(
            fixed_ids=grasp_point_ids,
        )
        loss = 0
        total_adv = 0.

        # scales = create_linear_array(n_stages)
        # scales = create_exp_array(n_stages)
        scales = create_custom_array(n_stages)
        print("scales:", scales)
        for j in range(n_stages):
            horizon = 10
            hpos, dpos = c1.pre_apply_grad(stage_idx=j, num_horizons=horizon)
            total_adv += dpos[0, 0, 0].item()
            print(f'stage {j} total adv: {total_adv:.4f} | {r1.get_all_verts()[0, grasp_point_ids, :]}')
            for i in range(horizon):
                c1.on_apply_grad(hpos[i])
                # print(f'  hid before step {i} |{hpos[i][0, 0, :]}|{r1.get_all_verts()[0, grasp_point_ids, :]}')
                scene.step()
                # print(f'  hid after step {i} |{hpos[i][0, 0, :]}|{r1.get_all_verts()[0, grasp_point_ids, :]}')
                for cid, cam in enumerate(cameras):
                    img = cam.render()[0]
                    frames[cid].append(img)
                    # if i == horizon - 1:
                    #     frames[cid].extend([img]*15)

            state = r1.get_state()
            total_horizon += horizon
            horizon_ids.append(total_horizon)
            # scale = 0.01
            # scale = 1 / n_stages
            scale = scales[j]
            if j == n_stages - 1:
                print(f'stage {j} end pos:\n{state.pos[0, :, :]}')
                # scale = 1.0

            loss += scale * torch.pow(state.pos - target_pos[None, None, :], 2).sum()
            # ensure z is above the table, pos is [env, n_vertices, 3]
            loss += scale * torch.relu(segment_radius - state.pos[:, :, 2]).sum()
            # if i == horizon - 1:
            # print("final pos z:", state.pos.detach().cpu().numpy())

        print(r1._queried_states.states.keys())
        # print(scene.rod_solver._ckpt.keys())
        print("backprop ...")
        loss.backward()
        # print(f"it: {it}, loss: {loss.item()} grad: {init_pos.grad.detach().cpu().numpy()}")
        print(f"it: {it}, loss: {loss.item()}")
        loss_list.append(loss.item())
        for stage_idx, hor_idx in enumerate(horizon_ids):
            # print(final_state_grad[:, kids, :])

            c1.gather_grad(
                stage_idx, hor_idx, lr=lr
            )

        # print(f"new traj: {c1.traj.detach().cpu().numpy()}")

        print("------------------------------")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.grid()
    plt.savefig(args.path.replace('.mp4', '_loss.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu)

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
            # dt=5e-3,
            # substeps=10,
            dt=1e-3,
            substeps=5,
            requires_grad=True,
            # gravity=(0.,0.,0.)
        ),
        rod_options=gs.options.RodOptions(
            damping=15.0,
            angular_damping=10.0,
            n_pbd_iters=20,
        ),
        show_viewer=args.vis,
    )

    cameras = list()
    if args.path is not None:
        cameras.append(scene.add_camera(
            res=(900, 675), pos=(1.2, -1.7, 1.4), up=(0, 0, 1),
            lookat=(0., 0.2, 0.02), fov=24, GUI=False
        ))
        cameras.append(scene.add_camera(
            res=(900, 675), pos=(-1.7, 1.2, 1.), up=(0, 0, 1),
            lookat=(0., 0.2, 0.02), fov=20, GUI=False
        ))

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.3,
        ),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    frames = defaultdict(list)

    # test_v1(args, scene, cameras, frames)
    # test_v2(args, scene, cameras, frames)
    # test_v3(args, scene, cameras, frames)
    # test_v4(args, scene, cameras, frames)
    test_v5(args, scene, cameras, frames)

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()
