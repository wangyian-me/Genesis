import argparse
import mediapy
import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('./examples/rod')
from controller import RobotController
from collections import defaultdict


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
    10 vertices fall down
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

    lr = 0.01

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
                loss += torch.relu(0.02 - state.pos[:, :, 2]).sum()
                print("final pos z:", state.pos.detach().cpu().numpy())

        loss.backward()
        print(f"it: {it}, loss: {loss.item()} grad: {init_pos.grad.detach().cpu().numpy()}")
        with torch.no_grad():
            init_pos -= lr * init_pos.grad
            init_pos.zero_grad()
        print(f"new init pos: {init_pos.detach().cpu().numpy()}")
        print("------------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-n", "--n_envs", type=int, default=49)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="info", backend=gs.gpu)

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
            substeps=10,
            requires_grad=True,
            # gravity=(0.,0.,0.)
        ),
        rod_options=gs.options.RodOptions(
            damping=5.0,
            angular_damping=5.0,
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

    for cid in frames:
        mediapy.write_video(args.path.replace(".mp4", f"_c{cid}.mp4"), frames[cid], fps=30, qp=18)

if __name__ == "__main__":
    main()
