import os
import torch
import argparse
import mediapy
import numpy as np
import genesis as gs
from collections import defaultdict


def test_v1(args):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -1.0, -0.1),
            upper_bound=(0.5, 1.0, 0.9),
            grid_density=100,
        ),
        rod_options=gs.options.RodOptions(
            damping=1.0,
            angular_damping=1.0,
            adjacent_gap=2,
            n_pbd_iters=20
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        show_viewer=args.vis,
    )

    cams = list()
    if args.path is not None:
        cams.append(scene.add_camera(
            res=(1024, 1024), pos=(3.0, -1.2, 1.2), up=(0, 0, 1),
            lookat=(0., 0., 0), fov=args.fov, GUI = False
        ))
        cams.append(scene.add_camera(
            res=(1024, 1024), pos=(-2.4, 1.6, 1.5), up=(0, 0, 1),
            lookat=(0., 0., 0), fov=args.fov, GUI = False
        ))

    ########################## entities ##########################
    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.1
    )

    plane = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.Plane(),
    )

    E = 1e5
    G = 1e4
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.004,
            E=E,
            G=G
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=41,
            interval=0.02,
            axis="x",
            pos=(0.08, 0.0, 0.08),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    box1 = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.05,
        ),
        morph=gs.morphs.Box(
            size=(0.16, 0.16, 0.16),
            pos=(0., 0., 0.08),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.93, 0.96, 0.98)
        )
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    n_envs = 1
    scene.build(n_envs=n_envs)
    link1 = box1.get_link('box_baselink')
    print(link1.idx)

    verts_ids = [0, 1]
    v1.attach_to_rigid_link(link1, verts_ids)

    box_vel1 = np.ones((n_envs, box1.n_dofs))
    box_vel1[:, 0] = -1.0
    box_vel1[:, 1:] = 0.0

    box_vel2 = np.ones((n_envs, box1.n_dofs))
    box_vel2[:, 0] = 0.0
    box_vel2[:, 1:3] = 1.0
    box_vel2[:, 3:] = 0.0

    frames = defaultdict(list)
    for i in range(args.steps):
        if i < args.steps // 5:
            box1.set_dofs_velocity(box_vel1)
        elif i < 3 * args.steps // 5:
            box1.set_dofs_velocity(box_vel2)
        elif i == 3 * args.steps // 5:
            v1.detach_from_rigid_link(verts_ids)
        scene.step()
        for cid, cam in enumerate(cams):
            img = cam.render()[0]
            frames[cid].append(img)

    return frames


def test_v2(args):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -1.0, -0.1),
            upper_bound=(0.5, 1.0, 0.9),
            grid_density=100,
        ),
        rod_options=gs.options.RodOptions(
            damping=1.0,
            angular_damping=1.0,
            adjacent_gap=2,
            n_pbd_iters=20
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        show_viewer=args.vis,
    )

    cams = list()
    if args.path is not None:
        cams.append(scene.add_camera(
            res=(1024, 1024), pos=(2.0, -1.2, 0.9), up=(0, 0, 1),
            lookat=(0., 0.3, 0), fov=args.fov, GUI = False
        ))
        cams.append(scene.add_camera(
            res=(1024, 1024), pos=(-1.6, 1.0, 1.), up=(0, 0, 1),
            lookat=(0., 0.2, 0), fov=args.fov, GUI = False
        ))

    ########################## entities ##########################
    friction_rigid = gs.materials.Rigid(
        needs_coup=True, coup_friction=0.3
    )

    plane = scene.add_entity(
        material=friction_rigid,
        morph=gs.morphs.Plane(),
    )

    E = 5e5
    G = 1e4
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            segment_mass=0.002,
            E=E,
            G=G
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=21,
            interval=0.02,
            axis="x",
            pos=(-0.02, 0.0, 0.185),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    plug = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.1,
        ),
        morph=gs.morphs.Mesh(
            file="meshes/plug01.obj",
            pos=(0.0, 0.0, 0.2),
            euler=(0, 0, 90),
            scale=(1, 1, 1),
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    n_envs = 1
    scene.build(n_envs=n_envs)
    link1 = plug.get_link('plug01_obj_baselink')
    print(link1.idx)

    verts_ids = [0, 1, 2]
    v1.attach_to_rigid_link(link1, verts_ids)

    plug_vel1 = np.zeros((n_envs, plug.n_dofs))
    plug_vel1[..., 3] = 2

    plug_vel2 = np.zeros((n_envs, plug.n_dofs))
    plug_vel2[:, 0] = 0.4
    plug_vel2[:, 2] = 0.1
    plug_vel2[:, 4] = 1

    plug_vel3 = np.zeros((n_envs, plug.n_dofs))
    plug_vel3[:, 0] = 0.4
    plug_vel3[:, 1] = 0.4
    plug_vel3[:, 4] = 1

    frames = defaultdict(list)
    for i in range(args.steps):
        if i < 150:
            plug.set_dofs_velocity(plug_vel1)
        elif i < 225:
            plug.set_dofs_velocity(plug_vel2)
        else:
            plug.set_dofs_velocity(plug_vel3)
        # print(plug.get_dofs_velocity())
        scene.step()
        for cid, cam in enumerate(cams):
            img = cam.render()[0]
            frames[cid].append(img)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--fov", type=float, default=30)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("-st", "--substeps", type=int, default=20)
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    frames = list()
    # frames = test_v1(args)
    # frames = test_v2(args)

    if args.path is not None:
        save_dir = args.path
        os.makedirs(save_dir, exist_ok=True)

        for cid in frames:
            ver = f"_{args.version}" if args.version is not None else ""
            video_path = os.path.join(save_dir, f"video{ver}_c{cid}.mp4")
            mediapy.write_video(video_path, frames[cid], fps=30, qp=18)


if __name__ == "__main__":
    main()
