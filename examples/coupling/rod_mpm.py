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
            n_vertices=15,
            interval=0.02,
            axis="x",
            pos=(-0.15, 0.25, 0.02),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    E = 2e5
    G = 1e4
    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.004,
            E=E,
            G=G,
            plastic_yield=100.0,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="half_circle",
            n_vertices=22,
            radius=0.16,
            axis="z",
            pos=(-0.15, -0.25, 0.02),
            euler=(90.0, 0.0, 45.0),
            rest_state="straight",
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        ),
    )

    obj_sand = scene.add_entity(
        material=gs.materials.MPM.Sand(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.3),
            vis_mode="particle",
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)

    return scene, cams

def test_v2(args):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.9),
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
            res=(1024, 1024), pos=(2.5, -1., 1.2), up=(0, 0, 1),
            lookat=(0., 0., 0), fov=args.fov, GUI = False
        ))
        cams.append(scene.add_camera(
            res=(1024, 1024), pos=(-2.0, 1.2, 1.5), up=(0, 0, 1),
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

    E = 1e3
    G = 1e4
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            segment_mass=1.0,
            E=E,
            G=G,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=31,
            interval=0.01,
            axis="x",
            pos=(-0.15, 0., 0.45),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            segment_mass=1.0,
            E=E,
            G=G,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=21,
            interval=0.01,
            axis="x",
            pos=(-0.1, -0.07, 0.43),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        ),
    )

    v3 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            segment_mass=1.0,
            E=E,
            G=G,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=21,
            interval=0.01,
            axis="x",
            pos=(-0.1, 0.07, 0.43),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        ),
    )

    obj_elastic = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=500),
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.22),
            radius=0.15,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.7, 0.7),
            vis_mode="particle",
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)

    return scene, cams

def test_v3(args):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.9),
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
            n_vertices=15,
            interval=0.02,
            axis="x",
            pos=(-0.15, 0.25, 0.02),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    E = 2e5
    G = 1e4
    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.004,
            E=E,
            G=G,
            plastic_yield=100.0,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="half_circle",
            n_vertices=22,
            radius=0.16,
            axis="z",
            pos=(-0.15, -0.25, 0.02),
            euler=(90.0, 0.0, 45.0),
            rest_state="straight",
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        ),
    )

    obj_liquid = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Mesh(
            file="meshes/bunny.obj",
            scale=0.3,
            pos=(0., 0., 0.2),
        ),
        surface=gs.surfaces.Default(
            color=(0., 0.4, 0.8),
            vis_mode="particle",
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)

    return scene, cams

def test_v4(args):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.9),
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

    E = 1e2
    G = 1e2
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.004,
            E=E,
            G=G
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=15,
            interval=0.02,
            axis="x",
            pos=(-0.15, 0., 0.1),
            euler=(0.0, 0.0, 90.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    obj_plastic = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(E=2e4, nu=0.3, von_mises_yield_stress=2e3),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.25),
            size=(0.15, 0.1, 0.07),
        ),
        surface=gs.surfaces.Default(
            color=(0.6, 0.8, 0.5),
            vis_mode="particle",
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)

    v1.set_fixed_states(
        fixed_ids=[0, 1, 13, 14]
    )

    return scene, cams

def test_v5(args):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.9),
            grid_density=100,
        ),
        rod_options=gs.options.RodOptions(
            damping=10.0,
            angular_damping=5.0,
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

    K = 1e5
    E = 1e4
    G = 0
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.002,
            K=K,
            E=E,
            G=G,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=16,
            radius=0.05,
            axis="x",
            pos=(0.0, -0.05, 0.1),
            euler=(90.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.002,
            K=K,
            E=E,
            G=G,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=16,
            radius=0.05,
            axis="x",
            pos=(0.0, 0.05, 0.1),
            euler=(90.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/rope02.png",
            ),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=4,
            interval=0.04,
            axis="y",
            pos=(0.0, -0.06, 0.11),
            euler=(0.0, 0.0, 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    obj_sand = scene.add_entity(
        material=gs.materials.MPM.Sand(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.16, 0.08),
            size=(0.125, 0.125, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.3),
            vis_mode="particle",
        ),
    )

    scene.rod_solver.register_gripper_geom_indices()

    ########################## build ##########################
    scene.build(n_envs=1)
    obj_sand.set_velocity(torch.tensor([0.0, -0.8, 0.0]))

    return scene, cams

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

    scene = None
    cams = list()

    # scene, cams = test_v1(args)
    # scene, cams = test_v2(args)
    # scene, cams = test_v3(args)
    # scene, cams = test_v4(args)
    scene, cams = test_v5(args)

    frames = defaultdict(list)
    for i in range(args.steps):
        scene.step()
        for cid, cam in enumerate(cams):
            img = cam.render()[0]
            frames[cid].append(img)

    save_dir = args.path
    os.makedirs(save_dir, exist_ok=True)

    for cid in frames:
        ver = f"_{args.version}" if args.version is not None else ""
        video_path = os.path.join(save_dir, f"video{ver}_c{cid}.mp4")
        mediapy.write_video(video_path, frames[cid], fps=30, qp=18)


if __name__ == "__main__":
    main()
