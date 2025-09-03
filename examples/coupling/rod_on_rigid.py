import argparse
import mediapy
import genesis as gs


def test_v1(scene):
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
        ),
        morph=gs.morphs.Rod(
            file="test.npy",
            scale=1.0,
            pos=(0.5, 0.5, 0.3),
            euler=(0.0, 0.0, 15.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
        ),
        morph=gs.morphs.Rod(
            file="test.npy",
            scale=1.0,
            pos=(0.55, 0.43, 0.4),
            euler=(0.0, 0.0, 0.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.Rod(
            file="fixed.npy",
            scale=1.0,
            pos=(0.75, 0.435, 0.2),
            euler=(0.0, 0.0, -75.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    b2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.Rod(
            file="fixed.npy",
            scale=1.0,
            pos=(1.05, 0.435, 0.25),
            euler=(0.0, 0.0, -75.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=2)

    v1.set_fixed_states(
        fixed_ids = [0, 1]
    )
    v2.set_fixed_states(
        fixed_ids = [98, 99]
    )
    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )
    b2.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )


def test_v2(scene):
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
        ),
        morph=gs.morphs.Rod(
            file="test.npy",
            scale=1.0,
            pos=(0.7, 0.05, 0.5),
            euler=(0.0, 0.0, 15.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            static_friction=1.5,
            kinetic_friction=1.25
        ),
        morph=gs.morphs.Rod(
            file="test.npy",
            scale=1.0,
            pos=(0.7, 0.15, 0.5),
            euler=(0.0, 0.0, 15.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.Rod(
            file="fixed.npy",
            scale=1.0,
            pos=(0.85, 0.05, 0.3),
            euler=(0.0, 0.0, 105.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    b2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.02,
        ),
        morph=gs.morphs.Rod(
            file="fixed.npy",
            scale=1.0,
            pos=(1.25, 0.15, 0.3),
            euler=(0.0, 0.0, 105.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=2)

    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )
    b2.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )


def test_v3(scene):
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.001,
            E=1e7,
            G=1e7
        ),
        morph=gs.morphs.Rod(
            file="circle.npy",
            scale=1.0,
            pos=(1.02, -0.03, 0.15),
            euler=(0.0, 90.0, 105.0),
            is_loop=True
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    v2 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.001,
            E=1e7,
            G=1e7
        ),
        morph=gs.morphs.Rod(
            file="testshort.npy",
            scale=1.0,
            pos=(0.95, 0.0, 0.1),
            euler=(0.0, 0.0, 15.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 1.0, 0.4),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
        ),
        morph=gs.morphs.Rod(
            file="fixed.npy",
            scale=1.0,
            pos=(0.9, 0.0, 0.05),
            euler=(0.0, 0.0, 105.0),
            is_loop=False 
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4),
            vis_mode='recon',
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=2)

    b1.set_fixed_states(
        fixed_ids = [0, 1, 2]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--save_path", type=str, default=None)
    parser.add_argument("--fov", type=float, default=30)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("-st", "--substeps", type=int, default=20)
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=20,
        ),
        rod_options=gs.options.RodOptions(
            damping=10,
            floor_height=0.0,
            floor_normal=(0., 0., 1.),
            adjacent_gap=2,
            n_pbd_iters=10
        ),
        show_viewer=args.vis,
    )

    if args.save_path is not None:
        cam = scene.add_camera(
            res=(600, 450), pos=(2.6, 1.8, 1.6), up=(0, 0, 1),
            lookat=(0.9, 0.1, 0), fov=args.fov, GUI = False
        )
    else:
        cam = None

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    plane = scene.add_entity(
        material=frictionless_rigid,
        morph=gs.morphs.Plane(),
    )

    # cube = scene.add_entity(
    #     material=frictionless_rigid,
    #     morph=gs.morphs.Box(
    #         pos=(0.5, 0.5, 0.2),
    #         size=(0.2, 0.2, 0.2),
    #         euler=(30, 40, 0),
    #         fixed=True,
    #     ),
    # )

    # test_v1(scene)
    # test_v2(scene)
    # test_v3(scene)

    frames = list()
    for i in range(args.steps):
        scene.step()
        if cam is not None:
            img = cam.render()[0]
            frames.append(img)

    if cam is not None:
        mediapy.write_video(args.save_path, frames, fps=18, qp=18)


if __name__ == "__main__":
    main()
