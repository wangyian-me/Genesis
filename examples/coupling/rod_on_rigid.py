import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="debug", backend=gs.cpu if args.cpu else gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=10,
        ),
        rod_options=gs.options.RodOptions(
            damping=0.001,
            floor_height=0.0,
            floor_normal=(0., 0., 1.),
            adjacent_gap=2,
            n_pbd_iters=10
        ),
        show_viewer=False,
    )

    ########################## entities ##########################
    frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)

    # plane = scene.add_entity(
    #     material=frictionless_rigid,
    #     morph=gs.morphs.Plane(),
    # )

    # cube = scene.add_entity(
    #     material=frictionless_rigid,
    #     morph=gs.morphs.Box(
    #         pos=(0.5, 0.5, 0.2),
    #         size=(0.2, 0.2, 0.2),
    #         euler=(30, 40, 0),
    #         fixed=True,
    #     ),
    # )

    rod = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
        ),
        morph=gs.morphs.Rod(
            file="test.npy",
            scale=1.0,
            pos=(0.5, 0.5, 0.5),
            euler=(0.0, 0.0, 0.0),
            is_loop=False 
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=2)

    horizon = 100

    for i in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
