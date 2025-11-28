import os
import json
import csv
import time
import pickle
import random
from typing import Tuple, List, Optional, Sequence
from argparse import ArgumentParser

import numpy as np
import cma

from train_env import Train_Env
from train_env_wiring_ring import Train_Env_Wiring_ring  # keep if you still want the example main
from train_env_wiring_post import Train_Env_Wiring_post
from train_env_lifting import Train_Env_Lifting
from train_env_slingshot import Train_Env_Slingshot
from train_env_wireart import Train_Env_Wireart

from train_env_coiling import Train_Env_Coiling
from train_env_gathering import Train_Env_Gathering
from train_env_separation import Train_Env_Separation
from train_env_wrapping import Train_Env_Wrapping


# ----------------------------
# Helpers: shape & constraints
# ----------------------------
def reshape_to_traj(x: np.ndarray, n_steps: int, act_dim: int) -> np.ndarray:
    return x.reshape(n_steps, act_dim)

def _as_per_comp_array(per_comp_bound: Optional[Sequence[float]], act_dim: int) -> np.ndarray:
    if per_comp_bound is None:
        return np.full((act_dim,), np.inf, dtype=np.float32)
    if np.isscalar(per_comp_bound):
        return np.full((act_dim,), float(per_comp_bound), dtype=np.float32)
    arr = np.asarray(per_comp_bound, dtype=np.float32).reshape(-1)
    if arr.size != act_dim:
        raise ValueError(f"per_comp_bound length {arr.size} != act_dim {act_dim}")
    return arr

def project_deltas(traj: np.ndarray,
                   pcb: np.ndarray,
                   max_l2_per_step: Optional[float],
                   scene_version: int = 1) -> np.ndarray:
    n_steps, act_dim = traj.shape
    assert pcb.shape == (act_dim,)
    # if scene_version == 2:
    #     # scale rotational components (original sigma0 is small for rotational components)
    #     traj[:, act_dim // 2:] *= 16.
    #     # convert radians to degrees for rotational components
    #     traj[:, act_dim // 2:] *= 180. / np.pi
    if np.isfinite(pcb).any():
        traj = np.clip(traj, -pcb, pcb)
    if max_l2_per_step is not None and np.isfinite(max_l2_per_step):
        if scene_version == 1:
            norms = np.linalg.norm(traj, axis=1, keepdims=True)
            scale = np.ones_like(norms, dtype=traj.dtype)
            over = norms > max_l2_per_step
            scale[over] = max_l2_per_step / (norms[over] + 1e-12)
            traj = traj * scale
        elif scene_version == 2:
            # only limit the first half of the action dimensions (x,y,z)
            norms = np.linalg.norm(traj[:, :act_dim // 2], axis=1, keepdims=True)
            scale = np.ones_like(norms, dtype=traj.dtype)
            over = norms > max_l2_per_step
            scale[over] = max_l2_per_step / (norms[over] + 1e-12)
            traj[:, :act_dim // 2] = traj[:, :act_dim // 2] * scale
    return traj


# ----------------------------
# Parallel evaluation (batch)
# ----------------------------
def evaluate_batch(env, traj_list: List[np.ndarray]):
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr
    # rewards = env.eval_traj(trajs)
    # return np.asarray(rewards, dtype=np.float32)

    # only for eval_traj_v3
    return env.eval_traj(trajs)

def evaluate_single(env: Train_Env, traj: np.ndarray, log_dir: str, n_steps: int) -> float:
    print(f'Traj shape: {traj.shape}')
    if env.scene_version == 1:
        rewards = env.eval_traj(traj[None, ...], debug=True)
        print(f'Single traj reward: {rewards[0]:.4f}')
    elif env.scene_version == 2:
        # TODO: hack here
        # if getattr(env, "c1", None) is not None:
        #     env.c1.debug = True
        # if getattr(env, "c2", None) is not None:
        #     env.c2.debug = True

        if os.path.exists(os.path.join(log_dir, "best_traj.npy")):
            placeholder = np.load(os.path.join(log_dir, "best_traj.npy"))
            placeholder = placeholder[None, ...]
        else:
            # Resolve shapes
            act_dim = _infer_act_dim(env)
            placeholder = np.zeros((1, n_steps, act_dim), dtype=np.float32)
        out = env.eval_traj(placeholder, debug=True, qpos=traj)
        cum_rewards = out['cum_reward']
        final_rewards = out['final_reward']
        print(f'Single traj cum reward: {cum_rewards[0]:.4f}, final reward: {final_rewards[0]:.4f}')
        rewards = cum_rewards

    env.save_animation(save_dir=log_dir)
    return rewards[0]

# ----------------------------
# Logging helpers
# ----------------------------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _maybe_write_header(path: str, header: List[str]):
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if needs_header:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def _append_rewards(log_dir: str, iteration: int, final: np.ndarray, cum: np.ndarray):
    path = os.path.join(log_dir, "rewards_all.csv")
    _maybe_write_header(path, ["iter", "idx", "final", "cum"])
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for idx in range(len(final)):
            w.writerow([iteration, idx, float(final[idx]), float(cum[idx])])

def _append_summary(log_dir: str, iteration: int, pop: int, n_chunks: int,
                    mean: float, std: float, rmin: float, rmax: float,
                    best_so_far: float, sigma_now: float,
                    forward_time: float,
                    t_iter_sec: float, t_total_sec: float):
    path = os.path.join(log_dir, "summary.csv")
    _maybe_write_header(path, [
        "iter", "pop", "chunks", "mean", "std", "min", "max",
        "best_so_far", "sigma", "forward_time", "t_iter_s", "t_total_s"
    ])
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            iteration, pop, n_chunks, mean, std, rmin, rmax,
            best_so_far, sigma_now, forward_time, t_iter_sec, t_total_sec
        ])

def _save_best_traj(log_dir: str, best_traj: np.ndarray):
    np.save(os.path.join(log_dir, "best_traj.npy"), best_traj)

def _save_best_qpos(log_dir: str, best_qpos: np.ndarray):
    np.save(os.path.join(log_dir, "best_qpos.npy"), best_qpos)

def _save_run_config(log_dir: str, cfg: dict):
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ----------------------------
# CMA-ES checkpoint helpers
# ----------------------------
def _ckpt_dir(work_dir: Optional[str], trial_name: Optional[str]) -> Optional[str]:
    if work_dir is None or trial_name is None:
        return None
    return os.path.join(work_dir, trial_name)

def _ckpt_paths(work_dir: Optional[str], trial_name: Optional[str]):
    d = _ckpt_dir(work_dir, trial_name)
    if d is None:
        return None, None
    _ensure_dir(d)
    return os.path.join(d, "cmaes_ckpt.pkl"), os.path.join(d, "resume_meta.json")

def _save_cma_ckpt(es, work_dir: Optional[str], trial_name: Optional[str], iter_idx: int, best_reward: float):
    pkl_path, meta_path = _ckpt_paths(work_dir, trial_name)
    if pkl_path is None:
        return
    # Save CMA-ES opaque state
    with open(pkl_path, "wb") as f:
        f.write(es.pickle_dumps())
    # Save minimal meta so logs can continue with correct iteration index
    with open(meta_path, "w") as f:
        json.dump({
            "iter": iter_idx,
            "best_reward": best_reward,
    }, f)

def _load_cma_ckpt(work_dir: Optional[str], trial_name: Optional[str]):
    pkl_path, meta_path = _ckpt_paths(work_dir, trial_name)
    if pkl_path is None:
        return None, 0, -np.inf
    if not os.path.exists(pkl_path):
        return None, 0, -np.inf
    with open(pkl_path, "rb") as f:
        es = pickle.load(f)
    start_iter = 0
    best_reward = -np.inf
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                content = json.load(f)
            start_iter = int(content.get("iter", 0)) + 1
            best_reward = float(content.get("best_reward", -np.inf))
        except Exception:
            start_iter = 0
            best_reward = -np.inf
    return es, start_iter, best_reward


# ----------------------------
# CMA-ES optimization (general)
# ----------------------------
def _infer_act_dim(env) -> Optional[int]:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "act_dim"):
        return int(env.act_dim)
    if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
        return int(env.action_space.shape[0])
    return None

def _infer_n_steps(env) -> Optional[int]:
    for attr in ("n_steps", "traj_len", "horizon", "T"):
        if hasattr(env, attr):
            return int(getattr(env, attr))
    return None


def optimize_trajectory(
    env,
    n_steps: Optional[int] = None,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.01,
    l2_bound: Optional[float] = None,
    angle_bound: Optional[float] = None,
    max_iters: int = 200,
    seed: int = 42,
    log_dir: Optional[str] = None,
    # NEW: checkpointing
    work_dir: Optional[str] = None,
    trial_name: Optional[str] = None,
    resume: bool = False,
    save_every: int = 1,
    scene_version: int = 1,
    use_last_state_reward: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Adds CMA-ES checkpointing via (work_dir/trial_name)/cmaes_ckpt.pkl.
    - If resume=True and the file exists, loads the CMA state and continues.
    - Saves checkpoint every `save_every` iters and at the end.

    Other behavior: general shapes, logging, and optional bound inference unchanged.
    """
    # Resolve shapes
    if act_dim is None:
        act_dim = _infer_act_dim(env)
    if n_steps is None:
        n_steps = _infer_n_steps(env)
    if act_dim is None:
        raise ValueError("act_dim could not be inferred; please pass act_dim explicitly.")
    if n_steps is None:
        raise ValueError("n_steps could not be inferred; please pass n_steps explicitly.")

    # Resolve l2 bound (optional)
    if l2_bound is None and hasattr(env, "l2_bound"):
        l2_bound = float(getattr(env, "l2_bound"))

    # Resolve log_dir
    if log_dir is None and hasattr(env, "log_dir"):
        log_dir = getattr(env, "log_dir")
    if log_dir is not None:
        _ensure_dir(log_dir)
        _save_run_config(log_dir, {
            "n_envs": getattr(env, "n_envs", None),
            "n_steps": n_steps,
            "n_steps_sub": getattr(env, "_cmaes_n_steps_sub", None),
            "eval_version": getattr(env, "_cmaes_eval_version", None),
            "act_dim": act_dim,
            "popsize": popsize,
            "sigma0": sigma0,
            "per_comp_bound": (float(per_comp_bound) if np.isscalar(per_comp_bound)
                               else (list(per_comp_bound) if per_comp_bound is not None else None)),
            "l2_bound": l2_bound,
            "angle_bound": angle_bound,
            "max_iters": max_iters,
            "seed": seed,
            "work_dir": work_dir,
            "trial_name": trial_name,
            "scene_version": scene_version,
            "use_last_state_reward": use_last_state_reward,
        })

    dim = n_steps * act_dim
    if scene_version == 1:
        pcb = _as_per_comp_array(per_comp_bound, act_dim)
    elif scene_version == 2:
        pcb = _as_per_comp_array(per_comp_bound, act_dim // 2)
        pcb_angle = _as_per_comp_array(angle_bound, act_dim // 2)
        pcb = np.concatenate([pcb, pcb_angle])

    print(f'Bound: {pcb}')
    lower, upper = [], []
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    print(f'Max moving distance {l2_bound}x{n_steps}={l2_bound * n_steps} m for each control point')

    best_traj = None
    best_reward = -np.inf

    # Try to resume CMA-ES
    es = None
    start_iter = 0
    if resume:
        es_loaded, start_iter, best_reward_loaded = _load_cma_ckpt(work_dir, trial_name)
        if es_loaded is not None:
            es = es_loaded
            best_reward = best_reward_loaded
            # quick sanity: check dimension matches
            if getattr(es, "N", dim) != dim:
                raise ValueError(f"Loaded CMA-ES dimension {getattr(es, 'N', None)} "
                                 f"does not match expected dim {dim}.")
            # Note: popsize is internal in es.opts; we trust the checkpoint.
            print(f"[resume] Loaded CMA-ES from iteration {start_iter} "
                  f"with dim={dim}, expected max_iters={max_iters}, "
                  f"loaded best reward {best_reward_loaded:.4f}.")
        else:
            print("[resume] No checkpoint found; starting fresh.")

    # Fresh CMA-ES if not resumed
    if es is None:
        es = cma.CMAEvolutionStrategy(
            x0=[0.0] * dim,
            sigma0=sigma0,
            inopts={
                'bounds': [lower, upper],
                'popsize': popsize,
                'seed': seed,
                'CMA_elitist': True,
                'verb_disp': 0,
            }
        )

    assert es.popsize == popsize, f"CMA-ES popsize {es.popsize} != expected {popsize}"

    batch_size = env.n_envs
    it = start_iter
    t0_all = time.time()

    # If resuming, keep the previous total time in resume_meta (optional)
    print(f"{'iter':>5} | {'pop':>4} | {'chunks':>6} | {'mean':>8} | {'std':>8} | "
          f"{'min':>8} | {'max':>8} | {'best':>8} | {'sigma':>7} | {'t_iter(s)':>8} | {'t_total(s)':>9}")

    while it < max_iters:
        t_iter = time.time()
        X = es.ask()    # (popsize, n_steps * act_dim)
        pop = len(X)
        n_chunks = (pop + batch_size - 1) // batch_size
        qpos_list = list()

        all_final_rewards = []
        all_cum_rewards = []
        forward_time = 0.0
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, pcb, l2_bound, scene_version=scene_version)
                trajs.append(tr)
            out = evaluate_batch(env, trajs)
            if scene_version == 2:
                qpos_list.extend(env.qpos_seq.tolist())
            if out.get('final_reward') is not None:
                all_final_rewards.extend(out['final_reward'].tolist())
            if out.get('cum_reward') is not None:
                all_cum_rewards.extend(out['cum_reward'].tolist())
            if out.get('forward_time') is not None:
                forward_time += out['forward_time']
            print(f"  └─ chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_cum_rewards, dtype=np.float32)
        assert all_rewards.shape[0] == len(X), f"all_rewards {all_rewards.shape[0]} vs X {len(X)} length mismatch"

        # Log raw rewards for this generation
        if log_dir is not None:
            _append_rewards(
                log_dir, it,
                np.asarray(all_final_rewards, dtype=np.float32), 
                np.asarray(all_cum_rewards, dtype=np.float32)
            )

        # CMA-ES minimizes; negate to maximize reward
        if use_last_state_reward:
            es.tell(X, (-np.asarray(all_final_rewards, dtype=np.float32)).tolist())
        else:
            es.tell(X, (-all_rewards).tolist())

        # Track best of gen
        gen_best_idx = int(np.argmax(all_rewards))
        gen_best_reward = float(all_rewards[gen_best_idx])
        if scene_version == 1:
            gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
            gen_best_traj = project_deltas(
                reshape_to_traj(gen_best_x, n_steps, act_dim),
                pcb, l2_bound, scene_version=scene_version
            )
        elif scene_version == 2:
            qpos_array = np.asarray(qpos_list, dtype=np.float32)
            assert qpos_array.shape[0] == len(X), f"qpos_array {qpos_array.shape[0]} and X lengths {len(X)} do not match"
            gen_best_qpos = np.asarray(qpos_array[gen_best_idx], dtype=np.float32)

            gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
            gen_best_traj = project_deltas(
                reshape_to_traj(gen_best_x, n_steps, act_dim),
                pcb, l2_bound, scene_version=scene_version
            )

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_traj = gen_best_traj.copy()
            if log_dir is not None:
                _save_best_traj(log_dir, best_traj)
                if scene_version == 2:
                    _save_best_qpos(log_dir, gen_best_qpos)

        # Iteration summary
        m = float(all_rewards.mean()) if all_rewards.size else float('nan')
        s = float(all_rewards.std()) if all_rewards.size else float('nan')
        mn = float(all_rewards.min()) if all_rewards.size else float('nan')
        mx = float(all_rewards.max()) if all_rewards.size else float('nan')
        try:
            sigma_now = float(es.sigma)
        except Exception:
            sigma_now = float(es.sigma0) if hasattr(es, 'sigma0') else float('nan')

        t_iter_sec = time.time() - t_iter
        t_total_sec = time.time() - t0_all

        print(f"{it:5d} | {pop:4d} | {n_chunks:6d} | {m:8.4f} | {s:8.4f} | "
              f"{mn:8.4f} | {mx:8.4f} | {best_reward:8.4f} | {sigma_now:7.4f} | "
              f"{t_iter_sec:8.3f} | {t_total_sec:9.3f}")

        if log_dir is not None:
            _append_summary(log_dir, it, pop, n_chunks, m, s, mn, mx, best_reward, sigma_now, forward_time, t_iter_sec, t_total_sec)

        # Save checkpoint periodically
        if save_every > 0 and (it % save_every == 0):
            _save_cma_ckpt(es, work_dir, trial_name, it, best_reward)

        it += 1

    # Final checkpoint
    _save_cma_ckpt(es, work_dir, trial_name, it - 1, best_reward)

    return best_traj, best_reward


# ----------------------------
# Example usage
# ----------------------------

def _build_env(
        task: str, log_dir: str, n_envs: int,
        vis_traj: Optional[str] = None, gui: bool = False,
        scene_version: int = 1, raytracer: bool = False
    ) -> Train_Env:
    task = task.lower()
    task_to_env = {
        "wiring_ring": Train_Env_Wiring_ring,
        "wiring_post": Train_Env_Wiring_post,
        "lifting":   Train_Env_Lifting,
        "slingshot": Train_Env_Slingshot,
        "wireart":   Train_Env_Wireart,
        "coiling":   Train_Env_Coiling,
        "gathering": Train_Env_Gathering,
        "separation": Train_Env_Separation,
        "wrapping":  Train_Env_Wrapping,
    }
    if task not in task_to_env:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(task_to_env.keys())}")
    EnvCls = task_to_env[task]
    if vis_traj is None:
        camera = False
    else:
        n_envs = 1
        camera = True

    return EnvCls(task=task, log_dir=log_dir, n_envs=n_envs, GUI=gui, camera=camera, raytracer=raytracer, scene_version=scene_version)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--task', type=str, default="wiring",
        help="Task / environment to optimize."
    )
    parser.add_argument(
        '--seed', type=int, default=123,
    )
    parser.add_argument(
        '--n_envs', type=int, default=10,
    )
    parser.add_argument(
        '--max_iter', type=int, default=20,
    )
    parser.add_argument(
        '--n_steps', type=int, default=10,
    )
    parser.add_argument(
        '--n_steps_sub', type=int, default=10,
    )
    parser.add_argument(
        '--eval_version', type=int, default=2,
    )
    parser.add_argument(
        '--use_last_state_reward', action='store_true',
        help="Whether to use the last state reward instead of accumulated reward."
    )
    parser.add_argument(
        '--vis_traj', type=str, default=None, 
        help="Path to saved trajectory .npy for visualization. If None, runs optimization."
    )
    parser.add_argument(
        '--bound', type=float, default=0.1,
        help="Per-step L2 bound for each control point."
    )
    parser.add_argument(
        '--angle_bound', type=float, default=10.0,
        help="Per-step angle bound for each control point."
    )
    parser.add_argument(
        '--sigma', type=float, default=0.005
    )
    parser.add_argument(
        '--exp_name', type=str, default=None,
    )
    parser.add_argument(
        '--scene_version', type=int, default=1,
    )
    parser.add_argument('--gui', action='store_true', help="Whether to show GUI.")
    parser.add_argument('--raytracer', '-r', action='store_true', help='Enable raytracer for rendering')
    args = parser.parse_args()

    exp_name = f"{args.exp_name}" if args.exp_name is not None else "cmaes"
    trial_name = f"trial_{args.task}/{exp_name}"
    log_dir = f"logs/{args.task}/{exp_name}"
    env = _build_env(args.task, log_dir, args.n_envs, args.vis_traj, args.gui, args.scene_version, args.raytracer)
    env.init_cmaes_env(
        n_steps_sub=args.n_steps_sub,
        eval_version=args.eval_version,
    )
    print(f'CMA-ES n_steps_sub: {env._cmaes_n_steps_sub}, eval_version: {env._cmaes_eval_version}')
    n_steps = args.n_steps

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.vis_traj is None:

        assert not env.requires_grad, "CMA-ES optimization does not need env with gradients."
        
        best_traj, best_reward = optimize_trajectory(
            env,
            n_steps=n_steps,
            act_dim=None,           # infer if available
            popsize=100,
            sigma0=args.sigma,
            per_comp_bound=args.bound,
            l2_bound=args.bound,          # use env.l2_bound if present
            angle_bound=args.angle_bound,
            max_iters=args.max_iter,
            seed=args.seed,
            log_dir=log_dir,
            # NEW: checkpoint controls
            work_dir="checkpoints",
            trial_name=trial_name,
            resume=True,            # set True to load if checkpoint exists
            save_every=1,           # save each generation
            scene_version=args.scene_version,
            use_last_state_reward=args.use_last_state_reward,
        )

    else:

        print(f'Visualizing CMA-ES trajectory from {args.vis_traj}')
        evaluate_single(env, np.load(args.vis_traj), log_dir, n_steps)
