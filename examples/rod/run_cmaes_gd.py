import os
import json
import csv
import time
import pickle
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
                   per_comp_bound: Optional[Sequence[float]],
                   max_l2_per_step: Optional[float]) -> np.ndarray:
    n_steps, act_dim = traj.shape
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    if np.isfinite(pcb).any():
        traj = np.clip(traj, -pcb, pcb)
    if max_l2_per_step is not None and np.isfinite(max_l2_per_step):
        norms = np.linalg.norm(traj, axis=1, keepdims=True)
        scale = np.ones_like(norms, dtype=traj.dtype)
        over = norms > max_l2_per_step
        scale[over] = max_l2_per_step / (norms[over] + 1e-12)
        traj = traj * scale
    return traj


# ----------------------------
# Parallel evaluation (batch)
# ----------------------------
def evaluate_batch(env: Train_Env, traj_list: List[np.ndarray], ratio: float, per_comp_bound, l2_bound) -> Tuple[np.ndarray, np.ndarray]:
    """ Evaluate all trajectories after taking one GD step """
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr
    if env.requires_grad:
        # before evaluation, we take a gradient step
        delta_base = env.gd_one_step(trajs.copy())
        print(f'traj: {np.abs(trajs).mean(0).mean(0)}')

        deltas = env.adaptive_scale(trajs, delta_base, ratio=ratio)
        print(f'delta: {np.abs(deltas).mean(0).mean(0)}\t with ratio: {ratio}')

        # Update trajs
        trajs_ = trajs.copy() + deltas

        for i in range(n_envs):
            trajs[i] = project_deltas(trajs_[i], per_comp_bound, l2_bound)
    rewards = env.eval_traj(trajs)
    return np.asarray(rewards, dtype=np.float32), np.asarray(trajs, dtype=np.float32)

def evaluate_batch_ratios(env: Train_Env, traj_list: List[np.ndarray], ratios: List[float], per_comp_bound, l2_bound) -> Tuple[np.ndarray, np.ndarray]:
    """ Evaluate all trajectories after taking one GD step """
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr
    if env.requires_grad:
        trajs_grad = np.zeros((n_envs, len(ratios), n_steps, act_dim), dtype=np.float32)
        rewards_grad = np.zeros((n_envs, len(ratios)), dtype=np.float32)
        # before evaluation, we take a gradient step
        delta_base = env.gd_one_step(trajs.copy())
        print(f'traj: {np.abs(trajs).mean(0).mean(0)}')
        for r_idx in range(len(ratios)):
            # ensure each delta is within ratio x trajs_origin
            deltas = env.adaptive_scale(trajs, delta_base, ratio=ratios[r_idx])
            print(f'delta: {np.abs(deltas).mean(0).mean(0)}\t with ratio: {ratios[r_idx]}')

            # Update trajs
            trajs_ = trajs.copy() + deltas

            for env_idx in range(n_envs):
                trajs_grad[env_idx, r_idx] = project_deltas(trajs_[env_idx], per_comp_bound, l2_bound)            
            rewards_grad[:, r_idx] = env.eval_traj(trajs_grad[:, r_idx, :, :])
        # Select the best reward and corresponding traj for each ratio
        best_indices = np.argmax(rewards_grad, axis=1)  # (n_envs,)
        rewards = rewards_grad[np.arange(n_envs), best_indices] # (n_envs,)
        for env_idx in range(n_envs):
            trajs[env_idx] = trajs_grad[env_idx, best_indices[env_idx]]
    else:
        rewards = env.eval_traj(trajs)
    # (n_envs, ), (n_envs, n_steps, act_dim)
    return np.asarray(rewards, dtype=np.float32), np.asarray(trajs, dtype=np.float32)

def evaluate_batch_pre(env: Train_Env, traj_list: List[np.ndarray]) -> np.ndarray:
    """ Evaluate all trajectories """
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr
    rewards = env.eval_traj(trajs)
    return np.asarray(rewards, dtype=np.float32)

def evaluate_single(env: Train_Env, traj: np.ndarray, log_dir: str) -> float:
    rewards = env.eval_traj(traj[None, ...], debug=True)
    print(f'Single traj reward: {rewards[0]:.4f}')
    env.save_animation(save_dir=log_dir)
    return rewards[0]

def cosine_learning_rate_scheduler(base_lr, cur_iter, max_iter, min_lr=1e-6):
    if cur_iter >= max_iter:
        return min_lr
    cosine_decay = 0.5 * (1 + np.cos(np.pi * cur_iter / max_iter))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    return lr

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

def _append_rewards(log_dir: str, iteration: int, rewards: np.ndarray):
    path = os.path.join(log_dir, "rewards_all.csv")
    _maybe_write_header(path, ["iter", "idx", "reward"])
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for idx, r in enumerate(rewards.tolist()):
            w.writerow([iteration, idx, float(r)])

def _append_summary(log_dir: str, iteration: int, pop: int, n_chunks: int,
                    mean: float, std: float, rmin: float, rmax: float,
                    best_so_far: float, sigma_now: float,
                    t_iter_sec: float, t_total_sec: float):
    path = os.path.join(log_dir, "summary.csv")
    _maybe_write_header(path, [
        "iter", "pop", "chunks", "mean", "std", "min", "max",
        "best_so_far", "sigma", "t_iter_s", "t_total_s"
    ])
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            iteration, pop, n_chunks, mean, std, rmin, rmax,
            best_so_far, sigma_now, t_iter_sec, t_total_sec
        ])

def _save_best_traj(log_dir: str, best_traj: np.ndarray):
    np.save(os.path.join(log_dir, "best_traj.npy"), best_traj)

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


def optimize_trajectory_v1(
    env: Train_Env,
    n_steps: Optional[int] = None,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.01,
    l2_bound: Optional[float] = None,
    max_iters: int = 200,
    seed: int = 42,
    log_dir: Optional[str] = None,
    # NEW: checkpointing
    work_dir: Optional[str] = None,
    trial_name: Optional[str] = None,
    resume: bool = False,
    save_every: int = 1,
    # NEW
    scale_method: Optional[str] = None,
    exp_base: float = 1.1,
    ratio: float = 0.1,
    **kwargs,
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
            "n_steps": n_steps,
            "act_dim": act_dim,
            "popsize": popsize,
            "sigma0": sigma0,
            "per_comp_bound": (float(per_comp_bound) if np.isscalar(per_comp_bound)
                               else (list(per_comp_bound) if per_comp_bound is not None else None)),
            "l2_bound": l2_bound,
            "max_iters": max_iters,
            "seed": seed,
            "work_dir": work_dir,
            "trial_name": trial_name,
            "scale_method": scale_method,
            "ratio": ratio,
        })

    # Construct traj_optim if needed
    if env.requires_grad:
        env.construct_traj_optim(max_ddist=l2_bound)
        env.construct_scale_array(scale_method=scale_method, n_steps=n_steps, exp_base=exp_base)

    dim = n_steps * act_dim
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    lower, upper = [], []
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    print(f'Max moving distance {l2_bound}x{n_steps}={l2_bound * n_steps} m for each control point')
    print(f'Using V1: CMA-ES + GD with ratio {ratio}')

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

        all_rewards = []
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                # (batch_size, n_steps, act_dim)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)
            rewards, _ = evaluate_batch(env, trajs, ratio, per_comp_bound, l2_bound)
            all_rewards.extend(rewards.tolist())
            print(f"  └─ chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_rewards, dtype=np.float32)

        # Log raw rewards for this generation
        if log_dir is not None:
            _append_rewards(log_dir, it, all_rewards)

        # CMA-ES minimizes; negate to maximize reward
        es.tell(X, (-all_rewards).tolist())

        # Track best of gen
        gen_best_idx = int(np.argmax(all_rewards))
        gen_best_reward = float(all_rewards[gen_best_idx])
        gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)  # NOTE: use updated traj
        gen_best_traj = project_deltas(
            reshape_to_traj(gen_best_x, n_steps, act_dim),
            per_comp_bound, l2_bound
        )

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_traj = gen_best_traj.copy()
            if log_dir is not None:
                _save_best_traj(log_dir, best_traj)

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
            _append_summary(log_dir, it, pop, n_chunks, m, s, mn, mx, best_reward, sigma_now, t_iter_sec, t_total_sec)

        # Save checkpoint periodically
        if save_every > 0 and (it % save_every == 0):
            _save_cma_ckpt(es, work_dir, trial_name, it, best_reward)

        it += 1

    # Final checkpoint
    _save_cma_ckpt(es, work_dir, trial_name, it - 1, best_reward)

    return best_traj, best_reward

def optimize_trajectory_v2(
    env: Train_Env,
    n_steps: Optional[int] = None,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.01,
    l2_bound: Optional[float] = None,
    max_iters: int = 200,
    seed: int = 42,
    log_dir: Optional[str] = None,
    # NEW: checkpointing
    work_dir: Optional[str] = None,
    trial_name: Optional[str] = None,
    resume: bool = False,
    save_every: int = 1,
    # NEW
    scale_method: Optional[str] = None,
    exp_base: float = 1.1,
    ratio: List[float] = [0.1],
    min_ratio: float = 1e-6,
    n_top_ratio: float = 0.2,
    scheduler: Optional[str] = None,
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
            "n_steps": n_steps,
            "act_dim": act_dim,
            "popsize": popsize,
            "sigma0": sigma0,
            "per_comp_bound": (float(per_comp_bound) if np.isscalar(per_comp_bound)
                               else (list(per_comp_bound) if per_comp_bound is not None else None)),
            "l2_bound": l2_bound,
            "max_iters": max_iters,
            "seed": seed,
            "work_dir": work_dir,
            "trial_name": trial_name,
            "scale_method": scale_method,
            "ratio": ratio,
            "min_ratio": min_ratio,
            "n_top_ratio": n_top_ratio,
            "scheduler": scheduler,
        })

    # Construct traj_optim if needed
    if env.requires_grad:
        env.construct_traj_optim(max_ddist=l2_bound)
        env.construct_scale_array(scale_method=scale_method, n_steps=n_steps, exp_base=exp_base)

    dim = n_steps * act_dim
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    lower, upper = [], []
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    print(f'Max moving distance {l2_bound}x{n_steps}={l2_bound * n_steps} m for each control point')
    print(f'Using V2: CMA-ES + GD with multiple ratios: {ratio}')

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

        # 1. First, evaluate all trajs without GD step
        all_rewards = []
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                # (batch_size, n_steps, act_dim)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)
            rewards = evaluate_batch_pre(env, trajs)
            all_rewards.extend(rewards.tolist())
            print(f"  └─ [pre] chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_rewards, dtype=np.float32)

        # 2. Sort the trajectories by reward and apply GD step to the top n_top_ratio
        n_top = max(1, int(n_top_ratio * pop))
        sorted_indices = np.argsort(-all_rewards)
        top_indices = sorted_indices[:n_top]

        X_top = [X[i] for i in top_indices]
        n_chunks_new = (max(1, (n_top + batch_size - 1) // batch_size))

        # 3. Evaluate the augmented trajectories with GD step
        if scheduler is None:
            ratio_it = ratio
        elif scheduler == "cosine":
            ratio_it = list()
            for r in ratio:
                # apply cosine scheduler
                r_ = cosine_learning_rate_scheduler(r, it, max_iters, min_ratio)
                ratio_it.append(r_)
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")

        all_rewards_new = []
        all_updated_trajs = []
        for ci, start in enumerate(range(0, n_top, batch_size), 1):
            t_chunk = time.time()
            chunk = X_top[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                # (batch_size, n_steps, act_dim)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)
            rewards, updated_trajs = evaluate_batch_ratios(env, trajs, ratio_it, per_comp_bound, l2_bound)
            all_rewards_new.extend(rewards.tolist())
            all_updated_trajs.append(updated_trajs)
            print(f"  └─ [selected] chunk {ci:>2}/{n_chunks_new}: {len(chunk):>3} evals | ratio={[float(i) for i in ratio_it]} | t={time.time() - t_chunk:.3f}s")

        all_rewards_new = np.asarray(all_rewards_new, dtype=np.float32)
        assert len(all_rewards_new) == n_top, f"Expected {n_top} new reward, got {len(all_rewards_new)}"
        # combine the new rewards with the original rewards
        all_rewards = np.concatenate([all_rewards, all_rewards_new], axis=0)

        # (n_augmented, n_steps, act_dim)
        all_updated_trajs = np.concatenate(all_updated_trajs, axis=0)
        assert len(all_updated_trajs) == n_top, f"Expected {n_top} updated trajs, got {len(all_updated_trajs)}"
        all_updated_trajs = all_updated_trajs.reshape(n_top, n_steps * act_dim)
        # combine the updated trajs with the original trajs
        X = X + list(all_updated_trajs)
        assert len(all_rewards) == len(X), f"Combined rewards length {len(all_rewards)} != trajs length {len(X)}"

        # Log raw rewards for this generation
        if log_dir is not None:
            _append_rewards(log_dir, it, all_rewards)

        # CMA-ES minimizes; negate to maximize reward
        es.tell(X, (-all_rewards).tolist())

        # Track best of gen
        gen_best_idx = int(np.argmax(all_rewards))
        gen_best_reward = float(all_rewards[gen_best_idx])
        gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
        gen_best_traj = project_deltas(
            reshape_to_traj(gen_best_x, n_steps, act_dim),
            per_comp_bound, l2_bound
        )

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_traj = gen_best_traj.copy()
            if log_dir is not None:
                _save_best_traj(log_dir, best_traj)

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
            _append_summary(log_dir, it, pop, n_chunks, m, s, mn, mx, best_reward, sigma_now, t_iter_sec, t_total_sec)

        # Save checkpoint periodically
        if save_every > 0 and (it % save_every == 0):
            _save_cma_ckpt(es, work_dir, trial_name, it, best_reward)

        it += 1

    # Final checkpoint
    _save_cma_ckpt(es, work_dir, trial_name, it - 1, best_reward)

    return best_traj, best_reward

def optimize_trajectory_v3(
    env: Train_Env,
    n_steps: Optional[int] = None,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.01,
    l2_bound: Optional[float] = None,
    max_iters: int = 200,
    seed: int = 42,
    log_dir: Optional[str] = None,
    # NEW: checkpointing
    work_dir: Optional[str] = None,
    trial_name: Optional[str] = None,
    resume: bool = False,
    save_every: int = 1,
    # NEW
    scale_method: Optional[str] = None,
    exp_base: float = 1.1,
    ratio: List[float] = [0.1],
    min_ratio: float = 1e-6,
    n_top_ratio: float = 0.2,
    scheduler: Optional[str] = None,
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
            "n_steps": n_steps,
            "act_dim": act_dim,
            "popsize": popsize,
            "sigma0": sigma0,
            "per_comp_bound": (float(per_comp_bound) if np.isscalar(per_comp_bound)
                               else (list(per_comp_bound) if per_comp_bound is not None else None)),
            "l2_bound": l2_bound,
            "max_iters": max_iters,
            "seed": seed,
            "work_dir": work_dir,
            "trial_name": trial_name,
            "scale_method": scale_method,
            "ratio": ratio,
            "min_ratio": min_ratio,
            "n_top_ratio": n_top_ratio,
            "scheduler": scheduler,
        })

    # Construct traj_optim if needed
    if env.requires_grad:
        env.construct_traj_optim(max_ddist=l2_bound)
        env.construct_scale_array(scale_method=scale_method, n_steps=n_steps, exp_base=exp_base)

    dim = n_steps * act_dim
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    lower, upper = [], []
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    print(f'Max moving distance {l2_bound}x{n_steps}={l2_bound * n_steps} m for each control point')
    print(f'Using V3: CMA-ES + GD with multiple ratios: {ratio} without modifying CMA-ES samples')

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

        # 1. First, evaluate all trajs without GD step
        all_rewards = []
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                # (batch_size, n_steps, act_dim)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)
            rewards = evaluate_batch_pre(env, trajs)
            all_rewards.extend(rewards.tolist())
            print(f"  └─ [pre] chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_rewards, dtype=np.float32)

        # 2. Sort the trajectories by reward and apply GD step to the top n_top_ratio
        n_top = max(1, int(n_top_ratio * pop))
        sorted_indices = np.argsort(-all_rewards)
        top_indices = sorted_indices[:n_top]

        X_top = [X[i] for i in top_indices]
        n_chunks_new = (max(1, (n_top + batch_size - 1) // batch_size))

        # 3. Evaluate the augmented trajectories with GD step
        if scheduler is None:
            ratio_it = ratio
        elif scheduler == "cosine":
            ratio_it = list()
            for r in ratio:
                # apply cosine scheduler
                r_ = cosine_learning_rate_scheduler(r, it, max_iters, min_ratio)
                ratio_it.append(r_)
        else:
            raise ValueError(f"Unknown scheduler '{scheduler}'")

        all_rewards_new = []
        all_updated_trajs = []
        for ci, start in enumerate(range(0, n_top, batch_size), 1):
            t_chunk = time.time()
            chunk = X_top[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                # (batch_size, n_steps, act_dim)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)
            rewards, updated_trajs = evaluate_batch_ratios(env, trajs, ratio_it, per_comp_bound, l2_bound)
            all_rewards_new.extend(rewards.tolist())
            all_updated_trajs.append(updated_trajs)
            print(f"  └─ [selected] chunk {ci:>2}/{n_chunks_new}: {len(chunk):>3} evals | ratio='{[float(i) for i in ratio_it]}' | t={time.time() - t_chunk:.3f}s")

        all_rewards_new = np.asarray(all_rewards_new, dtype=np.float32)
        assert len(all_rewards_new) == n_top, f"Expected {n_top} new reward, got {len(all_rewards_new)}"

        # check whether to use the new rewards (if better) or not for the top indices
        use_new = all_rewards_new > all_rewards[top_indices]    # (n_top, )
        all_rewards_new = np.where(use_new, all_rewards_new, all_rewards[top_indices])
        # update all_rewards for the top indices
        all_rewards[top_indices] = all_rewards_new
        assert len(all_rewards) == popsize, f"Expected {popsize} final rewards, got {len(all_rewards)}"

        # Log raw rewards for this generation
        if log_dir is not None:
            _append_rewards(log_dir, it, all_rewards)

        # CMA-ES minimizes; negate to maximize reward
        es.tell(X, (-all_rewards).tolist())

        # update X for the top indices for logging purposes
        all_updated_trajs = np.concatenate(all_updated_trajs, axis=0)
        assert len(all_updated_trajs) == n_top, f"Expected {n_top} updated trajs, got {len(all_updated_trajs)}"
        all_updated_trajs = all_updated_trajs.reshape(n_top, n_steps * act_dim)
        for idx, use, updated_traj in zip(top_indices, use_new, all_updated_trajs):
            if use:
                X[idx] = updated_traj

        # Track best of gen
        gen_best_idx = int(np.argmax(all_rewards))
        gen_best_reward = float(all_rewards[gen_best_idx])
        gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
        gen_best_traj = project_deltas(
            reshape_to_traj(gen_best_x, n_steps, act_dim),
            per_comp_bound, l2_bound
        )

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_traj = gen_best_traj.copy()
            if log_dir is not None:
                _save_best_traj(log_dir, best_traj)

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
            _append_summary(log_dir, it, pop, n_chunks, m, s, mn, mx, best_reward, sigma_now, t_iter_sec, t_total_sec)

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

def _build_env(task: str, log_dir: str, n_envs: int, vis_traj: Optional[str] = None, gui: bool = False) -> Train_Env:
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
        require_grad = True
        camera = False          # do not build cameras
    else:
        n_envs = 1
        require_grad = False
        camera = True           # build cameras for visualization

    return EnvCls(task=task, log_dir=log_dir, n_envs=n_envs, GUI=gui, camera=camera, requires_grad=require_grad)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--task', type=str, default='wiring',
        help="Task / environment to optimize."
    )
    parser.add_argument(
        '--seed', type=int, default=123,
    )
    parser.add_argument(
        '--max_iter', type=int, default=20,
    )
    parser.add_argument(
        '--n_steps', type=int, default=10,
    )
    parser.add_argument(
        '--scale_method', type=str, default=None,
        choices=[None, 'linear', 'exp', 'custom']
    )
    parser.add_argument(
        '--scheduler', type=str, default=None,
    )
    parser.add_argument(
        '--version', type=int, default=1,
    )
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.1])
    parser.add_argument('--min_ratio', type=float, default=1e-6)
    parser.add_argument('--n_top_ratio', type=float, default=0.2)
    parser.add_argument(
        '--vis_traj', type=str, default=None, 
        help="Path to saved trajectory .npy for visualization. If None, runs optimization."
    )
    parser.add_argument(
        '--exp_name', type=str, default=None,
    )
    parser.add_argument('--gui', action='store_true', help="Whether to show GUI.")
    args = parser.parse_args()

    exp_name = f"{args.exp_name}" if args.exp_name is not None else "cmaes-gd"
    trial_name = f"trial_{args.task}/{exp_name}"
    log_dir = f"logs/{args.task}/{exp_name}"
    env = _build_env(args.task, log_dir, 10, args.vis_traj, args.gui)

    if args.vis_traj is None:

        assert env.requires_grad, "Env must be created with requires_grad=True for traj optim."

        n_steps = args.n_steps

        if args.version == 1:
            func = optimize_trajectory_v1
            # eval sampled sol -> GD updated for all -> combined both -> tell
            assert len(args.ratio) == 1, "Version 1 only supports single ratio value."
            ratio_arg = args.ratio[0]
        elif args.version == 2:
            func = optimize_trajectory_v2
            # eval sampled sol -> GD updated for top -> combined original + top -> tell
            ratio_arg = args.ratio
        elif args.version == 3:
            func = optimize_trajectory_v3
            # eval sampled sol -> GD updated for top -> use updated rewards only if better -> tell orignal sol
            ratio_arg = args.ratio
        else:
            raise ValueError(f"Unknown version {args.version}")

        best_traj, best_reward = func(
            env,
            n_steps=n_steps,
            act_dim=None,           # infer if available
            popsize=200,
            sigma0=0.005,
            per_comp_bound=0.1,
            l2_bound=0.1,          # use env.l2_bound if present
            max_iters=args.max_iter,
            seed=args.seed,
            log_dir=log_dir,
            # NEW: checkpoint controls
            work_dir="checkpoints",
            trial_name=trial_name,
            resume=True,                            # set True to load if checkpoint exists
            save_every=1,                           # save each generation
            scale_method=args.scale_method,         # None, 'linear', 'exp', 'custom'
            exp_base=1.1,                           # only used if scale_method=='exp'
            ratio=ratio_arg,                        # ratio between traj over delta
            min_ratio=args.min_ratio,
            n_top_ratio=args.n_top_ratio,
            scheduler=args.scheduler,
        )

    else:

        print(f'Visualizing CMA-ES+GD trajectory from {args.vis_traj}')
        evaluate_single(env, np.load(args.vis_traj), log_dir)
