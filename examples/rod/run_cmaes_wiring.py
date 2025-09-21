import os
import json
import csv
import time
from typing import Tuple, List, Optional, Sequence

import numpy as np
import cma

from train_env_wiring import Train_Env_Wiring  # keep if you still want the example main


# ----------------------------
# Helpers: shape & constraints
# ----------------------------
def reshape_to_traj(x: np.ndarray, n_steps: int, act_dim: int) -> np.ndarray:
    """
    x: flat vector of length n_steps*act_dim
    returns (n_steps, act_dim)
    """
    return x.reshape(n_steps, act_dim)


def _as_per_comp_array(per_comp_bound: Optional[Sequence[float]], act_dim: int) -> np.ndarray:
    """
    Accept a float or a sequence; return np.ndarray of shape (act_dim,)
    If None, return +inf bounds (i.e., no per-component clamp).
    """
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
    """
    Enforce optional per-component bounds and per-step L2 norm bounds on the trajectory.
    traj: (n_steps, act_dim)
    per_comp_bound: float or sequence of length act_dim, or None for no per-comp clamp
    max_l2_per_step: float or None for no L2 clamp
    """
    n_steps, act_dim = traj.shape

    # Per-component clamp (if provided)
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    if np.isfinite(pcb).any():
        traj = np.clip(traj, -pcb, pcb)

    # Per-step L2 clamp (if provided)
    if max_l2_per_step is not None and np.isfinite(max_l2_per_step):
        norms = np.linalg.norm(traj, axis=1, keepdims=True)  # (n_steps, 1)
        scale = np.ones_like(norms, dtype=traj.dtype)
        over = norms > max_l2_per_step
        # avoid divide-by-zero
        scale[over] = max_l2_per_step / (norms[over] + 1e-12)
        traj = traj * scale

    return traj


# ----------------------------
# Parallel evaluation (batch)
# ----------------------------
def evaluate_batch(env, traj_list: List[np.ndarray]) -> np.ndarray:
    """
    env: your multi-env. Must provide env.eval_traj(trajs) -> rewards
    traj_list: list of (n_steps, act_dim) arrays, length <= env.n_envs
    Returns: rewards np.array of shape (len(traj_list),)
    """
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]

    # Prepare a (n_envs, n_steps, act_dim) tensor; pad if needed
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr

    rewards = env.eval_traj(trajs)  # advances each env to the end of its traj
    return np.asarray(rewards, dtype=np.float32)


# ----------------------------
# Logging helpers
# ----------------------------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _maybe_write_header(path: str, header: List[str]):
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if needs_header:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def _append_rewards(log_dir: str, iteration: int, rewards: np.ndarray):
    """
    Appends rows to rewards_all.csv with columns:
    iter, idx, reward
    """
    path = os.path.join(log_dir, "rewards_all.csv")
    _maybe_write_header(path, ["iter", "idx", "reward"])
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for idx, r in enumerate(rewards.tolist()):
            writer.writerow([iteration, idx, float(r)])


def _append_summary(log_dir: str, iteration: int, pop: int, n_chunks: int,
                    mean: float, std: float, rmin: float, rmax: float,
                    best_so_far: float, sigma_now: float,
                    t_iter_sec: float, t_total_sec: float):
    """
    Appends a row to summary.csv with per-generation stats.
    """
    path = os.path.join(log_dir, "summary.csv")
    _maybe_write_header(path, [
        "iter", "pop", "chunks", "mean", "std", "min", "max",
        "best_so_far", "sigma", "t_iter_s", "t_total_s"
    ])
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            iteration, pop, n_chunks, mean, std, rmin, rmax,
            best_so_far, sigma_now, t_iter_sec, t_total_sec
        ])


def _save_best_traj(log_dir: str, best_traj: np.ndarray):
    np.save(os.path.join(log_dir, "best_traj.npy"), best_traj)


def _save_run_config(log_dir: str, cfg: dict):
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


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


def optimize_wiring_trajectory(
    env,
    n_steps: Optional[int] = None,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.03,
    l2_bound: Optional[float] = None,
    max_iters: int = 200,
    seed: int = 42,
    log_dir: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    General CMA-ES trajectory optimizer across tasks.

    Requirements the env must satisfy:
      - env.n_envs : int (parallel evaluation batch size)
      - env.eval_traj(trajs: (n_envs, n_steps, act_dim)) -> rewards (n_envs,)
    Optional env attributes used if not given as args:
      - env.l2_bound : float
      - env.action_dim or env.action_space.shape[0]
      - env.n_steps / env.traj_len / env.horizon / env.T
      - env.log_dir

    Args:
      n_steps: if None, try to infer from env; otherwise required.
      act_dim: if None, try to infer from env or from first candidate shape.
      per_comp_bound: float or sequence (len==act_dim) or None (no per-comp clamp)
      l2_bound: if None, will try env.l2_bound; can be None for no L2 clamp
      log_dir: where to write logs; if None, tries env.log_dir; if still None, logging disabled (prints only).
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
        # Save a one-time config snapshot
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
        })

    dim = n_steps * act_dim
    lower = []
    upper = []
    # Build bounds for CMA (per-component box bounds)
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    # CMA-ES setup
    es = cma.CMAEvolutionStrategy(
        x0=[0.0] * dim,
        sigma0=sigma0,
        inopts={
            'bounds': [lower, upper],
            'popsize': popsize,
            'seed': seed,
            'CMA_elitist': True,
            'verb_disp': 0,   # quieter
        }
    )

    best_traj = None
    best_reward = -np.inf

    batch_size = env.n_envs
    it = 0
    eval_count = 0
    t0_all = time.time()

    print(f"{'iter':>5} | {'pop':>4} | {'chunks':>6} | {'mean':>8} | {'std':>8} | "
          f"{'min':>8} | {'max':>8} | {'best':>8} | {'sigma':>7} | {'t_iter(s)':>8} | {'t_total(s)':>9}")

    while not es.stop() and it < max_iters:
        t_iter = time.time()
        X = es.ask()  # list of candidate flat vectors
        pop = len(X)
        n_chunks = (pop + batch_size - 1) // batch_size

        # Evaluate in chunks
        all_rewards = []
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)

            rewards = evaluate_batch(env, trajs)
            all_rewards.extend(rewards.tolist())

            print(f"  └─ chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_rewards, dtype=np.float32)
        eval_count += all_rewards.size

        # Log raw rewards for this generation
        if log_dir is not None:
            _append_rewards(log_dir, it, all_rewards)

        # CMA-ES minimizes; negate to maximize reward
        losses = (-all_rewards).tolist()
        es.tell(X, losses)

        # Track best
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

        # Log summary line
        if log_dir is not None:
            _append_summary(log_dir, it, pop, n_chunks, m, s, mn, mx, best_reward, sigma_now, t_iter_sec, t_total_sec)

        it += 1

    return best_traj, best_reward


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example env (your wiring task)
    env = Train_Env_Wiring(task='wiring', log_dir="logs/wiring", n_envs=10)

    # If your task has a natural step count, you can pass it; else rely on inference.
    n_steps = 10

    best_traj, best_reward = optimize_wiring_trajectory(
        env,
        n_steps=n_steps,        # or None to try to infer
        act_dim=None,           # try to infer from env
        popsize=100,
        sigma0=0.01,
        per_comp_bound=0.03,    # float or sequence length act_dim
        l2_bound=None,          # pull from env.l2_bound if present
        max_iters=15,
        seed=123,
        log_dir="logs/wiring"   # overrides env.log_dir if set
    )

    # Optionally visualize:
    # env.eval_traj(best_traj[None, ...])
