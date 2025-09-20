import numpy as np
import cma
from typing import Tuple, List
from train_env_wiring import Train_Env_Wiring
import time

# ----------------------------
# Helper: reshape & constrain
# ----------------------------
def reshape_to_traj(x: np.ndarray, n_steps: int) -> np.ndarray:
    """
    x: flat vector of length n_steps*3
    returns (n_steps, 3)
    """
    return x.reshape(n_steps, 3)

def project_deltas(traj: np.ndarray,
                   per_comp_bound: float,
                   max_l2_per_step: float) -> np.ndarray:
    """
    Enforce per-component bounds and per-step L2 norm bounds on the trajectory.
    traj: (n_steps, 3)
    """
    # Per-component clamp
    traj = np.clip(traj, -per_comp_bound, per_comp_bound)

    # Per-step L2 clamp
    norms = np.linalg.norm(traj, axis=1, keepdims=True)  # (n_steps, 1)
    scale = np.ones_like(norms)
    over = norms > max_l2_per_step
    scale[over] = max_l2_per_step / (norms[over] + 1e-12)
    traj = traj * scale
    return traj

# ----------------------------
# Parallel evaluation (batch)
# ----------------------------
def evaluate_batch(env,
                   traj_list: List[np.ndarray]) -> np.ndarray:
    """
    env: your Train_Env_Wiring (multi-env)
    traj_list: list of (n_steps, 3) arrays, length <= env.n_envs
    Returns: rewards np.array of shape (len(traj_list),)
    """
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = 3

    # Prepare a (n_envs, n_steps, 3) tensor; pad if needed
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr

    # Run the whole batch in the simulator
    env.eval_traj(trajs)  # advances the scene to the end of each traj

    # Get terminal rewards for all envs and slice to actual batch size
    rewards = np.asarray(env.reward(), dtype=np.float32)[:len(traj_list)]
    return rewards

# ----------------------------
# CMA-ES optimization
# ----------------------------
def optimize_wiring_trajectory(env,
                               n_steps: int = 20,
                               popsize: int = None,
                               sigma0: float = 0.01,
                               per_comp_bound: float = 0.03,
                               l2_bound: float = 0.04,
                               max_iters: int = 200,
                               seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    env: Train_Env_Wiring instance (multi-envs). Its n_envs is the per-iteration batch size.
    n_steps: number of delta waypoints; each applied over 0.25 s (50 sim steps at 0.005 s).
    popsize: CMA population size. Defaults to 4 + int(3*log(D)).
    sigma0: initial CMA sigma (in meters).
    per_comp_bound: bound per component (|dx|,|dy|,|dz|) per 0.25 s.
    l2_bound: maximum L2 step size per 0.25 s.
    max_iters: CMA generations.
    seed: RNG seed.
    Returns: (best_traj (n_steps,3), best_reward)
    """
    dim = n_steps * 3
    rng = np.random.RandomState(seed)

    # CMA-ES with box bounds on each dimension (per-component)
    lower = [-per_comp_bound] * dim
    upper = [ per_comp_bound] * dim

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

    best_x = None
    best_reward = -np.inf

    # convenience: one CMA "ask" batch may be larger than env.n_envs; we’ll evaluate in chunks
    batch_size = env.n_envs

    it = 0
    eval_count = 0
    t0_all = time.time()
    print(f"{'iter':>5} | {'pop':>4} | {'chunks':>6} | {'mean':>8} | {'std':>8} | {'min':>8} | {'max':>8} | {'best':>8} | {'sigma':>7} | {'t_iter(s)':>8} | {'t_total(s)':>9}")

    while not es.stop() and it < max_iters:
        t_iter = time.time()
        X = es.ask()  # list of candidate flat vectors
        pop = len(X)
        n_chunks = (pop + batch_size - 1) // batch_size

        # Evaluate in chunks of size env.n_envs
        all_rewards = []
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start+batch_size]  # list

            # reshape and project each candidate
            trajs = []
            for x in chunk:
                tr = reshape_to_traj(np.asarray(x, dtype=np.float32), n_steps)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)

            # evaluate this chunk in parallel across envs
            rewards = evaluate_batch(env, trajs)
            all_rewards.extend(rewards.tolist())

            # Optional: per-chunk timing
            print(f"  └─ chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time()-t_chunk:.3f}s")

        all_rewards = np.asarray(all_rewards, dtype=np.float32)
        eval_count += all_rewards.size

        # CMA-ES minimizes; we want to maximize reward -> use negative as loss
        losses = (-all_rewards).tolist()
        es.tell(X, losses)

        # Track best
        gen_best_idx = int(np.argmax(all_rewards))
        gen_best_reward = float(all_rewards[gen_best_idx])
        gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
        # Re-apply projection to record the actual executed policy
        gen_best_traj = project_deltas(
            reshape_to_traj(gen_best_x, n_steps),
            per_comp_bound, l2_bound
        )

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_x = gen_best_traj.copy()

        # Iteration summary
        m = float(all_rewards.mean()) if all_rewards.size else float('nan')
        s = float(all_rewards.std())  if all_rewards.size else float('nan')
        mn = float(all_rewards.min()) if all_rewards.size else float('nan')
        mx = float(all_rewards.max()) if all_rewards.size else float('nan')
        # CMA sigma (may be under es.sigma or es.sigma)
        try:
            sigma_now = float(es.sigma)
        except Exception:
            sigma_now = float(es.sigma0) if hasattr(es, 'sigma0') else float('nan')

        t_iter_sec = time.time() - t_iter
        t_total_sec = time.time() - t0_all

        print(f"{it:5d} | {pop:4d} | {n_chunks:6d} | {m:8.4f} | {s:8.4f} | {mn:8.4f} | {mx:8.4f} | {best_reward:8.4f} | {sigma_now:7.4f} | {t_iter_sec:8.3f} | {t_total_sec:9.3f}")

        it += 1

    return best_x, best_reward

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Build env with desired number of parallel envs (controls batch size)
    env = Train_Env_Wiring(task='wiring', log_dir="logs/wiring", n_envs=10)

    # Each delta runs for 0.25s (50 * 0.005s). n_steps=20 -> 5 seconds of control.
    n_steps = 10

    # Bounds:
    #  - per_comp_bound=0.03 => each component in [-3 cm, 3 cm] per 0.25s
    #  - l2_bound=0.04       => overall per-step movement <= 4 cm per 0.25s
    best_traj, best_reward = optimize_wiring_trajectory(
        env,
        popsize=100,
        n_steps=n_steps,
        sigma0=0.01,          # initial exploration ~1 cm per component
        per_comp_bound=0.03,  # 3 cm/component cap
        l2_bound=0.04,        # 4 cm per 0.25s cap
        max_iters=15,        # tweak as needed
        seed=123
    )

    # print("Best terminal reward:", best_reward)
    # Optionally, re-run to visualize the best trajectory:
    # env.eval_traj(best_traj[None, ...])  # shape (1, n_steps, 3); simulator will pad to n_envs
