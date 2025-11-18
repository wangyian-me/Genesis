import torch
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import time
from pathlib import Path

from tqdm import trange

from mushroom_rl.core import VectorCore
from mushroom_rl.algorithms.actor_critic import PPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils import TorchUtils

import sys
sys.path.append('.')
from train_env import Train_Env
from train_env_coiling import Train_Env_Coiling
from train_env_gathering import Train_Env_Gathering
from train_env_lifting import Train_Env_Lifting
from train_env_separation import Train_Env_Separation
from train_env_wireart import Train_Env_Wireart
from train_env_wrapping import Train_Env_Wrapping

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features[0])
        self._h2 = nn.Linear(n_features[0], n_features[1])
        self._h3 = nn.Linear(n_features[1], n_features[2])
        self._h4 = nn.Linear(n_features[2], n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))

        # CRITICAL: Initialize final layer with very small weights for near-zero initial outputs
        # This prevents wild random actions aqt the start of training
        nn.init.uniform_(self._h4.weight, -0.001, 0.001)
        nn.init.constant_(self._h4.bias, 0.0)

        # Ensure parameters are float32 to avoid Float/Double mismatches
        self.float()

    def forward(self, state, **kwargs):
        x = torch.squeeze(state, 1)
        # Align dtype/device with layer weights to avoid mismatches
        x = x.to(dtype=self._h1.weight.dtype, device=self._h1.weight.device)
        features1 = F.relu(self._h1(x))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        a = self._h4(features3)

        return a

def experiment(alg, n_envs, n_epochs, n_outer_steps, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, policy_params, critic_params,
               task="wiring_ring", exp_name="PPO",
               scene_version=1, pos_bound=0.1, angle_bound=5.0,
               args=None):

    # n_outer_steps is the HORIZON (steps per episode)

    env_dict = {
        "coiling": Train_Env_Coiling,
        "gathering": Train_Env_Gathering,
        "lifting": Train_Env_Lifting,
        "separation": Train_Env_Separation,
        "wireart": Train_Env_Wireart,
        "wrapping": Train_Env_Wrapping,
    }
    mdp: Train_Env = env_dict[task](
        task=task,
        log_dir=os.path.join("logs", task, exp_name),
        n_envs=n_envs,
        GUI=args.gui,
        camera=False,
        scene_version=scene_version,
    )

    if task == "coiling":
        mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound, n_rigid_obs=1, debug=args.gui)
    elif task == "gathering":
        mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound, n_rigid_obs=3, debug=args.gui)
    elif task == "lifting":
        mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound, n_rigid_obs=2, debug=args.gui)
    elif task == "separation":
        mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound, n_rigid_obs=mdp.rope2.n_vertices, debug=args.gui)
    elif task == "wireart":
        mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound, n_rigid_obs=0, debug=args.gui)
    elif task == "wrapping":
        mdp.init_rl_env(n_steps=n_outer_steps, pos_bound=pos_bound, angle_bound=angle_bound, n_rigid_obs=1, debug=args.gui)
    else:
        raise ValueError(f"Unknown env_name: {task}")

    print(f'Max moving distance {mdp._l2_limit}x{n_outer_steps}={mdp._l2_limit * n_outer_steps} m for each control point')

    # Prepare curve logging file: logs/<task>/<EXP_ID>
    def _get_min_unused_exp_id(directory: Path) -> int:
        existing_ids = set()
        if directory.exists():
            for child in directory.iterdir():
                if child.is_file() and child.suffix == ".csv" and child.stem.isdigit():
                    existing_ids.add(int(child.stem))
        exp_id = 0
        while exp_id in existing_ids:
            exp_id += 1
        return exp_id

    curve_dir = Path("logs") / task / exp_name
    curve_dir.mkdir(parents=True, exist_ok=True)
    exp_id = _get_min_unused_exp_id(curve_dir)
    curve_path = curve_dir / f"{exp_id}.csv"
    curve_file = open(curve_path, "w")
    curve_file.write(f"epoch,R,F,best_so_far,epoch_duration\n")

    full_log_path = curve_dir / f"{exp_id}_full.csv"
    full_log_file = open(full_log_path, "w")
    full_log_file.write(f"epoch,idx,R,F,last_idx\n")

    ckpt_dir = curve_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    args_path = curve_dir / f"{exp_id}_args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
    diagnostic_log_path = curve_dir / f"{exp_id}_diag.txt"
    diagnostic_log_file = open(diagnostic_log_path, "w")

    best_so_far = -np.inf

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    critic_params.update(input_shape=mdp.info.observation_space.shape)
    alg_params['critic_params'] = critic_params

    agent = alg(mdp.info, policy, **alg_params)

    core = VectorCore(agent, mdp)

    print("Starting training")
    for it in trange(n_epochs, leave=False):
        epoch_start = time.time()
        batch_R = list()
        batch_F = list()

        # for i in range(batch_size):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

        # CRITICAL FIX: Manually constrain std to prevent explosion
        # Allow std in range [0.01, 0.2] - enough exploration but prevent collapse
        with torch.no_grad():
            agent.policy._log_sigma.clamp_(torch.log(torch.tensor(0.01)), torch.log(torch.tensor(0.2)))
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, record=False)

        ur = dataset.undiscounted_return
        fsr = dataset.reward_sequence
        episode_length = dataset.episodes_length
        last_idx = episode_length - 1
        action = dataset.action

        # Diagnostic: Print action and policy statistics
        policy_std = torch.exp(agent.policy._log_sigma).cpu().detach()
        diagnostic_log_file.write(f"\n\n{'='*60}\n")
        diagnostic_log_file.write(f"DIAGNOSTICS (epoch {it})\n")
        diagnostic_log_file.write(f"{'='*60}\n")
        diagnostic_log_file.write(f"Policy std: min={policy_std.min():.4f}, max={policy_std.max():.4f}, mean={policy_std.mean():.4f}\n")
        diagnostic_log_file.write(f"Episode lengths: min={episode_length.min().item()}, max={episode_length.max().item()}, mean={episode_length.float().mean().item():.1f}\n")
        diagnostic_log_file.write(f"Action stats - Pos: mean={action[:, :3].mean():.4f}, std={action[:, :3].std():.4f}\n")
        diagnostic_log_file.write(f"Action stats - Rot: mean={action[:, 3:].mean():.4f}, std={action[:, 3:].std():.4f}\n")
        success_rate = (episode_length == 10).float().mean().item() * 100
        diagnostic_log_file.write(f"Success rate: {success_rate:.1f}% ({(episode_length == 10).sum().item()}/{len(episode_length)} episodes)\n")
        diagnostic_log_file.write(f"Best final reward: {fsr[:, last_idx].max():.2f}\n")
        diagnostic_log_file.write(f"{'='*60}\n\n")
        diagnostic_log_file.flush()
        os.fsync(diagnostic_log_file.fileno())

        n_data = len(ur)
        for j in range(n_data):
            actual_last_idx = episode_length[j] - 1
            last_reward = fsr[j, actual_last_idx]
            batch_R.append(ur[j])
            batch_F.append(last_reward)

            full_log_file.write(f"{it},{j},{ur[j]},{last_reward},{actual_last_idx}\n")
            full_log_file.flush()
            os.fsync(full_log_file.fileno())

        del dataset

        # (n_envs * batch_size, )
        batch_R = torch.as_tensor(batch_R)
        batch_R = batch_R.cpu().numpy()
        batch_F = torch.as_tensor(batch_F)
        batch_F = batch_F.cpu().numpy()

        # print(f"batch_R: {batch_R.shape}, batch_F: {batch_F.shape}")

        Return = np.max(batch_R)
        FinalReward = np.max(batch_F)
        agent.save(path=ckpt_dir / f"{it}_ppo.pkl")
        if FinalReward > best_so_far:
            agent.save(path=curve_dir / "best_ppo.pkl", full_save=True)
            best_so_far = FinalReward

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start

        # Log reward for this iteration to curve file
        curve_file.write(f"{it},{Return},{FinalReward},{best_so_far},{epoch_duration}\n")
        curve_file.flush()
        os.fsync(curve_file.fileno())

    # Close curve file after training
    curve_file.close()
    full_log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wiring_ring', help='Task name')
    parser.add_argument('--exp_name', type=str, default='PPO', help='Experiment name')
    parser.add_argument('--n_traj', type=int, default=20, help='Number of trajectories per environment')
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--bound', type=float, default=0.1)
    parser.add_argument('--angle_bound', type=float, default=5.0)
    parser.add_argument('--scene_version', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Enforce float32 globally for new tensors/modules
    torch.set_default_dtype(torch.float32)
    TorchUtils.set_default_device('cuda:0')
    n_envs = 10
    # ppo_params = dict(
    #     actor_optimizer={
    #         'class': optim.Adam,
    #         'params': {'lr': 1e-3}
    #     },
    #     n_epochs_policy=5,
    #     batch_size=int((n_envs*24) / 16),
    #     eps_ppo=.2,
    #     lam=.95,
    #     ent_coeff=0.01
    # )
    ppo_params = dict(
        actor_optimizer={
            'class': optim.Adam,
            'params': {'lr': 5e-4}  # Increased: Learn faster from rare successes
        },
        n_epochs_policy=5,  # Slightly more updates
        batch_size=25,  # Smaller: More frequent updates with limited data
        eps_ppo=.2,  # Standard clipping
        lam=.95,
        ent_coeff=0.01  # Small entropy bonus for exploration
    )
    policy_params = dict(
        std_0=0.05,  # Moderate initial std - allows exploration
        n_features=[256, 128, 64],
        use_cuda=True
    )
    critic_params = dict(
        network=Network,
        optimizer={
            'class': optim.Adam,
            'params': {'lr': 5e-4}  # Match actor LR
        },
        loss=F.mse_loss,
        n_features=[256, 128, 64],
        batch_size=25,
        use_cuda=True,
        output_shape=(1,)
    )

    # Setup for: 10 envs, 10 steps/trajectory, 20 trajectories before policy update
    n_trajectories = args.n_traj  # Number of trajectories to collect per env

    experiment(
        alg=PPO,
        n_envs=n_envs,
        n_epochs=20,
        n_outer_steps=args.n_steps,
        n_steps=n_envs * args.n_steps * n_trajectories,
        n_steps_per_fit=args.n_steps * n_trajectories,
        n_episodes_test=n_envs * 3,  # Evaluate with 30 episodes (3 per env)
        alg_params=ppo_params,
        policy_params=policy_params,
        critic_params=critic_params,
        task=args.task,
        exp_name=args.exp_name,
        scene_version=args.scene_version,
        pos_bound=args.bound,
        angle_bound=args.angle_bound,
        args=args
    )
