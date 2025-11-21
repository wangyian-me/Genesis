import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse the training log file and extract metrics.

    If an iteration appears multiple times (due to process restarts),
    the latest values will be used.

    Returns:
        dict: Dictionary containing lists of metrics indexed by iteration
    """
    # Use a dictionary to track metrics by iteration number
    # This allows us to overwrite with the latest values if an iteration is repeated
    iteration_data = {}

    with open(log_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for iteration marker
        iteration_match = re.search(r'Iteration (\d+):', line)
        if iteration_match:
            iteration = int(iteration_match.group(1))

            # Look ahead to parse the metrics on the next lines
            if i + 2 < len(lines):
                # Parse rewards and vf_loss from the next line
                metrics_line1 = lines[i + 1].strip()
                rewards_match = re.search(r'rewards\s+([-+]?[0-9]*\.?[0-9]+)', metrics_line1)
                vf_loss_match = re.search(r'vf_loss\s+\[([-+]?[0-9]*\.?[0-9]+)\]', metrics_line1)

                # Parse entropy, kl, std from the line after
                metrics_line2 = lines[i + 2].strip()
                entropy_match = re.search(r'entropy\s+([-+]?[0-9]*\.?[0-9]+)', metrics_line2)
                kl_match = re.search(r'kl\s+([-+]?[0-9]*\.?[0-9]+)', metrics_line2)
                std_match = re.search(r'std\s+([-+]?[0-9]*\.?[0-9]+)', metrics_line2)

                # Only add/update if all metrics are found
                if all([rewards_match, vf_loss_match, entropy_match, kl_match, std_match]):
                    # Store/overwrite data for this iteration (latest values will be kept)
                    iteration_data[iteration] = {
                        'rewards': float(rewards_match.group(1)),
                        'vf_loss': float(vf_loss_match.group(1)),
                        'entropy': float(entropy_match.group(1)),
                        'kl': float(kl_match.group(1)),
                        'std': float(std_match.group(1))
                    }

        i += 1

    # Convert to sorted arrays
    sorted_iterations = sorted(iteration_data.keys())

    data = {
        'iteration': np.array(sorted_iterations),
        'rewards': np.array([iteration_data[i]['rewards'] for i in sorted_iterations]),
        'vf_loss': np.array([iteration_data[i]['vf_loss'] for i in sorted_iterations]),
        'entropy': np.array([iteration_data[i]['entropy'] for i in sorted_iterations]),
        'kl': np.array([iteration_data[i]['kl'] for i in sorted_iterations]),
        'std': np.array([iteration_data[i]['std'] for i in sorted_iterations])
    }

    return data


def plot_metrics(data, plot_rewards=True, plot_vf_loss=True, plot_entropy=True,
                 plot_kl=True, plot_std=True, save_path=None):
    """
    Plot selected metrics vs iteration number.

    Args:
        data: Dictionary containing parsed metrics
        plot_rewards: Whether to plot rewards
        plot_vf_loss: Whether to plot vf_loss
        plot_entropy: Whether to plot entropy
        plot_kl: Whether to plot kl divergence
        plot_std: Whether to plot std
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    # Count how many subplots we need
    subplots_needed = sum([plot_rewards, plot_vf_loss, plot_entropy, plot_kl, plot_std])

    if subplots_needed == 0:
        print("No metrics selected to plot!")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(subplots_needed, 1, figsize=(10, 3 * subplots_needed))

    # If only one subplot, axes is not a list
    if subplots_needed == 1:
        axes = [axes]

    subplot_idx = 0
    iterations = data['iteration']

    # Plot rewards
    if plot_rewards:
        axes[subplot_idx].plot(iterations, data['rewards'], 'b-', linewidth=2, label='Rewards')
        axes[subplot_idx].set_xlabel('Iteration')
        axes[subplot_idx].set_ylabel('Rewards')
        axes[subplot_idx].set_title('Rewards vs Iteration')
        axes[subplot_idx].grid(True, alpha=0.3)
        axes[subplot_idx].legend()
        subplot_idx += 1

    # Plot vf_loss
    if plot_vf_loss:
        axes[subplot_idx].plot(iterations, data['vf_loss'], 'r-', linewidth=2, label='VF Loss')
        axes[subplot_idx].set_xlabel('Iteration')
        axes[subplot_idx].set_ylabel('Value Function Loss')
        axes[subplot_idx].set_title('Value Function Loss vs Iteration')
        axes[subplot_idx].grid(True, alpha=0.3)
        axes[subplot_idx].legend()
        subplot_idx += 1

    # Plot entropy
    if plot_entropy:
        axes[subplot_idx].plot(iterations, data['entropy'], 'g-', linewidth=2, label='Entropy')
        axes[subplot_idx].set_xlabel('Iteration')
        axes[subplot_idx].set_ylabel('Entropy')
        axes[subplot_idx].set_title('Entropy vs Iteration')
        axes[subplot_idx].grid(True, alpha=0.3)
        axes[subplot_idx].legend()
        subplot_idx += 1

    # Plot kl
    if plot_kl:
        axes[subplot_idx].plot(iterations, data['kl'], 'm-', linewidth=2, label='KL Divergence')
        axes[subplot_idx].set_xlabel('Iteration')
        axes[subplot_idx].set_ylabel('KL Divergence')
        axes[subplot_idx].set_title('KL Divergence vs Iteration')
        axes[subplot_idx].grid(True, alpha=0.3)
        axes[subplot_idx].legend()
        subplot_idx += 1

    # Plot std
    if plot_std:
        axes[subplot_idx].plot(iterations, data['std'], 'c-', linewidth=2, label='Std')
        axes[subplot_idx].set_xlabel('Iteration')
        axes[subplot_idx].set_ylabel('Standard Deviation')
        axes[subplot_idx].set_title('Standard Deviation vs Iteration')
        axes[subplot_idx].grid(True, alpha=0.3)
        axes[subplot_idx].legend()
        subplot_idx += 1

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {save_path}")

    return fig


def print_statistics(data):
    """Print summary statistics for each metric."""
    print("\n" + "="*60)
    print("Training Statistics Summary")
    print("="*60)

    metrics = ['rewards', 'vf_loss', 'entropy', 'kl', 'std']

    for metric in metrics:
        values = data[metric]
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {np.mean(values):.4f}")
        print(f"  Std:    {np.std(values):.4f}")
        print(f"  Min:    {np.min(values):.4f} (iter {data['iteration'][np.argmin(values)]})")
        print(f"  Max:    {np.max(values):.4f} (iter {data['iteration'][np.argmax(values)]})")
        print(f"  Final:  {values[-1]:.4f}")

    print("\n" + "="*60)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Analyze and plot PPO training logs')

    parser.add_argument('log_dir', type=str,
                        help='Path to the training log directory')
    parser.add_argument('--no-rewards', action='store_true',
                        help='Do not plot rewards')
    parser.add_argument('--no-vf-loss', action='store_true',
                        help='Do not plot value function loss')
    parser.add_argument('--no-entropy', action='store_true',
                        help='Do not plot entropy')
    parser.add_argument('--no-kl', action='store_true',
                        help='Do not plot KL divergence')
    parser.add_argument('--no-std', action='store_true',
                        help='Do not plot standard deviation')
    parser.add_argument('--stats', action='store_true',
                        help='Print summary statistics')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_path = log_dir / 'train.log'

    # Parse the log file
    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)
    print(f"Found {len(data['iteration'])} iterations")

    # Print statistics if requested
    if args.stats:
        print_statistics(data)

    save_path = log_dir / 'training_metrics.png'

    # Plot the metrics
    plot_metrics(
        data,
        plot_rewards=not args.no_rewards,
        plot_vf_loss=not args.no_vf_loss,
        plot_entropy=not args.no_entropy,
        plot_kl=not args.no_kl,
        plot_std=not args.no_std,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
