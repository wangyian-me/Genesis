import argparse
import sys
sys.path.append('.')
from gd import(
    Train_Env_GD,
    Train_GD_Coiling,
    Train_GD_Separation,
    Train_GD_Wire_Art,
    Train_GD_Wiring_Post,
    Train_GD_Wiring_Ring,
    Train_GD_Wrapping
)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--steps_interval', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_min', type=float, default=0.000001)
    parser.add_argument('--max_ddist', type=float, default=0.002)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--exp_base', type=float, default=1.1)
    parser.add_argument('--scale_method', type=str, default=None,
                        choices=[None, 'linear', 'exp', 'custom'])
    parser.add_argument('--show_gui', action='store_true')
    parser.add_argument('--vis_path', type=str, default=None)
    parser.add_argument('--task', type=str, default='wiring')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, 'cosine'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def construct_env(args) -> Train_Env_GD:
    exp_name = f"{args.exp_name}" if args.exp_name is not None else "gd"
    args.log_dir = f'logs/{args.task}/{exp_name}'

    if args.task == 'coiling':
        return Train_GD_Coiling(args)
    elif args.task == 'separation':
        return Train_GD_Separation(args)
    elif args.task == 'wireart':
        return Train_GD_Wire_Art(args)
    elif args.task == 'wiring_post':
        return Train_GD_Wiring_Post(args)
    elif args.task == 'wiring_ring':
        return Train_GD_Wiring_Ring(args)
    elif args.task == 'wrapping':
        return Train_GD_Wrapping(args)
    else:
        raise ValueError(f'Unknown task: {args.task}')


def main():
    args = arg_parser()

    trainer = construct_env(args)
    trainer.train()


if __name__ == "__main__":
    main()
