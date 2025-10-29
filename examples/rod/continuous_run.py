#!/usr/bin/env python3
import argparse
import subprocess
import os
import time
import json
from datetime import datetime


def check_completion(args):
    meta_path = f"checkpoints/trial_{args.task}/{args.exp_name}/resume_meta.json"
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    iteration = int(meta.get('iter', 0)) + 1
    return iteration >= args.max_iter

def main():
    ap = argparse.ArgumentParser(description="Continuously rerun run_cmaes.py with --task.")
    ap.add_argument("--task", type=str, required=True, help="Task to pass to run_cmaes.py")
    ap.add_argument("--seed", type=int, default=123, help="Random seed to use for each run (default: 123)")
    ap.add_argument("--requires_grad", action="store_true",
                    help="If set, use CMAES with GD")
    ap.add_argument('--scale_method', type=str, default=None,
                    choices=[None, 'linear', 'exp', 'custom'])
    ap.add_argument('--version', type=int, default=1)
    ap.add_argument('--ratio', type=float, nargs='+', default=[0.1])
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait before restarting after exit (default: 2.0)")
    ap.add_argument('--exp_name', type=str, required=True)
    ap.add_argument('--n_steps', type=int, default=10)
    ap.add_argument('--max_iter', type=int, default=20)
    args = ap.parse_args()

    if not args.requires_grad and "gd" in args.exp_name:
        print("[supervisor] Error: --exp_name indicates `gd` but --requires_grad not set.")
        exit(0)
    elif args.requires_grad and "gd" not in args.exp_name:
        print("[supervisor] Error: --exp_name does not indicate `gd` but --requires_grad is set.")
        exit(0)

    if args.requires_grad:
        cmd = ["python", "run_cmaes_gd.py", "--task", args.task, "--seed", str(args.seed)]
        if args.scale_method is not None:
            cmd.extend(["--scale_method", str(args.scale_method)])
        cmd.extend(["--version", str(args.version)])
        cmd.extend(["--ratio"] + [str(r) for r in args.ratio])
        if args.exp_name is not None:
            cmd.extend(["--exp_name", str(args.exp_name)])
        cmd.extend(["--n_steps", str(args.n_steps)])
    else:
        cmd = ["python", "run_cmaes.py", "--task", args.task, "--seed", str(args.seed)]
        if args.exp_name is not None:
            cmd.extend(["--exp_name", str(args.exp_name)])
        cmd.extend(["--n_steps", str(args.n_steps)])

    print(f"[supervisor] Starting loop. Will run: {' '.join(cmd)}")
    try:
        i = 1
        while True:
            if check_completion(args):
                print(f"[supervisor] Detected completion for task '{args.task}' "
                      f"with exp_name '{args.exp_name}', max_iter '{args.max_iter}'. Exiting.")
                break
            print(f"\n[supervisor] Launch #{i} at {datetime.now().isoformat(timespec='seconds')}")
            # Run the child; do not raise on non-zero (we want to restart regardless)
            result = subprocess.run(cmd)
            print(f"[supervisor] Child exited with return code {result.returncode}")
            i += 1
            if args.delay > 0:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\n[supervisor] Stopped by user. Bye!")

if __name__ == "__main__":
    main()
