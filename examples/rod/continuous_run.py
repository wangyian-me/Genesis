#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Continuously rerun run_cmaes.py with --task.")
    ap.add_argument("--task", type=str, required=True, help="Task to pass to run_cmaes.py")
    ap.add_argument("--requires_grad", action="store_true",
                    help="If set, use CMAES with GD")
    ap.add_argument('--scale_method', type=str, default=None,
                    choices=[None, 'linear', 'exp', 'custom'])
    ap.add_argument('--ratio', type=float, default=0.1)
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait before restarting after exit (default: 2.0)")
    args = ap.parse_args()

    if args.requires_grad:
        cmd = ["python", "run_cmaes_gd.py", "--task", args.task]
        if args.scale_method is not None:
            cmd.extend(["--scale_method", str(args.scale_method)])
        cmd.extend(["--ratio", str(args.ratio)])
    else:
        cmd = ["python", "run_cmaes.py", "--task", args.task]

    print(f"[supervisor] Starting loop. Will run: {' '.join(cmd)}")
    try:
        i = 1
        while True:
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
