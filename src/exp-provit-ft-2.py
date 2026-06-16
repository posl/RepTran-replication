"""Launcher: run PRoViTFT for reps 0..n_reps-1, each in its own subprocess
(exp-provit-ft-1.py). Running each rep as a fresh process isolates segfaults
(a transient CUDA/cuDNN crash kills only that rep) and fully releases GPU memory
between reps. A crashed rep is retried; because the runner saves per benchmark
and resumes, a retry continues where it left off. After all reps, the per-rep
JSONs are merged into a single combined results file.

Usage:
    python exp-provit-ft-2.py c100 0
    python exp-provit-ft-2.py tiny-imagenet 0 --n-reps 5
"""
import os, sys, json, glob, subprocess
import argparse
from utils.constant import ViTExperiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, choices=["c100", "tiny-imagenet"])
    parser.add_argument("k", type=int, help="fold id")
    parser.add_argument("--n-reps", type=int, default=5,
                        help="number of repetitions (REPTRAN protocol: 5)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="max restarts per rep on crash/non-zero exit (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    args = parser.parse_args()

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp-provit-ft-1.py")
    common = ["--lr", str(args.lr), "--gamma", str(args.gamma),
              "--batch-size", str(args.batch_size), "--time-limit", str(args.time_limit)]

    for reps_id in range(args.n_reps):
        cmd = [sys.executable, runner, args.ds, str(args.k), str(reps_id)] + common
        for attempt in range(1, args.max_retries + 1):
            print(f"\n>>> rep {reps_id} attempt {attempt}/{args.max_retries}: {' '.join(cmd)}",
                  flush=True)
            rc = subprocess.run(cmd).returncode
            if rc == 0:
                print(f">>> rep {reps_id} finished (rc=0).", flush=True)
                break
            print(f">>> rep {reps_id} exited rc={rc} "
                  f"(<0 = killed by signal, e.g. -11 SIGSEGV). Retrying with resume...",
                  flush=True)
        else:
            print(f"!!! rep {reps_id} did not finish after {args.max_retries} attempts.",
                  flush=True)

    # Merge per-rep JSONs into one combined results file.
    exp_obj = getattr(ViTExperiment, args.ds.replace("-", "_"))
    save_dir = os.path.join(exp_obj.OUTPUT_DIR.format(k=args.k), "provit_ft")
    pat = os.path.join(save_dir, f"results_lr{args.lr}_tl{int(args.time_limit)}_rep*.json")
    merged = []
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            merged.extend(json.load(f))
    out = os.path.join(save_dir, f"results_lr{args.lr}_tl{int(args.time_limit)}.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged {len(merged)} rows from {len(glob.glob(pat))} rep files -> {out}", flush=True)
