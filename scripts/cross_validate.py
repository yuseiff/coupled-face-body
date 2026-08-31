"""
Cross-validation stability check for the mismatch discrimination result.

The single-split result (59.9% discrimination accuracy for A3, seed=42) was evaluated on only the
2 subjects held out by that one split -- we can't yet tell if that's a real, generalizable effect
or a quirk of those 2 people's motion style. This script repeats the FULL cycle (train A1/A2/A3,
then run the mismatch test) across several different random subject splits, so we can see whether
A3's discrimination accuracy is consistently above chance, or bounces around.

WARNING: this retrains 3 models per split, on CPU. With --epochs 10 (matching prior runs), expect
each split to take roughly as long as your earlier single-split runs combined (~3 training runs +
1 eval). Budget real time for this -- it's meant to run unattended, not be watched.

Usage:
    python scripts/cross_validate.py --converted_dir ./data/beat2/converted/beat_english_v2.0.0 \
        --n_splits 5 --epochs 10
"""
import argparse
import json
import os
import subprocess
import sys


def run_streaming(cmd: list[str]) -> str:
    """Runs a command, prints its output live (so a multi-hour run doesn't look frozen), and
    also returns the full captured stdout for parsing afterward."""
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return "".join(lines)


def extract_result_json(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError("No RESULT_JSON line found in eval_mismatch.py output -- did it crash silently?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--output_root", default="./outputs/cv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Explicit seed list; overrides --n_splits if given. "
                              "Include 42 to reuse your original split's models are NOT reused "
                              "(fresh training happens for every seed here, for a clean comparison).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for subprocesses")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.n_splits))
    print(f"Running cross-validation across {len(seeds)} seed(s): {seeds}")
    print(f"Each seed trains A1, A2, A3 for {args.epochs} epochs, then runs the mismatch test.\n")

    all_results = {}  # seed -> {ablation -> summary dict}

    for seed in seeds:
        split_output_dir = os.path.join(args.output_root, f"seed{seed}")
        checkpoint_dir = os.path.join(split_output_dir, "checkpoints")
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")

        for ablation in ["A1", "A2", "A3"]:
            cmd = [
                args.python, "scripts/train_prior.py",
                "--converted_dir", args.converted_dir,
                "--output_dir", split_output_dir,
                "--ablation", ablation,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--seed", str(seed),
                "--val_fraction", str(args.val_fraction),
            ]
            run_streaming(cmd)

        eval_cmd = [
            args.python, "scripts/eval_mismatch.py",
            "--converted_dir", args.converted_dir,
            "--checkpoint_dir", checkpoint_dir,
            "--seed", str(seed),
            "--val_fraction", str(args.val_fraction),
        ]
        eval_output = run_streaming(eval_cmd)
        all_results[seed] = extract_result_json(eval_output)

    # --- final cross-split summary ---
    print(f"\n\n{'='*70}\nCROSS-VALIDATION SUMMARY ({len(seeds)} splits)\n{'='*70}")
    print(f"{'Seed':<8}{'A2 disc. acc.':<16}{'A3 disc. acc.':<16}")
    a2_vals, a3_vals = [], []
    for seed in seeds:
        r = all_results[seed]
        a2 = r.get("A2", {}).get("discrimination_acc")
        a3 = r.get("A3", {}).get("discrimination_acc")
        if a2 is not None:
            a2_vals.append(a2)
        if a3 is not None:
            a3_vals.append(a3)
        a2_str = f"{a2*100:.1f}%" if a2 is not None else "N/A"
        a3_str = f"{a3*100:.1f}%" if a3 is not None else "N/A"
        print(f"{seed:<8}{a2_str:<16}{a3_str:<16}")

    def mean_std(vals):
        if not vals:
            return float("nan"), float("nan")
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        return m, var ** 0.5

    a2_mean, a2_std = mean_std(a2_vals)
    a3_mean, a3_std = mean_std(a3_vals)
    print("-" * 40)
    print(f"{'A2 mean±std':<8}{a2_mean*100:.1f}% ± {a2_std*100:.1f}%")
    print(f"{'A3 mean±std':<8}{a3_mean*100:.1f}% ± {a3_std*100:.1f}%")

    # save everything for the paper -- raw per-seed results plus the summary
    results_path = os.path.join(args.output_root, "cross_validation_results.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "per_seed": all_results,
            "summary": {"A2_mean": a2_mean, "A2_std": a2_std, "A3_mean": a3_mean, "A3_std": a3_std},
        }, f, indent=2)
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
