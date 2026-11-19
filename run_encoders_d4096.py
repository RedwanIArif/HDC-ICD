"""
run_encoders_d4096.py -- the encoder comparison of Table III, repeated at D = 4096.

Uses the same train_hdc.py as the D = 2048 table, so encoder, training (20 retraining
epochs), integer prototypes, threshold tuning on validation and the patient-wise split are
unchanged. Only the dimension differs.

Put this file next to train_hdc.py, encoders.py, features.py, data.py and metrics.py, and run
it from that folder, so that it finds cache_iegm.npz and features_cache.npz:

    python run_encoders_d4096.py                  # all four encoders, seeds 0, 1, 2
    python run_encoders_d4096.py --seeds 1        # seed 0 only, if time is short
    python run_encoders_d4096.py --report_only    # table from the runs finished so far

The loop runs seed 0 of every encoder first, then seed 1, then seed 2, so a complete table
exists after the first round. Finished runs are skipped, so the script can be stopped and
restarted at any time. The table is printed and written to results_enc_d4096/TABLE_D4096.md.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

PY = sys.executable
# feature encoders first (seconds), raw-sample encoders last (minutes)
ENCODERS = ["feature", "sinusoid", "ngram", "record"]
NAMES = {"record": "Time record", "ngram": "$n$-gram ($n=4$)",
         "sinusoid": "Sinusoid projection", "feature": "\\textbf{Key-level (HDC-ICD)}"}
INPUT = {"record": "samples", "ngram": "samples", "sinusoid": "features", "feature": "features"}
PROTO_BITS = 15          # integer prototypes, as in Table III


def tag_of(enc, dim):
    return f"enc_{enc}_d{dim}"


def run(enc, dim, seed, outdir, cache):
    tag = tag_of(enc, dim)
    path = os.path.join(outdir, f"{tag}_s{seed}.json")
    if os.path.exists(path):
        print(f"  skip {tag} seed {seed} (done)")
        return 0.0
    cmd = [PY, "train_hdc.py", "--tag", tag, "--seed", str(seed), "--outdir", outdir,
           "--cache", cache, "--encoder", enc, "--dim", str(dim), "--epochs", "20"]
    print("  " + " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"  -> {tag} seed {seed} took {dt/60:.1f} min", flush=True)
    return dt


def load(outdir, enc, dim):
    runs = []
    for f in sorted(glob.glob(os.path.join(outdir, f"{tag_of(enc, dim)}_s*.json"))):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def kib(run, dim):
    """Item memory (1 bit per element) plus two 15-bit prototypes, as in Table III."""
    item = run["size"]["item_memory_bits"]
    return (item + 2 * dim * PROTO_BITS) / 8 / 1024


def report(outdir, dim):
    rows = []
    for enc in ["record", "ngram", "sinusoid", "feature"]:
        r = load(outdir, enc, dim)
        if not r:
            continue
        fb = np.array([x["test"]["fbeta_pooled"] for x in r])
        se = np.array([x["test"]["sensitivity"] for x in r]) * 100
        sp = np.array([x["test"]["specificity"] for x in r]) * 100
        va = np.array([x["val"]["fbeta_pooled"] for x in r])
        rows.append(dict(enc=enc, n=len(r), kib=kib(r[0], dim), fb=fb, se=se, sp=sp, va=va))
    if not rows:
        print("no finished runs in", outdir)
        return

    def pm(v, f):
        return f.format(v.mean()) + ("" if len(v) == 1 else (" ± " + f).format(v.std()))

    md = [f"## Encoder comparison at D = {dim} (integer prototypes, 20 epochs, all 30 features)", "",
          "| Encoder | Input | Seeds | KiB | Val Fβ | Test Fβ | Sens (%) | Spec (%) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['enc']} | {INPUT[r['enc']]} | {r['n']} | {r['kib']:.1f} | "
                  f"{pm(r['va'], '{:.3f}')} | {pm(r['fb'], '{:.3f}')} | "
                  f"{pm(r['se'], '{:.1f}')} | {pm(r['sp'], '{:.1f}')} |")
    md += ["", "LaTeX rows for Table III (means over the finished seeds):", "", "```latex"]
    best = max(rows, key=lambda r: r["fb"].mean())["enc"]
    for r in rows:
        fb = f"{r['fb'].mean():.3f}"
        if r["enc"] == best:
            fb = f"\\textbf{{{fb}}}"
        md.append(f"    {NAMES[r['enc']]} & {INPUT[r['enc']]} & {r['kib']:.1f} & {fb} & "
                  f"{r['se'].mean():.1f} & {r['sp'].mean():.1f} \\\\")
    md += ["```", ""]
    if any(r["n"] < 3 for r in rows):
        md.append("Note: some encoders have fewer than 3 seeds. Say so in the caption, "
                  "or run the remaining seeds.")
    text = "\n".join(md)
    print("\n" + text)
    with open(os.path.join(outdir, f"TABLE_D{dim}.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--encoders", nargs="*", default=ENCODERS, choices=ENCODERS)
    p.add_argument("--outdir", default="./results_enc_d4096")
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()

    if not a.report_only:
        if not os.path.exists("train_hdc.py"):
            sys.exit("run this script from the folder that contains train_hdc.py")
        if not os.path.exists(a.cache):
            sys.exit(f"{a.cache} not found; run it from the folder used for the D = 2048 runs")
        os.makedirs(a.outdir, exist_ok=True)
        t_start = time.time()
        for seed in range(a.seeds):
            print(f"\n=== seed {seed} ===", flush=True)
            for enc in a.encoders:
                run(enc, a.dim, seed, a.outdir, a.cache)
            print(f"seed {seed} finished, {(time.time() - t_start)/60:.1f} min so far", flush=True)
            report(a.outdir, a.dim)
    report(a.outdir, a.dim)


if __name__ == "__main__":
    main()
