"""
select_config.py -- E2: choose the feature set, retraining epochs and dimension D on VALIDATION data,
for both the bipolar (integer-prototype) and the fully binary model. The chosen configuration is the
one to deploy on the STM32F303K8.

For every (feature set, epochs, D, seed) it trains exactly like train_hdc.py, then evaluates the same
trained model two ways:
  bipolar : +-1 queries, integer prototypes, cosine margin       (what train_hdc.py reports)
  binary  : queries and prototypes as bits, Hamming distance      (1 bit per prototype element)
Each has its own decision threshold, tuned on validation. Test metrics are recorded for the final table
but are NEVER used for selection.

Selection rule (fixed before looking at test results, the same idea as choosing the smallest model that
is not measurably worse):
  1. keep configurations whose model memory fits --budget_kib
  2. find the best mean validation F-beta among them
  3. keep every configuration within one seed-standard-deviation of that best
  4. pick the one with the smallest model memory (ties: higher validation F-beta)

Every run is saved as its own JSON, so an interrupted run resumes where it stopped.

Usage (next to train_hdc.py):
  python select_config.py                     # full grid: 4 feature sets x 2 epochs x 4 D x 3 seeds
  python select_config.py --report_only       # rebuild the tables from saved runs
  python select_config.py --budget_kib 40     # stricter memory budget for the selection step
"""
import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch

import data as D
import features as FT
import metrics as M
from encoders import build_encoder
from train_hdc import encode_all, fit_centroid, scores, tune_offset

# Candidate feature sets. Group names are the keys of features.FEATURE_GROUPS.
SETS = {
    "all30":        dict(drop=[], only=[]),                                   # 30 features
    "no_bandpower": dict(drop=["band_power"], only=[]),                       # 19 features
    "no_fft":       dict(drop=["band_power", "spectral_summary"], only=[]),   # 15 features, no FFT needed
    "beat_only":    dict(drop=[], only=["beat_interval"]),                    # 5 features
}
METRICS = ["accuracy", "sensitivity", "specificity", "fbeta_pooled", "fbeta_per_subject"]


def to_bits(bipolar):
    return (bipolar <= 0).to(torch.uint8)                       # +1 -> 0, -1 -> 1


def hamming_scores(q_bits, proto_bits):
    ham = torch.stack([(q_bits ^ p).sum(1) for p in proto_bits], dim=1).float()
    return (2.0 * (ham[:, 0] - ham[:, 1]) / q_bits.shape[1]).numpy()


def run_one(splits, feats, n_feat, set_name, epochs, dim, seed, a):
    torch.manual_seed(seed)                                     # same seeding order as train_hdc.py
    np.random.seed(seed)
    enc = build_encoder("feature", dim, n_features=n_feat, n_levels=a.levels)
    batch = max(32, int(512 * 2048 / dim))                      # keep RAM in check at large D
    t0 = time.time()
    H = {k: encode_all(enc, splits[k]["x"], feats[k], batch=batch) for k in ("train", "val", "test")}
    y = {k: torch.as_tensor(splits[k]["y_bin"], dtype=torch.long) for k in H}
    cls = fit_centroid(H["train"], y["train"], 2, epochs=epochs, lr=1.0, seed=seed)
    yv, yt = y["val"].numpy(), y["test"].numpy()
    sv, st = splits["val"]["subject"], splits["test"]["subject"]

    out = dict(set=set_name, n_features=n_feat, epochs=epochs, dim=dim, seed=seed,
               seconds=time.time() - t0)

    sA = {k: scores(cls, H[k]) for k in ("val", "test")}
    offA, valA = tune_offset(sA["val"], yv, a.beta)
    out["bipolar"] = dict(offset=offA, val_fbeta=valA,
                          test=M.primary_report(yt, (sA["test"] > offA).astype(int), st, beta=a.beta))

    W = cls.weight.data
    proto_bits = to_bits(torch.where(W > 0, 1.0, -1.0))
    sC = {k: hamming_scores(to_bits(H[k]), proto_bits) for k in ("val", "test")}
    offC, valC = tune_offset(sC["val"], yv, a.beta)
    out["binary"] = dict(offset=offC, val_fbeta=valC,
                         test=M.primary_report(yt, (sC["test"] > offC).astype(int), st, beta=a.beta))

    item_kib = enc.item_memory_bits() / 8 / 1024
    int_bits = int(np.ceil(np.log2(float(W.abs().max()) + 1))) + 1
    out["memory_kib"] = dict(item=item_kib, int_bits=int_bits,
                             bipolar=item_kib + 2 * dim * int_bits / 8 / 1024,
                             binary=item_kib + 2 * dim / 8 / 1024)
    return out


def summarise(runs, budget):
    rows = []
    keys = sorted({(r["set"], r["epochs"], r["dim"]) for r in runs},
                  key=lambda k: (list(SETS).index(k[0]) if k[0] in SETS else 99, k[1], k[2]))
    for s, ep, d in keys:
        rs = [r for r in runs if (r["set"], r["epochs"], r["dim"]) == (s, ep, d)]
        for model in ("bipolar", "binary"):
            val = np.array([r[model]["val_fbeta"] for r in rs], float)
            row = dict(set=s, n_features=rs[0]["n_features"], epochs=ep, dim=d, model=model,
                       seeds=len(rs), mem_kib=float(np.mean([r["memory_kib"][model] for r in rs])),
                       val_mean=val.mean(), val_std=val.std() if len(val) > 1 else 0.0)
            row["fits"] = row["mem_kib"] <= budget
            for m in METRICS:
                v = np.array([r[model]["test"][m] for r in rs], float)
                row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), (v.std() if len(v) > 1 else 0.0)
            rows.append(row)
    return rows


def select(rows):
    cand = [r for r in rows if r["fits"]]
    if not cand:
        return None, []
    best = max(cand, key=lambda r: r["val_mean"])
    tol = best["val_std"]
    near = [r for r in cand if r["val_mean"] >= best["val_mean"] - tol]
    chosen = min(near, key=lambda r: (r["mem_kib"], -r["val_mean"]))
    return chosen, near


def pm(r, k, scale=1.0, fmt="{:.3f}"):
    return (fmt + " ± " + fmt).format(r[k + "_mean"] * scale, r[k + "_std"] * scale)


def write_report(rows, outdir, budget):
    with open(os.path.join(outdir, "select_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = ["Feature set", "Feat.", "Epochs", "D", "Model", "Mem (KiB)", "Fits", "Val Fβ",
           "Test Fβ pooled", "Test Fβ subject", "Sens (%)", "Spec (%)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["set"], str(r["n_features"]), str(r["epochs"]), str(r["dim"]), r["model"],
            f"{r['mem_kib']:.1f}", "yes" if r["fits"] else "no",
            f"{r['val_mean']:.4f} ± {r['val_std']:.4f}",
            pm(r, "fbeta_pooled", fmt="{:.4f}"), pm(r, "fbeta_per_subject", fmt="{:.4f}"),
            pm(r, "sensitivity", 100, "{:.1f}"), pm(r, "specificity", 100, "{:.1f}")]) + " |")

    chosen, near = select(rows)
    lines += ["", f"## Selection (validation only, model memory <= {budget:g} KiB)", ""]
    if chosen is None:
        lines.append("No configuration fits the budget.")
    else:
        best = max((r for r in rows if r["fits"]), key=lambda r: r["val_mean"])
        lines.append(f"Best validation F-beta within budget: {best['set']}, {best['epochs']} epochs, "
                     f"D={best['dim']}, {best['model']}: {best['val_mean']:.4f} ± {best['val_std']:.4f}")
        lines.append(f"Configurations within one standard deviation of it: {len(near)}")
        lines.append("")
        lines.append(f"**Chosen (smallest of those): {chosen['set']} ({chosen['n_features']} features), "
                     f"{chosen['epochs']} epochs, D={chosen['dim']}, {chosen['model']}, "
                     f"{chosen['mem_kib']:.1f} KiB**")
        lines.append(f"- validation F-beta {chosen['val_mean']:.4f} ± {chosen['val_std']:.4f}")
        lines.append(f"- test F-beta pooled {pm(chosen, 'fbeta_pooled', fmt='{:.4f}')}, "
                     f"per subject {pm(chosen, 'fbeta_per_subject', fmt='{:.4f}')}")
        lines.append(f"- test sensitivity {pm(chosen, 'sensitivity', 100, '{:.1f}')}%, "
                     f"specificity {pm(chosen, 'specificity', 100, '{:.1f}')}%")
    md = "\n".join(lines)
    with open(os.path.join(outdir, "select_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default="./results_select")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--epochs", type=int, nargs="+", default=[5, 20])
    p.add_argument("--dims", type=int, nargs="+", default=[1024, 2048, 4096, 5000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--levels", type=int, default=64)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--budget_kib", type=float, default=48.0,
                   help="model-memory budget for selection; 48 leaves 16 KiB of the 64 KiB flash for code")
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if not a.report_only:
        splits = D.load_splits(a.cache)
        raw = FT.cached_features(splits)                        # all 30, extracted once
        todo = [(s, ep, d, sd) for s in a.sets for ep in a.epochs for d in a.dims for sd in a.seeds]
        prepared = {}
        for i, (s, ep, d, sd) in enumerate(todo, 1):
            path = os.path.join(a.outdir, f"{s}_e{ep}_d{d}_s{sd}.json")
            if os.path.exists(path):
                print(f"[{i}/{len(todo)}] {s} e{ep} D={d} seed {sd}: done, skipping")
                continue
            if s not in prepared:                               # columns + scaler fitted on train only
                keep = FT.select_columns(SETS[s]["drop"], SETS[s]["only"])
                sub = {k: v[:, keep] for k, v in raw.items()}
                scaler = FT.fit_scaler(sub["train"])
                prepared[s] = ({k: FT.apply_scaler(v, scaler) for k, v in sub.items()}, len(keep))
            feats, n_feat = prepared[s]
            r = run_one(splits, feats, n_feat, s, ep, d, sd, a)
            with open(path, "w") as f:
                json.dump(r, f, indent=2)
            print(f"[{i}/{len(todo)}] {s} ({n_feat} feat) e{ep} D={d} seed {sd}  "
                  f"val Fb bipolar {r['bipolar']['val_fbeta']:.4f} / binary {r['binary']['val_fbeta']:.4f}  "
                  f"mem {r['memory_kib']['bipolar']:.1f} / {r['memory_kib']['binary']:.1f} KiB  "
                  f"{r['seconds']:.0f}s")

    runs = []
    for fpath in sorted(glob.glob(os.path.join(a.outdir, "*_e*_d*_s*.json"))):
        with open(fpath) as f:
            runs.append(json.load(f))
    if not runs:
        print("no runs found in", a.outdir)
        return
    print()
    write_report(summarise(runs, a.budget_kib), a.outdir, a.budget_kib)


if __name__ == "__main__":
    main()
