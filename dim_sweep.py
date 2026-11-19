"""
dim_sweep.py -- ablation over the hypervector dimension D.

For every (D, seed) it trains exactly as train_hdc.py does and evaluates the SAME trained model two ways:
  * bipolar     : +/-1 queries, integer class prototypes, cosine margin   (what train_hdc.py reports)
  * binary      : queries AND prototypes as bits, Hamming distance        (the 1-bit/element model)
The decision offset is tuned on VALIDATION separately for each, then the test set is read once.

Every run is saved as its own JSON, so an interrupted sweep resumes where it stopped.
At the end it writes a summary table (CSV + Markdown) and a figure.

Usage (put next to train_hdc.py):
  python dim_sweep.py --encoder feature --dims 512 1024 2048 4096 5000 8192 10000 --seeds 0 1 2
  python dim_sweep.py --encoder feature --report_only        # rebuild table + figure from saved runs
"""
import argparse
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

FLASH_KIB = 64            # STM32F303K8 flash
METRICS = ["accuracy", "sensitivity", "specificity", "fbeta_pooled", "fbeta_per_subject"]


# ----------------------------------------------------------------------------- binary helpers
def to_bits(bipolar):
    """+1 -> 0, -1 -> 1   (XOR of bits == product of +/-1)."""
    return (bipolar <= 0).to(torch.uint8)


def hamming_scores(q_bits, proto_bits):
    """Margin toward VA from Hamming distances, on the same scale as the cosine margin."""
    ham = torch.stack([(q_bits ^ p).sum(1) for p in proto_bits], dim=1).float()
    return (2.0 * (ham[:, 0] - ham[:, 1]) / q_bits.shape[1]).numpy()


# ----------------------------------------------------------------------------- one (D, seed) run
def run_one(splits, feats, n_feat, a, dim, seed):
    torch.manual_seed(seed)                      # same seeding order as train_hdc.py
    np.random.seed(seed)
    enc = build_encoder(a.encoder, dim, n_features=n_feat, n_levels=a.levels,
                        ngram=a.ngram, decimate=a.decimate)
    # The feature encoder materialises (batch, 30, D) floats twice; shrink the batch as D grows
    # so D=10000 does not need gigabytes of RAM. Raw-signal encoders size their own batch.
    batch = max(32, int(512 * 2048 / dim)) if enc.needs_features else None

    t0 = time.time()
    H = {k: encode_all(enc, splits[k]["x"], feats.get(k), batch=batch) for k in ("train", "val", "test")}
    t_enc = time.time() - t0
    y = {k: torch.as_tensor(splits[k]["y_bin"], dtype=torch.long) for k in H}

    t0 = time.time()
    cls = fit_centroid(H["train"], y["train"], 2, epochs=a.epochs, lr=a.lr, seed=seed)
    t_fit = time.time() - t0

    yv, yt = y["val"].numpy(), y["test"].numpy()
    sv, st = splits["val"]["subject"], splits["test"]["subject"]
    out = dict(dim=dim, seed=seed, encoder=a.encoder, epochs=a.epochs, levels=a.levels,
               encode_seconds=t_enc, fit_seconds=t_fit)

    # bipolar (= train_hdc.py)
    sA = {k: scores(cls, H[k]) for k in ("val", "test")}
    offA, valA = tune_offset(sA["val"], yv, a.beta)
    out["bipolar"] = dict(offset=offA, val_fbeta=valA,
                          test=M.primary_report(yt, (sA["test"] > offA).astype(int), st, beta=a.beta))

    # fully binary
    W = cls.weight.data
    proto_bits = to_bits(torch.where(W > 0, 1.0, -1.0))
    sC = {k: hamming_scores(to_bits(H[k]), proto_bits) for k in ("val", "test")}
    offC, valC = tune_offset(sC["val"], yv, a.beta)
    out["binary"] = dict(offset=offC, val_fbeta=valC,
                         test=M.primary_report(yt, (sC["test"] > offC).astype(int), st, beta=a.beta))

    # memory
    item_kib = enc.item_memory_bits() / 8 / 1024
    int_bits = int(np.ceil(np.log2(float(W.abs().max()) + 1))) + 1     # signed integer width
    out["memory_kib"] = dict(item=item_kib,
                             class_integer=2 * dim * int_bits / 8 / 1024,
                             class_binary=2 * dim / 8 / 1024,
                             integer_bits_per_element=int_bits)
    out["memory_kib"]["total_bipolar"] = item_kib + out["memory_kib"]["class_integer"]
    out["memory_kib"]["total_binary"] = item_kib + out["memory_kib"]["class_binary"]
    return out


# ----------------------------------------------------------------------------- summary
def load_runs(outdir):
    runs = []
    for f in sorted(glob.glob(os.path.join(outdir, "dim*_s*.json"))):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def summarise(runs):
    rows = []
    for dim in sorted({r["dim"] for r in runs}):
        rs = [r for r in runs if r["dim"] == dim]
        row = dict(dim=dim, n_seeds=len(rs),
                   mem_bipolar=np.mean([r["memory_kib"]["total_bipolar"] for r in rs]),
                   mem_binary=np.mean([r["memory_kib"]["total_binary"] for r in rs]),
                   int_bits=max(r["memory_kib"]["integer_bits_per_element"] for r in rs),
                   enc_s=np.mean([r["encode_seconds"] for r in rs]))
        for model in ("bipolar", "binary"):
            for m in METRICS:
                v = np.array([r[model]["test"][m] for r in rs], float)
                row[f"{model}_{m}_mean"] = v.mean()
                row[f"{model}_{m}_std"] = v.std() if len(v) > 1 else 0.0
            v = np.array([r[model]["val_fbeta"] for r in rs], float)
            row[f"{model}_valfb_mean"] = v.mean()
        rows.append(row)
    return rows


def pm(row, key, scale=1.0, fmt="{:.2f}"):
    m, s = row[key + "_mean"] * scale, row[key + "_std"] * scale
    return (fmt + " ± " + fmt).format(m, s) if row["n_seeds"] > 1 else fmt.format(m)


def write_tables(rows, outdir):
    import csv
    with open(os.path.join(outdir, "dim_sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = ["D", "seeds", "Model", "Mem (KiB)", "Fits 64 KiB", "Acc (%)", "Sens (%)", "Spec (%)",
           "Fb pooled", "Fb subject", "Fb val"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        for model, mem in (("bipolar", r["mem_bipolar"]), ("binary", r["mem_binary"])):
            lines.append("| " + " | ".join([
                str(r["dim"]), str(r["n_seeds"]),
                model + (f" ({r['int_bits']}-bit proto)" if model == "bipolar" else " (1-bit)"),
                f"{mem:.1f}", "yes" if mem <= FLASH_KIB else "no",
                pm(r, f"{model}_accuracy", 100), pm(r, f"{model}_sensitivity", 100),
                pm(r, f"{model}_specificity", 100),
                pm(r, f"{model}_fbeta_pooled", 1, "{:.4f}"), pm(r, f"{model}_fbeta_per_subject", 1, "{:.4f}"),
                f"{r[model + '_valfb_mean']:.4f}"]) + " |")
    md = "\n".join(lines)
    with open(os.path.join(outdir, "dim_sweep_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)


def figure(rows, outdir, encoder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"bipolar": "#2a78d6", "binary": "#eb6834"}          # categorical slots 1 and 2
    STYLE = {"bipolar": dict(marker="o", ls="-"), "binary": dict(marker="s", ls="--")}
    LABEL = {"bipolar": "bipolar (integer prototypes)", "binary": "fully binary (1-bit)"}
    dims = np.array([r["dim"] for r in rows])
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9a9a9a", "axes.grid": True, "grid.color": "#e6e6e6",
                         "grid.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("fbeta_pooled", 1, r"Test $F_\beta$ ($\beta$=2), pooled"),
              ("accuracy", 100, "Test accuracy (%)")]
    for ax, (m, sc, title) in zip(axes[:2], panels):
        for model in ("bipolar", "binary"):
            mu = np.array([r[f"{model}_{m}_mean"] for r in rows]) * sc
            sd = np.array([r[f"{model}_{m}_std"] for r in rows]) * sc
            ax.errorbar(dims, mu, yerr=sd, color=C[model], lw=2, ms=6, capsize=3,
                        label=LABEL[model], **STYLE[model])
        ax.set_title(title, loc="left", fontsize=10)

    ax = axes[2]
    for model, key in (("bipolar", "mem_bipolar"), ("binary", "mem_binary")):
        ax.plot(dims, [r[key] for r in rows], color=C[model], lw=2, ms=6, label=LABEL[model], **STYLE[model])
    ax.axhline(FLASH_KIB, color="#555555", ls=":", lw=1.2)
    ax.text(dims[0], FLASH_KIB * 1.08, "64 KiB flash", fontsize=8, color="#333333")
    ax.set_yscale("log")
    ax.set_title("Model memory (KiB, item + class)", loc="left", fontsize=10)

    # mark where each model stops fitting on the device
    for model, key in (("bipolar", "mem_bipolar"), ("binary", "mem_binary")):
        over = [r["dim"] for r in rows if r[key] > FLASH_KIB]
        if over:
            for a_ in axes[:2]:
                a_.axvline(min(over), color=C[model], ls=":", lw=1, alpha=0.8)

    for a_ in axes:
        a_.set_xscale("log", base=2)
        a_.set_xticks(dims)
        a_.set_xticklabels([str(d) for d in dims], rotation=45, fontsize=8)
        a_.set_xlabel("hypervector dimension D")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    n = rows[0]["n_seeds"]
    fig.suptitle(f"{encoder} encoder: effect of D (mean ± std over {n} seed{'s' if n > 1 else ''}; "
                 "dotted verticals = first D that exceeds 64 KiB)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(outdir, "dim_sweep.png")
    fig.savefig(path, dpi=160)
    print("wrote", path)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default=None, help="default: ./results_dim_<encoder>")
    p.add_argument("--encoder", default="feature", choices=["record", "ngram", "feature", "sinusoid"])
    p.add_argument("--dims", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096, 5000, 8192, 10000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--levels", type=int, default=64)
    p.add_argument("--ngram", type=int, default=4)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()
    outdir = a.outdir or f"./results_dim_{a.encoder}"
    os.makedirs(outdir, exist_ok=True)

    if not a.report_only:
        splits = D.load_splits(a.cache)
        feats, n_feat = {}, len(FT.FEATURE_NAMES)
        if a.encoder in ("feature", "sinusoid"):
            raw = FT.cached_features(splits)                     # extracted once, reused for every D
            scaler = FT.fit_scaler(raw["train"])                  # fitted on train only
            feats = {k: FT.apply_scaler(v, scaler) for k, v in raw.items()}

        todo = [(d, s) for d in a.dims for s in a.seeds]
        for i, (dim, seed) in enumerate(todo, 1):
            path = os.path.join(outdir, f"dim{dim}_s{seed}.json")
            if os.path.exists(path):
                print(f"[{i}/{len(todo)}] D={dim} seed={seed}: already done, skipping")
                continue
            t0 = time.time()
            r = run_one(splits, feats, n_feat, a, dim, seed)
            with open(path, "w") as f:
                json.dump(r, f, indent=2)
            b, c = r["bipolar"]["test"], r["binary"]["test"]
            print(f"[{i}/{len(todo)}] D={dim:6d} seed={seed}  "
                  f"bipolar acc={100*b['accuracy']:.2f}% Fb={b['fbeta_pooled']:.4f} | "
                  f"binary acc={100*c['accuracy']:.2f}% Fb={c['fbeta_pooled']:.4f} | "
                  f"mem {r['memory_kib']['total_bipolar']:.1f}/{r['memory_kib']['total_binary']:.1f} KiB | "
                  f"{time.time()-t0:.0f}s")

    runs = [r for r in load_runs(outdir) if r["encoder"] == a.encoder]
    if not runs:
        print("no runs found in", outdir)
        return
    rows = summarise(runs)
    print()
    write_tables(rows, outdir)
    figure(rows, outdir, a.encoder)


if __name__ == "__main__":
    main()
