"""
robustness.py -- E7: how gracefully does each detector degrade when the bits of its stored model flip?

Motivation: an implant keeps its model in memory for years. Radiation, ageing, and running SRAM or
flash at reduced voltage (to save battery) all cause random bit flips. A model that degrades slowly
under bit errors can run on cheaper, lower-voltage memory without error-correction hardware.

Fault model (the same for every model):
  * each stored model bit flips independently with probability p (the bit error rate, BER)
  * the model is trained and its decision threshold tuned on CLEAN data; faults happen afterwards,
    on the device; the threshold is kept fixed
  * the feature scaler and the threshold are assumed protected (identical for all models)

What is faulted:
  HDC (binary)    every bit of the key vectors, the level vectors and the two binary prototypes
  logreg / MLP    the IEEE-754 float32 weights and biases (all 32 bits of each), and separately an
                  int8-quantised copy (symmetric, per-tensor scale; scales assumed protected)
  tree            float32 split thresholds and leaf probabilities (the node indices are NOT faulted,
                  which favours the tree: a flipped index would usually be worse)

For each BER and each training seed, --trials random fault patterns are drawn; we report the mean
test F-beta and its 10th percentile (a near-worst case over fault patterns).

Usage (next to train_hdc.py):
  python robustness.py                    # no_fft features, HDC D=4096 binary 20 epochs, 3 seeds
  python robustness.py --report_only
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import data as D
import features as FT
import metrics as M
from encoders import build_encoder
from train_hdc import encode_all, fit_centroid, tune_offset

SETS = {
    "all30":  dict(drop=[], only=[]),
    "no_fft": dict(drop=["band_power", "spectral_summary"], only=[]),
}
BERS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]


# ------------------------------------------------------------------------------------ fault injection
def flip_float32(w, p, rng):
    """Flip each of the 32 bits of every float32 value independently with probability p."""
    w = np.asarray(w, np.float32).copy()
    if p <= 0:
        return w
    bits = w.view(np.uint32)
    flips = rng.random(bits.shape + (32,)) < p
    mask = (flips * (np.uint32(1) << np.arange(32, dtype=np.uint32))).sum(-1).astype(np.uint32)
    return (bits ^ mask).view(np.float32)


def quantise_int8(w):
    w = np.asarray(w, np.float32)
    scale = float(np.abs(w).max()) / 127 if np.abs(w).max() > 0 else 1.0
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale


def flip_int8(q, p, rng):
    q = q.copy()
    if p <= 0:
        return q
    u = q.view(np.uint8)
    flips = rng.random(u.shape + (8,)) < p
    mask = (flips * (np.uint8(1) << np.arange(8, dtype=np.uint8))).sum(-1).astype(np.uint8)
    return (u ^ mask).view(np.int8)


def flip_bipolar(t, p, gen):
    """Flip each +-1 element (= one stored bit) with probability p."""
    if p <= 0:
        return t.clone()
    m = torch.rand(t.shape, generator=gen) < p
    return torch.where(m, -t, t)


# ------------------------------------------------------------------------------------ scoring
def to_bits(bipolar):
    return (bipolar <= 0).to(torch.uint8)


def hamming_scores(q_bits, proto_bits):
    ham = torch.stack([(q_bits ^ p).sum(1) for p in proto_bits], dim=1).float()
    return (2.0 * (ham[:, 0] - ham[:, 1]) / q_bits.shape[1]).numpy()


def safe(s):
    s = np.asarray(s, np.float64)
    return np.where(np.isfinite(s), s, -np.inf)      # a NaN/Inf score counts as "not VA"


def logreg_score(coef, intercept, X):
    return safe(X @ coef.ravel() + intercept[0])


def mlp_score(coefs, intercepts, X):
    h = X.astype(np.float64)
    with np.errstate(all="ignore"):
        for i, (W, b) in enumerate(zip(coefs, intercepts)):
            h = h @ W + b
            if i < len(coefs) - 1:
                h = np.maximum(h, 0)
        return safe(1.0 / (1.0 + np.exp(-h[:, 0])))


def tree_score(tree, thresholds, leaf_p1, X):
    left, right, feat = tree.children_left, tree.children_right, tree.feature
    out = np.empty(len(X))
    for i, x in enumerate(X):
        n = 0
        while left[n] != -1:
            n = left[n] if x[feat[n]] <= thresholds[n] else right[n]
        out[i] = leaf_p1[n]
    return safe(out)


# ------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default="./results_robustness")
    p.add_argument("--feature_set", default="no_fft", choices=list(SETS))
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--logreg_C", type=float, default=10.0)          # chosen on validation in E3 (no_fft)
    p.add_argument("--mlp_hidden", type=int, nargs="+", default=[16, 8])
    p.add_argument("--tree_depth", type=int, default=4)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res_path = os.path.join(a.outdir, f"robustness_{a.feature_set}.json")

    if not a.report_only:
        splits = D.load_splits(a.cache)
        raw = FT.cached_features(splits)
        keep = FT.select_columns(SETS[a.feature_set]["drop"], SETS[a.feature_set]["only"])
        sub = {k: v[:, keep] for k, v in raw.items()}
        scaler = FT.fit_scaler(sub["train"])
        X = {k: FT.apply_scaler(v, scaler) for k, v in sub.items()}
        y = {k: np.asarray(splits[k]["y_bin"]) for k in X}
        yt, st_subj = y["test"], splits["test"]["subject"]
        n_feat = len(keep)

        def fb(score, off):
            return M.primary_report(yt, (score > off).astype(int), st_subj, beta=a.beta)["fbeta_pooled"]

        results = {}                                   # model -> ber -> list of F-beta over seeds x trials
        def add(model, ber, v):
            results.setdefault(model, {}).setdefault(str(ber), []).append(float(v))

        for seed in a.seeds:
            t0 = time.time()
            rng = np.random.default_rng(1000 + seed)
            gen = torch.Generator().manual_seed(1000 + seed)

            # ---- HDC: same training as train_hdc.py, then binarised
            torch.manual_seed(seed); np.random.seed(seed)
            enc = build_encoder("feature", a.dim, n_features=n_feat)
            batch = max(32, int(512 * 2048 / a.dim))
            H = {k: encode_all(enc, None, X[k], batch=batch) for k in ("train", "val")}
            cls = fit_centroid(H["train"], torch.as_tensor(y["train"]), 2, epochs=a.epochs, seed=seed)
            proto = torch.where(cls.weight.data > 0, 1.0, -1.0)
            off_h, _ = tune_offset(hamming_scores(to_bits(H["val"]), to_bits(proto)), y["val"], a.beta)
            # rebuilt-level storage: two end vectors A, B and a 6-bit switch level per element
            Lb = to_bits(enc.value).numpy()
            lvA, lvB = Lb[0].astype(np.uint8), Lb[-1].astype(np.uint8)
            sw = np.full(a.dim, 32, np.int64)
            for j in np.where(lvA != lvB)[0]:
                sw[j] = int(np.argmax(Lb[:, j] != lvA[j]))
            kk = np.arange(Lb.shape[0])[:, None]
            assert np.array_equal(np.where(kk >= sw[None, :], lvB[None, :], lvA[None, :]), Lb)

            # ---- baselines, hyperparameters as chosen on validation in E3
            lr = LogisticRegression(C=a.logreg_C, max_iter=2000).fit(X["train"], y["train"])
            off_lr, _ = tune_offset(lr.decision_function(X["val"]), y["val"], a.beta)
            mlp = MLPClassifier(hidden_layer_sizes=tuple(a.mlp_hidden), max_iter=300, early_stopping=True,
                                random_state=seed).fit(X["train"], y["train"])
            off_mlp, _ = tune_offset(mlp.predict_proba(X["val"])[:, 1], y["val"], a.beta)
            tr = DecisionTreeClassifier(max_depth=a.tree_depth, random_state=seed).fit(X["train"], y["train"])
            off_tr, _ = tune_offset(tr.predict_proba(X["val"])[:, 1], y["val"], a.beta)
            tree = tr.tree_
            vals = tree.value[:, 0, :]
            leaf_p1 = (vals[:, 1] / np.maximum(vals.sum(1), 1e-12)).astype(np.float32)
            thr32 = tree.threshold.astype(np.float32)
            lr_q = [quantise_int8(lr.coef_), quantise_int8(lr.intercept_)]
            mlp_q = [quantise_int8(w) for w in mlp.coefs_], [quantise_int8(b) for b in mlp.intercepts_]

            for ber in BERS:
                for t in range(a.trials if ber > 0 else 1):
                    # HDC: fault keys, levels and prototypes, re-encode the test set
                    fenc = copy.copy(enc)
                    fenc.position = flip_bipolar(enc.position, ber, gen)
                    fenc.value = flip_bipolar(enc.value, ber, gen)
                    Ht = encode_all(fenc, None, X["test"], batch=batch)
                    s = hamming_scores(to_bits(Ht), to_bits(flip_bipolar(proto, ber, gen)))
                    add("HDC binary", ber, fb(s, off_h))
                    # HDC with rebuilt levels: fault A, B, the 6 switch-level bits, keys and prototypes
                    fA = lvA ^ (rng.random(a.dim) < ber).astype(np.uint8)
                    fB = lvB ^ (rng.random(a.dim) < ber).astype(np.uint8)
                    fsw = sw ^ ((rng.random((a.dim, 6)) < ber) * (1 << np.arange(6))).sum(1)
                    fL = np.where(kk >= fsw[None, :], fB[None, :], fA[None, :])
                    renc = copy.copy(enc)
                    renc.position = flip_bipolar(enc.position, ber, gen)
                    renc.value = torch.as_tensor(1.0 - 2.0 * fL, dtype=torch.float32)
                    Hr = encode_all(renc, None, X["test"], batch=batch)
                    s = hamming_scores(to_bits(Hr), to_bits(flip_bipolar(proto, ber, gen)))
                    add("HDC binary, rebuilt levels", ber, fb(s, off_h))
                    # logistic regression, float32 and int8
                    add("LogReg float32", ber, fb(logreg_score(flip_float32(lr.coef_, ber, rng),
                                                               flip_float32(lr.intercept_, ber, rng), X["test"]), off_lr))
                    (qc, sc), (qb, sb) = lr_q
                    add("LogReg int8", ber, fb(logreg_score(flip_int8(qc, ber, rng) * sc,
                                                            flip_int8(qb, ber, rng) * sb, X["test"]), off_lr))
                    # MLP, float32 and int8
                    add("MLP float32", ber, fb(mlp_score([flip_float32(w, ber, rng) for w in mlp.coefs_],
                                                         [flip_float32(b, ber, rng) for b in mlp.intercepts_],
                                                         X["test"]), off_mlp))
                    add("MLP int8", ber, fb(mlp_score([flip_int8(q, ber, rng) * s_ for q, s_ in mlp_q[0]],
                                                      [flip_int8(q, ber, rng) * s_ for q, s_ in mlp_q[1]],
                                                      X["test"]), off_mlp))
                    # decision tree, float32 thresholds and leaf probabilities
                    add("Tree float32", ber, fb(tree_score(tree, flip_float32(thr32, ber, rng),
                                                           flip_float32(leaf_p1, ber, rng), X["test"]), off_tr))
            print(f"seed {seed} done in {time.time() - t0:.0f}s")

        bits = {"HDC binary": (n_feat + 64) * a.dim + 2 * a.dim,
                "HDC binary, rebuilt levels": (n_feat + 2 + 6) * a.dim + 2 * a.dim,
                "LogReg float32": 32 * (lr.coef_.size + 1), "LogReg int8": 8 * (lr.coef_.size + 1),
                "MLP float32": 32 * (sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
                "MLP int8": 8 * (sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
                "Tree float32": 32 * 2 * tree.node_count}
        with open(res_path, "w") as f:
            json.dump(dict(config=vars(a), bers=BERS, results=results, model_bits=bits), f, indent=2)

    with open(res_path) as f:
        R = json.load(f)
    results, bers, bits = R["results"], R["bers"], R["model_bits"]

    hdr = ["Model", "Stored bits"] + [("clean" if b == 0 else f"{b:g}") for b in bers]
    lines = [f"## Test F-beta under random bit flips ({R['config']['feature_set']} features)", "",
             "Mean over seeds x fault patterns; 10th percentile in brackets. Always predicting VA scores 0.787.", "",
             "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for m in results:
        cells = []
        for b in bers:
            v = np.array(results[m][str(b)])
            cells.append(f"{v.mean():.3f}" if b == 0 else f"{v.mean():.3f} ({np.percentile(v, 10):.3f})")
        lines.append("| " + " | ".join([m, f"{bits[m]:,}"] + cells) + " |")
    md = "\n".join(lines)
    with open(os.path.join(a.outdir, f"robustness_{R['config']['feature_set']}.md"), "w") as f:
        f.write(md + "\n")
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6"})
    style = {"HDC binary": ("#2a78d6", "o", "-"), "HDC binary, rebuilt levels": ("#2a78d6", "D", ":"),
             "LogReg float32": ("#eb6834", "s", "--"),
             "LogReg int8": ("#eb6834", "s", ":"), "MLP float32": ("#1baf7a", "^", "--"),
             "MLP int8": ("#1baf7a", "^", ":"), "Tree float32": ("#888780", "v", "--")}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = [b for b in bers if b > 0]
    for m in results:
        c, mk, ls = style.get(m, ("k", "o", "-"))
        ax.plot(xs, [np.mean(results[m][str(b)]) for b in xs], color=c, marker=mk, ls=ls, lw=1.8, ms=5, label=m)
    ax.axhline(0.787, color="#555555", ls=":", lw=1)
    ax.text(xs[0], 0.79, "always predict VA", fontsize=7, color="#555555", va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("bit error rate in stored model"); ax.set_ylabel("test F-beta (pooled)")
    ax.set_title("Degradation under random bit flips", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, f"robustness_{R['config']['feature_set']}.png"), dpi=200)
    print("wrote figure to", a.outdir)


if __name__ == "__main__":
    main()
