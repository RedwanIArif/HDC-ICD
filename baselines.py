"""
baselines.py -- E3: classical machine-learning baselines on the SAME features, the SAME patient-wise
split and the SAME decision rule as the HDC detector, so the comparison isolates the classifier.

Protocol (identical to train_hdc.py / select_config.py):
  * features from features.py, columns chosen by --feature_set, scaler fitted on TRAIN only
  * every model outputs a continuous score; the decision threshold is tuned on VALIDATION for F-beta
  * hyperparameters are chosen by mean validation F-beta over seeds; the test set is read once
  * stochastic models (tree, MLP) are trained with 3 seeds and reported as mean +- std

Models (kept small, because the target is a microcontroller):
  logreg  logistic regression                 the linear baseline (HDC prototypes are a linear rule too)
  tree    decision tree, depth tuned          the family of the smallest published IEGM detector
  svm     RBF support-vector machine          the classic kernel baseline (memory grows with support vectors)
  mlp     one or two small hidden layers      the smallest neural-network baseline

Usage (next to train_hdc.py):
  python baselines.py                              # no_fft feature set (the HDC detector's input)
  python baselines.py --feature_set all30          # all 30 features
  python baselines.py --report_only
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data as D
import features as FT
import metrics as M
from train_hdc import tune_offset

warnings.filterwarnings("ignore")

SETS = {
    "all30":        dict(drop=[], only=[]),
    "no_bandpower": dict(drop=["band_power"], only=[]),
    "no_fft":       dict(drop=["band_power", "spectral_summary"], only=[]),
    "beat_only":    dict(drop=[], only=["beat_interval"]),
}
METRICS = ["accuracy", "sensitivity", "specificity", "fbeta_pooled", "fbeta_per_subject"]

# (model name, hyperparameter label, constructor(seed), stochastic?)
GRID = (
    [("logreg", f"C={c}", (lambda s, c=c: LogisticRegression(C=c, max_iter=2000)), False)
     for c in (0.1, 1.0, 10.0)]
    + [("tree", f"depth={d}", (lambda s, d=d: DecisionTreeClassifier(max_depth=d, random_state=s)), True)
       for d in (3, 4, 6, 8)]
    + [("svm", f"C={c}", (lambda s, c=c: SVC(C=c, kernel="rbf", gamma="scale")), False)
       for c in (0.3, 1.0, 3.0)]
    + [("mlp", f"hidden={h}", (lambda s, h=h: MLPClassifier(hidden_layer_sizes=h, max_iter=300,
                                                             early_stopping=True, random_state=s)), True)
       for h in ((8,), (16,), (16, 8))]
)


def score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def model_size(model, n_feat):
    """Parameter count and float32 storage in KiB (what a straightforward MCU port would hold)."""
    if isinstance(model, LogisticRegression):
        params = n_feat + 1
    elif isinstance(model, DecisionTreeClassifier):
        n = model.tree_.node_count
        params = n * 4               # per node: feature index, threshold, left child, right child (leaf value folded in)
    elif isinstance(model, SVC):
        n_sv = int(model.support_vectors_.shape[0])
        params = n_sv * n_feat + n_sv + 1          # support vectors, dual coefficients, bias
    elif isinstance(model, MLPClassifier):
        params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    else:
        params = 0
    return int(params), params * 4 / 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default="./results_baselines")
    p.add_argument("--feature_set", default="no_fft", choices=list(SETS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()
    outdir = os.path.join(a.outdir, a.feature_set)
    os.makedirs(outdir, exist_ok=True)

    if not a.report_only:
        splits = D.load_splits(a.cache)
        raw = FT.cached_features(splits)
        keep = FT.select_columns(SETS[a.feature_set]["drop"], SETS[a.feature_set]["only"])
        sub = {k: v[:, keep] for k, v in raw.items()}
        scaler = FT.fit_scaler(sub["train"])                       # train only
        X = {k: FT.apply_scaler(v, scaler) for k, v in sub.items()}
        y = {k: np.asarray(splits[k]["y_bin"]) for k in X}
        n_feat = len(keep)
        print(f"feature set {a.feature_set}: {n_feat} features, train {len(y['train'])} segments")

        for name, hp, make, stochastic in GRID:
            for seed in (a.seeds if stochastic else a.seeds[:1]):
                path = os.path.join(outdir, f"{name}_{hp.replace('=', '').replace(',', '-').replace(' ', '')}_s{seed}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                model = make(seed).fit(X["train"], y["train"])
                fit_s = time.time() - t0
                sv = score(model, X["val"])
                off, val_fb = tune_offset(sv, y["val"], a.beta)
                st = score(model, X["test"])
                test = M.primary_report(y["test"], (st > off).astype(int), splits["test"]["subject"], beta=a.beta)
                params, kib = model_size(model, n_feat)
                r = dict(model=name, hp=hp, seed=seed, n_features=n_feat, feature_set=a.feature_set,
                         val_fbeta=val_fb, offset=float(off), test=test, params=params, kib_float32=kib,
                         fit_seconds=fit_s)
                with open(path, "w") as f:
                    json.dump(r, f, indent=2)
                print(f"  {name:6s} {hp:16s} seed {seed}  val Fb {val_fb:.4f}  test Fb {test['fbeta_pooled']:.4f}"
                      f"  params {params}  {kib:.2f} KiB  fit {fit_s:.1f}s")

    runs = []
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".json"):
            with open(os.path.join(outdir, f)) as fh:
                runs.append(json.load(fh))
    if not runs:
        print("no runs found in", outdir)
        return

    # per model: choose the hyperparameter with the best mean validation F-beta, report its test numbers
    hdr = ["Model", "Chosen on validation", "Seeds", "Params", "Size (KiB, float32)", "Val Fβ",
           "Test Fβ pooled", "Test Fβ subject", "Sens (%)", "Spec (%)"]
    lines = [f"## Baselines on the {runs[0]['feature_set']} feature set ({runs[0]['n_features']} features)", "",
             "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    full = ["", "## Every hyperparameter tried", "", "| Model | Hyperparameter | Val Fβ | Test Fβ pooled |",
            "|---|---|---|---|"]
    for name in dict.fromkeys(r["model"] for r in runs):
        groups = {}
        for r in runs:
            if r["model"] == name:
                groups.setdefault(r["hp"], []).append(r)
        for hp, rs in groups.items():
            full.append(f"| {name} | {hp} | {np.mean([r['val_fbeta'] for r in rs]):.4f} | "
                        f"{np.mean([r['test']['fbeta_pooled'] for r in rs]):.4f} |")
        hp_best = max(groups, key=lambda h: np.mean([r["val_fbeta"] for r in groups[h]]))
        rs = groups[hp_best]

        def pm(key, scale=1.0, fmt="{:.4f}", src="test"):
            v = np.array([(r[src][key] if src else r[key]) for r in rs], float) * scale
            return (fmt + " ± " + fmt).format(v.mean(), v.std()) if len(v) > 1 else fmt.format(v.mean())

        lines.append("| " + " | ".join([
            name, hp_best, str(len(rs)), str(int(np.mean([r["params"] for r in rs]))),
            f"{np.mean([r['kib_float32'] for r in rs]):.2f}",
            pm("val_fbeta", src=None), pm("fbeta_pooled"), pm("fbeta_per_subject"),
            pm("sensitivity", 100, "{:.1f}"), pm("specificity", 100, "{:.1f}")]) + " |")
    md = "\n".join(lines + full)
    with open(os.path.join(outdir, "baselines_summary.md"), "w") as f:
        f.write(md + "\n")
    print()
    print(md)


if __name__ == "__main__":
    main()
