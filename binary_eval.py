"""
binary_eval.py -- what happens to accuracy when the HDC model is converted from bipolar to binary.

Trains exactly as train_hdc.py does, then evaluates the SAME trained model three ways:

  A  bipolar      : +/-1 queries, integer prototypes, cosine margin       (what train_hdc.py reports)
  B  binary query : queries stored as bits {0,1}, mapped back to +/-1      (should be identical to A)
  C  fully binary : queries AND prototypes as bits, Hamming distance       (what 1 bit/element memory implies)

For C the decision offset is re-tuned on validation (the cosine offset from A is not valid for
Hamming scores) and also reported at offset 0 (plain nearest prototype, no tuning).

Usage (same flags as train_hdc.py):
  python binary_eval.py --encoder feature --dim 2048 --epochs 20 --seed 0
"""
import argparse
import json
import os

import numpy as np
import torch

import data as D
import features as FT
import metrics as M
from encoders import build_encoder
from train_hdc import encode_all, fit_centroid, scores, tune_offset


def to_bits(bipolar):
    """+1 -> 0, -1 -> 1   (the usual convention: XOR of bits == product of +/-1)."""
    return (bipolar <= 0).to(torch.uint8)


def from_bits(bits):
    return 1.0 - 2.0 * bits.float()


def hamming_scores(q_bits, proto_bits):
    """Signed margin toward VA: (Hamming to non-VA) - (Hamming to VA), scaled to the cosine range.
    For +/-1 vectors cos = 1 - 2*Ham/D, so this equals cos(q, P_VA) - cos(q, P_nonVA) of the
    binarised prototypes exactly."""
    D_ = q_bits.shape[1]
    ham = torch.stack([(q_bits ^ p).sum(1) for p in proto_bits], dim=1).float()   # (N, 2)
    return (2.0 * (ham[:, 0] - ham[:, 1]) / D_).numpy()


def evaluate(s, y, subj, offset, beta):
    return M.primary_report(y, (s > offset).astype(int), subj, beta=beta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default="./results_binary")
    p.add_argument("--encoder", default="feature", choices=["record", "ngram", "feature", "sinusoid"])
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--levels", type=int, default=64)
    p.add_argument("--ngram", type=int, default=4)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # ---- identical pipeline to train_hdc.py
    splits = D.load_splits(a.cache)
    feats, n_feat = {}, len(FT.FEATURE_NAMES)
    if a.encoder in ("feature", "sinusoid"):
        raw = FT.cached_features(splits)
        scaler = FT.fit_scaler(raw["train"])
        feats = {k: FT.apply_scaler(v, scaler) for k, v in raw.items()}
    enc = build_encoder(a.encoder, a.dim, n_features=n_feat, n_levels=a.levels,
                        ngram=a.ngram, decimate=a.decimate)
    H = {k: encode_all(enc, splits[k]["x"], feats.get(k)) for k in ("train", "val", "test")}
    y = {k: torch.as_tensor(splits[k]["y_bin"], dtype=torch.long) for k in H}
    cls = fit_centroid(H["train"], y["train"], 2, epochs=a.epochs, lr=a.lr, seed=a.seed)
    yv, yt = y["val"].numpy(), y["test"].numpy()
    sv, st = splits["val"]["subject"], splits["test"]["subject"]

    # sanity: encodings really are +/-1
    for k in H:
        assert torch.all(H[k].abs() == 1), f"{k} encodings are not bipolar"

    res = {"config": vars(a)}

    # ---- A: bipolar (exactly what train_hdc.py reports)
    sA = {k: scores(cls, H[k]) for k in ("val", "test")}
    offA, _ = tune_offset(sA["val"], yv, a.beta)
    res["A_bipolar"] = dict(offset=offA, val=evaluate(sA["val"], yv, sv, offA, a.beta),
                            test=evaluate(sA["test"], yt, st, offA, a.beta))

    # ---- B: queries stored as bits, prototypes unchanged
    Hb = {k: from_bits(to_bits(H[k])) for k in ("val", "test")}
    sB = {k: scores(cls, Hb[k]) for k in ("val", "test")}
    max_diff = max(float(np.abs(sA[k] - sB[k]).max()) for k in sA)
    res["B_binary_query"] = dict(max_score_difference_vs_A=max_diff,
                                 test=evaluate(sB["test"], yt, st, offA, a.beta))

    # ---- C: fully binary, Hamming distance
    W = cls.weight.data
    proto_bits = to_bits(torch.where(W > 0, 1.0, -1.0))
    qb = {k: to_bits(H[k]) for k in ("val", "test")}
    sC = {k: hamming_scores(qb[k], proto_bits) for k in ("val", "test")}
    offC, _ = tune_offset(sC["val"], yv, a.beta)
    res["C_binary_tuned"] = dict(offset=offC, val=evaluate(sC["val"], yv, sv, offC, a.beta),
                                 test=evaluate(sC["test"], yt, st, offC, a.beta))
    res["C_binary_offset0"] = dict(offset=0.0, test=evaluate(sC["test"], yt, st, 0.0, a.beta))
    res["C_binary_with_A_offset"] = dict(offset=offA, test=evaluate(sC["test"], yt, st, offA, a.beta))

    # how much information did binarising the prototypes throw away?
    Wn = torch.nn.functional.normalize(W, dim=1)
    Bn = torch.nn.functional.normalize(from_bits(proto_bits), dim=1)
    res["prototype_cosine_integer_vs_binary"] = [float((Wn[c] * Bn[c]).sum()) for c in range(2)]
    res["decision_agreement_test_A_vs_C"] = float(np.mean((sA["test"] > offA) == (sC["test"] > offC)))
    res["prototype_value_range"] = [float(W.min()), float(W.max())]
    bits_int = int(np.ceil(np.log2(float(W.abs().max()) + 1))) + 1        # signed integer width
    res["class_memory_kib"] = dict(integer=2 * a.dim * bits_int / 8 / 1024,
                                   integer_bits_per_element=bits_int,
                                   binary=2 * a.dim / 8 / 1024)

    # ---- report
    tag = f"{a.encoder}_d{a.dim}_e{a.epochs}_s{a.seed}"
    with open(os.path.join(a.outdir, f"binary_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[{tag}]  bipolar vs binary, TEST set")
    print(f"{'variant':34s} {'acc':>7s} {'sens':>7s} {'spec':>7s} {'Fb pool':>8s} {'Fb subj':>8s}")
    for name, key in [("A bipolar (train_hdc.py)", "A_bipolar"),
                      ("B binary queries only", "B_binary_query"),
                      ("C fully binary, offset re-tuned", "C_binary_tuned"),
                      ("C fully binary, offset 0", "C_binary_offset0"),
                      ("C fully binary, A's offset (wrong)", "C_binary_with_A_offset")]:
        t = res[key]["test"]
        print(f"{name:34s} {100*t['accuracy']:6.2f}% {100*t['sensitivity']:6.2f}% "
              f"{100*t['specificity']:6.2f}% {t['fbeta_pooled']:8.4f} {t['fbeta_per_subject']:8.4f}")
    print(f"\nB vs A max score difference: {max_diff:.2e}  (0 means the bit storage is lossless)")
    print(f"prototype cosine, integer vs sign(): non-VA {res['prototype_cosine_integer_vs_binary'][0]:.3f}, "
          f"VA {res['prototype_cosine_integer_vs_binary'][1]:.3f}")
    print(f"test decisions that agree A vs C(tuned): {100*res['decision_agreement_test_A_vs_C']:.1f}%")
    cm = res["class_memory_kib"]
    print(f"class memory: integer {cm['integer']:.2f} KiB ({cm['integer_bits_per_element']} bits/element) "
          f"vs binary {cm['binary']:.2f} KiB")


if __name__ == "__main__":
    main()
