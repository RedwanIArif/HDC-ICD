"""
export_model.py -- train the deployed configuration and export it as C source for the STM32F303K8.

Deployed configuration (chosen on validation in E2): feature encoder, the 15 FFT-free features
(amplitude, time domain, periodicity, beat interval), D = 4096, 64 levels, 20 retraining epochs,
fully binary prototypes and Hamming distance. Seed 0 (the same run as results_select/no_fft_e20_d4096_s0).

Writes, into --outdir:
  hdc_model.h / hdc_model.c   the model as packed bit arrays (bit = 1 means -1), both level formats:
                              stored levels (64 x D bits) and rebuilt levels (two end vectors + a
                              6-bit switch level per element), chosen at compile time by HDC_REBUILD_LEVELS
  device_reference.npz        what Python predicts for every test segment (prediction, level indices,
                              Hamming distances): host_eval.py compares the board against it
  memory_report.txt           bytes of every stored array, for both builds

Usage (next to train_hdc.py):
  python export_model.py
"""
import argparse
import os

import numpy as np
import torch

import data as D
import features as FT
import metrics as M
from encoders import build_encoder
from train_hdc import encode_all, fit_centroid, tune_offset

NO_FFT_DROP = ["band_power", "spectral_summary"]
EXPECTED_COLUMNS = [0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 26, 27, 28, 29]


def to_bits(bipolar):
    return (bipolar <= 0).to(torch.uint8)                       # +1 -> 0, -1 -> 1


def pack(bits_2d):
    """(rows, D) array of 0/1 -> (rows, D/32) uint32; element j -> word j>>5, bit j&31."""
    b = np.asarray(bits_2d, np.uint8)
    rows, d = b.shape
    w = b.reshape(rows, d // 32, 32).astype(np.uint64)
    return (w << np.arange(32, dtype=np.uint64)).sum(-1).astype(np.uint32)


def c_array(name, arr, ctype="uint32_t"):
    arr = np.asarray(arr)
    dims = "".join(f"[{s}]" for s in arr.shape)
    flat = arr.ravel()
    if ctype == "float":
        items = [f"{float(v):.9e}f" for v in flat]
    elif ctype == "uint32_t":
        items = [f"0x{int(v):08x}u" for v in flat]
    else:
        items = [str(int(v)) for v in flat]
    lines = [", ".join(items[i:i + 8]) for i in range(0, len(items), 8)]
    return f"const {ctype} {name}{dims} = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default="./device_export")
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--levels", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=2.0)
    a = p.parse_args()
    assert a.dim % 32 == 0 and a.levels == 64, "the C code assumes D divisible by 32 and 64 levels"
    os.makedirs(a.outdir, exist_ok=True)

    # ---- identical training to select_config.py
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits = D.load_splits(a.cache)
    raw = FT.cached_features(splits)
    keep = FT.select_columns(drop=NO_FFT_DROP)
    assert keep == EXPECTED_COLUMNS, keep
    sub = {k: v[:, keep] for k, v in raw.items()}
    lo, hi = FT.fit_scaler(sub["train"])
    feats = {k: FT.apply_scaler(v, (lo, hi)) for k, v in sub.items()}
    n_feat = len(keep)
    enc = build_encoder("feature", a.dim, n_features=n_feat, n_levels=a.levels)
    batch = max(32, int(512 * 2048 / a.dim))
    H = {k: encode_all(enc, None, feats[k], batch=batch) for k in ("train", "val", "test")}
    y = {k: torch.as_tensor(splits[k]["y_bin"], dtype=torch.long) for k in H}
    cls = fit_centroid(H["train"], y["train"], 2, epochs=a.epochs, seed=a.seed)
    proto = torch.where(cls.weight.data > 0, 1.0, -1.0)
    pbits = to_bits(proto)

    def ham(Hk):
        qb = to_bits(Hk)
        return torch.stack([(qb ^ pb).sum(1) for pb in pbits], 1).numpy().astype(np.int64)

    hv, ht = ham(H["val"]), ham(H["test"])
    s_val = (2.0 * torch.as_tensor(hv[:, 0] - hv[:, 1]).float() / a.dim).numpy()
    off, val_fb = tune_offset(s_val, y["val"].numpy(), a.beta)

    # integer threshold: predict VA  <=>  (h0 - h1) > t_int   (exactly the float rule s > off)
    d_range = np.arange(-a.dim, a.dim + 1)
    va = (2.0 * torch.as_tensor(d_range).float() / a.dim).numpy() > np.float32(off)
    t_int = int(d_range[np.argmax(va)] - 1)
    pred_float = ((2.0 * torch.as_tensor(ht[:, 0] - ht[:, 1]).float() / a.dim).numpy() > off).astype(int)
    pred_int = ((ht[:, 0] - ht[:, 1]) > t_int).astype(int)
    assert np.array_equal(pred_float, pred_int), "integer threshold does not reproduce the float rule"
    rep = M.primary_report(y["test"].numpy(), pred_int, splits["test"]["subject"], beta=a.beta)
    print(f"deployed model (seed {a.seed}): val F-beta {val_fb:.4f}  TEST F-beta pooled {rep['fbeta_pooled']:.4f}  "
          f"sens {100 * rep['sensitivity']:.1f}%  spec {100 * rep['specificity']:.1f}%  threshold {t_int}")

    # ---- level vectors: stored form and rebuilt form (end vectors + switch level), checked to be identical
    L = to_bits(enc.value).numpy()                             # (64, D) bits
    A, B = L[0], L[-1]
    s = np.full(a.dim, 32, np.int64)                           # where A == B the switch level is irrelevant
    for j in np.where(A != B)[0]:
        s[j] = int(np.argmax(L[:, j] != A[j]))
    rebuilt = np.array([np.where(k >= s, B, A) for k in range(a.levels)], np.uint8)
    assert np.array_equal(rebuilt, L), "rebuilt levels differ from the stored ones"
    planes = np.array([(s >> b) & 1 for b in range(6)], np.uint8)

    keys = to_bits(enc.position).numpy()
    arrays = {
        "keys": pack(keys), "levels": pack(L), "level_a": pack(A[None])[0], "level_b": pack(B[None])[0],
        "planes": pack(planes), "proto": pack(pbits.numpy()),
    }

    # ---- C source
    h = f"""/* Generated by export_model.py: HDC VA detector, {n_feat} features, D={a.dim}, binary prototypes. */
#ifndef HDC_MODEL_H
#define HDC_MODEL_H
#include <stdint.h>

#define HDC_D          {a.dim}
#define HDC_WORDS      {a.dim // 32}
#define HDC_NFEAT      {n_feat}
#define HDC_NLEVELS    {a.levels}
#define HDC_LEVEL_BITS 6

/* 0: store all 64 level vectors.  1: rebuild them from two end vectors and a 6-bit switch level.
   Both give identical predictions; set it here or with -DHDC_REBUILD_LEVELS=1 in the project. */
#ifndef HDC_REBUILD_LEVELS
#define HDC_REBUILD_LEVELS 0
#endif

extern const float    hdc_scale_lo[HDC_NFEAT];
extern const float    hdc_scale_hi[HDC_NFEAT];
extern const uint32_t hdc_keys[HDC_NFEAT][HDC_WORDS];
#if HDC_REBUILD_LEVELS
extern const uint32_t hdc_level_a[HDC_WORDS];
extern const uint32_t hdc_level_b[HDC_WORDS];
extern const uint32_t hdc_switch_planes[HDC_LEVEL_BITS][HDC_WORDS];
#else
extern const uint32_t hdc_levels[HDC_NLEVELS][HDC_WORDS];
#endif
extern const uint32_t hdc_proto[2][HDC_WORDS];   /* row 0: non-VA, row 1: VA */
extern const int32_t  hdc_threshold;              /* predict VA when (hamming_nonVA - hamming_VA) > threshold */

#endif
"""
    c = ['/* Generated by export_model.py. Bit = 1 means -1; element j is bit (j & 31) of word (j >> 5). */',
         '#include "hdc_model.h"', "",
         c_array("hdc_scale_lo", lo, "float"), c_array("hdc_scale_hi", hi, "float"),
         c_array("hdc_keys", arrays["keys"]),
         "#if HDC_REBUILD_LEVELS",
         c_array("hdc_level_a", arrays["level_a"]), c_array("hdc_level_b", arrays["level_b"]),
         c_array("hdc_switch_planes", arrays["planes"]),
         "#else", c_array("hdc_levels", arrays["levels"]), "#endif",
         c_array("hdc_proto", arrays["proto"]),
         f"const int32_t hdc_threshold = {t_int};", ""]
    with open(os.path.join(a.outdir, "hdc_model.h"), "w") as f:
        f.write(h)
    with open(os.path.join(a.outdir, "hdc_model.c"), "w") as f:
        f.write("\n".join(c))

    # ---- what Python predicts, for the on-board comparison
    lvl_test = (torch.as_tensor(feats["test"]).clamp(0, 1) * (a.levels - 1)).long().numpy()
    np.savez(os.path.join(a.outdir, "device_reference.npz"), pred=pred_int, levels=lvl_test,
             h0=ht[:, 0], h1=ht[:, 1], y=y["test"].numpy(), subject=splits["test"]["subject"],
             features=sub["test"], threshold=t_int, dim=a.dim)

    kib = lambda n: n / 1024
    common = lo.nbytes + hi.nbytes + arrays["keys"].nbytes + arrays["proto"].nbytes + 4
    stored = common + arrays["levels"].nbytes
    rebuilt = common + arrays["level_a"].nbytes + arrays["level_b"].nbytes + arrays["planes"].nbytes
    report = [f"keys          {arrays['keys'].nbytes:7d} B  ({n_feat} x {a.dim} bits)",
              f"prototypes    {arrays['proto'].nbytes:7d} B  (2 x {a.dim} bits)",
              f"scaler        {lo.nbytes + hi.nbytes:7d} B  (2 x {n_feat} float32)",
              f"levels stored {arrays['levels'].nbytes:7d} B  (64 x {a.dim} bits)",
              f"levels rebuilt{arrays['level_a'].nbytes + arrays['level_b'].nbytes + arrays['planes'].nbytes:7d} B  "
              f"(2 end vectors + 6-bit switch level per element)",
              f"MODEL TOTAL, stored levels : {stored:6d} B = {kib(stored):.2f} KiB",
              f"MODEL TOTAL, rebuilt levels: {rebuilt:6d} B = {kib(rebuilt):.2f} KiB"]
    with open(os.path.join(a.outdir, "memory_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()
