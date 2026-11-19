"""
c_check.py -- optional desktop check before flashing: compile the C detector with gcc (both level
formats), run it on test segments, and compare with Python's predictions in device_reference.npz.

Usage (next to train_hdc.py, after export_model.py, needs gcc on the PATH):
  python c_check.py --n 1000
"""
import argparse
import os
import subprocess

import numpy as np

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--export", default="./device_export")
    p.add_argument("--n", type=int, default=1000)
    a = p.parse_args()

    splits = D.load_splits(a.cache)
    x = np.asarray(splits["test"]["x"][..., 0], np.float32)[: a.n]
    ref = np.load(os.path.join(a.export, "device_reference.npz"), allow_pickle=True)
    seg_path = os.path.join(a.export, "check_segments.bin")
    x.astype("<f4").tofile(seg_path)

    for rebuild in (0, 1):
        exe = os.path.join(a.export, f"hdc_check_{rebuild}")
        cmd = ["gcc", "-O2", "-std=c99", "-ffp-contract=off", f"-DHDC_REBUILD_LEVELS={rebuild}",
               "-I", a.export, "-I", HERE,
               os.path.join(HERE, "hdc_infer.c"), os.path.join(a.export, "hdc_model.c"),
               os.path.join(HERE, "host_test.c"), "-lm", "-o", exe]
        subprocess.run(cmd, check=True)
        out_path = os.path.join(a.export, f"check_out_{rebuild}.bin")
        subprocess.run([exe, seg_path, out_path], check=True)
        rec = np.fromfile(out_path, dtype="<i2").reshape(-1, 18)
        n = len(rec)
        agree = np.mean(rec[:, 0] == ref["pred"][:n])
        lev = np.mean(np.all(rec[:, 3:] == ref["levels"][:n], axis=1))
        per_feat = np.mean(rec[:, 3:] == ref["levels"][:n], axis=0)
        ham = np.mean((rec[:, 1] == ref["h0"][:n]) & (rec[:, 2] == ref["h1"][:n]))
        print(f"{'rebuilt' if rebuild else 'stored '} levels: {n} segments  predictions agree {100 * agree:.2f}%  "
              f"all 15 levels agree {100 * lev:.2f}%  Hamming agree {100 * ham:.2f}%")
        print("   per-feature level agreement (%):", " ".join(f"{100 * v:.1f}" for v in per_feat))


if __name__ == "__main__":
    main()
