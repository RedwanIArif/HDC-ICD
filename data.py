"""
data.py -- loading, subtype labels, subject IDs, and the patient-wise validation split.

Replaces the DataGenerator in training_file.py. Differences that matter:
  * subject ID is carried through, so per-subject F_beta (the TDC'22 convention) is computable;
  * a VALIDATION split is carved out of TRAIN by subject, so the test set is never used for
    model selection, checkpointing or hyperparameter choice;
  * segments are cached to an .npz so 24,588 text files are parsed once, not once per run.
"""
import csv
import os
import collections
import numpy as np

SEGMENT_SIZE = 1250
VA_SUBTYPE_MAP = {"VT": 0, "VFb": 1, "VFt": 2}          # 3-class subtype task
VF_LIKE = {"VFb", "VFt"}                                 # merged "VF-like" class for the 2-class task


VAL_SUBJECTS = ["S13", "S31", "S35", "S36", "S45", "S59", "S60", "S81", "S86", "S89"]


def subject_of(filename):
    return os.path.basename(filename).split("-")[0]


def rhythm_of(filename):
    parts = os.path.basename(filename).split("-")
    return parts[1] if len(parts) >= 2 else ""


def load_index(indice_dir, mode):
    """Read {train,test}_indice.csv -> list of (filename, binary_label)."""
    path = os.path.join(indice_dir, f"{mode}_indice.csv")
    out = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)                                # header: label,Filename
        for row in reader:
            if len(row) < 2:
                continue
            if ".txt" in row[1]:
                label, filename = int(row[0].strip()), row[1].strip()
            else:
                label, filename = int(row[1].strip()), row[0].strip()
            out.append((filename, label))
    return out


def txt_to_numpy(path, size=SEGMENT_SIZE):
    vals = np.zeros(size, dtype=np.float32)
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= size:
                break
            line = line.strip()
            if line:
                vals[i] = float(line.split()[0])
    return vals


def build_cache(data_dir, indice_dir, cache="cache_iegm.npz", size=SEGMENT_SIZE):
    """Parse every segment once and store x, y_bin, y_sub3, subject, rhythm, split."""
    xs, yb, ys, subj, rhy, spl = [], [], [], [], [], []
    for mode in ("train", "test"):
        missing = 0
        for filename, label in load_index(indice_dir, mode):
            path = os.path.join(data_dir, filename)
            if not os.path.isfile(path):
                missing += 1
                continue
            r = rhythm_of(filename)
            xs.append(txt_to_numpy(path, size))
            yb.append(label)
            ys.append(VA_SUBTYPE_MAP.get(r, -1))
            subj.append(subject_of(filename))
            rhy.append(r)
            spl.append(mode)
        if missing:
            print(f"[warn] {missing} {mode} files listed in the index were not found on disk")

    np.savez_compressed(
        cache,
        x=np.asarray(xs, np.float32)[..., None],          # (N, 1250, 1)
        y_bin=np.asarray(yb, np.int32),
        y_sub3=np.asarray(ys, np.int32),                  # 0/1/2 for VT/VFb/VFt, -1 for non-VA
        subject=np.asarray(subj), rhythm=np.asarray(rhy), split=np.asarray(spl),
    )
    print(f"cached {len(xs)} segments -> {cache}")
    return cache


def load_splits(cache="cache_iegm.npz", subtype_task="vt_vs_vf"):

    d = np.load(cache, allow_pickle=True)
    x, y_bin, y3 = d["x"], d["y_bin"], d["y_sub3"]
    subject, split = d["subject"], d["split"]

    if subtype_task == "three":
        y_sub, names = y3.copy(), ("VT", "VFb", "VFt")
    elif subtype_task == "vt_vs_vf":
        y_sub = np.where(y3 < 0, -1, np.where(y3 == 0, 0, 1)).astype(np.int32)
        names = ("VT", "VF-like")
    else:
        raise ValueError(subtype_task)

    is_val = np.isin(subject, VAL_SUBJECTS)
    masks = {
        "train": (split == "train") & ~is_val,
        "val":   (split == "train") & is_val,
        "test":  (split == "test"),
    }
    out = {"subtype_names": names}
    for name, m in masks.items():
        out[name] = dict(x=x[m], y_bin=y_bin[m], y_sub=y_sub[m], subject=subject[m])
    return out


def describe(splits):
    names = splits["subtype_names"]
    for name in ("train", "val", "test"):
        s = splits[name]
        line = [f"{name:5s} n={len(s['x']):5d} subj={len(set(s['subject'])):3d}",
                f"VA={int((s['y_bin']==1).sum()):5d} nonVA={int((s['y_bin']==0).sum()):5d}"]
        for c, cn in enumerate(names):
            m = s["y_sub"] == c
            line.append(f"{cn}={int(m.sum()):4d}({len(set(s['subject'][m])):2d} subj)")
        print("  ".join(line))


def class_counts(y_sub, n_classes):
    return [int((y_sub == c).sum()) for c in range(n_classes)]


def rhythm_subject_table(cache="cache_iegm.npz"):
    d = np.load(cache, allow_pickle=True)
    rows = {}
    for mode in ("train", "test"):
        m = d["split"] == mode
        cnt = collections.Counter(d["rhythm"][m])
        subs = collections.defaultdict(set)
        for r, s in zip(d["rhythm"][m], d["subject"][m]):
            subs[r].add(s)
        rows[mode] = {r: (cnt[r], len(subs[r])) for r in cnt}
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path_data", default="./tinyml_contest_data_training/")
    p.add_argument("--path_indices", default="./data_indices")
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--subtype_task", default="vt_vs_vf", choices=["three", "vt_vs_vf"])
    a = p.parse_args()
    if not os.path.exists(a.cache):
        build_cache(a.path_data, a.path_indices, a.cache)
    describe(load_splits(a.cache, a.subtype_task))
    print()
    for mode, rows in rhythm_subject_table(a.cache).items():
        print(mode, {r: f"{n} seg/{k} subj" for r, (n, k) in sorted(rows.items(), key=lambda kv: -kv[1][0])})
