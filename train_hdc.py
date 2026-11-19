import argparse
import json
import os
import time
import numpy as np
import torch
import torchhd
from torchhd import models

import data as D
import features as FT
import metrics as M
from encoders import build_encoder


def encode_all(enc, x, feats, batch=None):
    src = feats if enc.needs_features else x
    if batch is None:
        if enc.needs_features:
            batch = 512
        else:
            steps = getattr(enc, "n_pos", None) or (1250 // max(getattr(enc, "decimate", 1), 1))
            per = steps * enc.D * (getattr(enc, "n", 1) + 1)
            batch = int(max(4, min(256, 32_000_000 // max(per, 1))))
    out = []
    for i in range(0, len(src), batch):
        chunk = torch.as_tensor(src[i:i + batch], dtype=torch.float32)
        out.append(enc.encode(chunk).float().cpu())
    return torch.cat(out)


def fit_centroid(H, y, n_classes, epochs=0, lr=1.0, class_weight=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    D_ = H.shape[1]
    cls = models.Centroid(D_, n_classes)
    w = torch.ones(n_classes) if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)

    for c in range(n_classes):                      # pass 1: accumulate
        m = y == c
        if m.any():
            cls.weight.data[c] += w[c] * H[m].sum(0)

    # Error-driven correction, mini-batched (AdaptHD). Vectorised with index_add_ rather than
    # a per-sample Python loop, which is ~100x faster and lets this scale to 30k segments.
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(len(H), generator=g)
        n_wrong = 0
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            wn = torch.nn.functional.normalize(cls.weight.data, dim=1)
            pred = (H[sel] @ wn.T).argmax(1)
            yt = y[sel]
            bad = pred != yt
            if not bad.any():
                continue
            n_wrong += int(bad.sum())
            upd = H[sel][bad] * (lr * w[yt[bad]]).unsqueeze(1)
            cls.weight.data.index_add_(0, yt[bad], upd)
            cls.weight.data.index_add_(0, pred[bad], -upd)
        if n_wrong == 0:
            break
    return cls


def scores(cls, H):
    """Signed margin toward class 1 (VA), cosine-normalised so the offset is scale-free."""
    wn = torch.nn.functional.normalize(cls.weight.data, dim=1)
    sim = torch.nn.functional.normalize(H, dim=1) @ wn.T
    return (sim[:, 1] - sim[:, 0]).numpy()


def tune_offset(s_val, y_val, beta=2.0):
    """Pick the decision offset maximising F_beta on validation."""
    cand = np.unique(np.quantile(s_val, np.linspace(0.001, 0.999, 400)))
    best = (-1.0, 0.0)
    for t in cand:
        pred = (s_val > t).astype(int)
        tp = int(((y_val == 1) & (pred == 1)).sum())
        fp = int(((y_val == 0) & (pred == 1)).sum())
        fn = int(((y_val == 1) & (pred == 0)).sum())
        ppv = tp / (tp + fp) if tp + fp else 0.0
        sen = tp / (tp + fn) if tp + fn else 0.0
        b2 = beta * beta
        f = (1 + b2) * ppv * sen / (b2 * ppv + sen) if ppv + sen else 0.0
        if f > best[0]:
            best = (f, float(t))
    return best[1], best[0]


def model_size_bits(enc, n_classes, D_):
    item = enc.item_memory_bits()
    cls = n_classes * D_                    
    return dict(item_memory_bits=int(item), class_memory_bits=int(cls),
                total_bits=int(item + cls), total_kib=(item + cls) / 8 / 1024)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--outdir", default="./results_hdc")
    p.add_argument("--tag", default=None)
    p.add_argument("--encoder", default="feature",
                   choices=["record", "ngram", "feature", "sinusoid"])
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--levels", type=int, default=64)
    p.add_argument("--ngram", type=int, default=4)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20, help="error-driven epochs after pass 1")
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--balance", action="store_true", help="weight classes by inverse frequency")
    p.add_argument("--no_tune", action="store_true", help="plain argmax, no offset tuning")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop_group", nargs="*", default=[],
                   help=f"feature groups to remove: {list(FT.FEATURE_GROUPS)}")
    p.add_argument("--only_group", nargs="*", default=[], help="keep only these groups")
    a = p.parse_args()
    tag = a.tag or f"{a.encoder}_d{a.dim}_e{a.epochs}"
    os.makedirs(a.outdir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    splits = D.load_splits(a.cache)
    need_feat = a.encoder in ("feature", "sinusoid")
    feats, scaler = {}, None
    n_feat = len(FT.FEATURE_NAMES)
    if need_feat:
        t0 = time.time()
        raw = FT.cached_features(splits)
        keep = FT.select_columns(a.drop_group, a.only_group)
        raw = {k: v[:, keep] for k, v in raw.items()}
        n_feat = len(keep)
        scaler = FT.fit_scaler(raw["train"])                 # fitted on train only
        feats = {k: FT.apply_scaler(v, scaler) for k, v in raw.items()}
        print(f"features: {n_feat} per segment, {time.time()-t0:.1f}s")

    enc = build_encoder(a.encoder, a.dim, n_features=n_feat, n_levels=a.levels,
                        ngram=a.ngram, decimate=a.decimate)

    t0 = time.time()
    H = {k: encode_all(enc, splits[k]["x"], feats.get(k)) for k in ("train", "val", "test")}
    t_enc = time.time() - t0
    y = {k: torch.as_tensor(splits[k]["y_bin"], dtype=torch.long) for k in H}

    cw = None
    if a.balance:
        n = np.bincount(y["train"].numpy(), minlength=2)
        cw = (n.sum() / (2 * np.maximum(n, 1))).tolist()

    t0 = time.time()
    cls = fit_centroid(H["train"], y["train"], 2, epochs=a.epochs, lr=a.lr,
                       class_weight=cw, seed=a.seed)
    t_fit = time.time() - t0

    s = {k: scores(cls, H[k]) for k in H}
    offset, val_fb = (0.0, None) if a.no_tune else tune_offset(s["val"], y["val"].numpy(), a.beta)

    res = dict(config=vars(a), tag=tag, offset=offset, val_fbeta_at_offset=val_fb,
               encode_seconds=t_enc, fit_seconds=t_fit,
               per_segment_encode_ms=1000 * t_enc / sum(len(v) for v in H.values()),
               size=model_size_bits(enc, 2, a.dim))
    for k in ("val", "test"):
        pred = (s[k] > offset).astype(int)
        res[k] = M.primary_report(y[k].numpy(), pred, splits[k]["subject"], beta=a.beta)

    path = os.path.join(a.outdir, f"{tag}_s{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    np.savez(os.path.join(a.outdir, f"{tag}_s{a.seed}_scores.npz"),
             s_test=s["test"], y_test=y["test"].numpy(), subject=splits["test"]["subject"])

    t = res["test"]
    print(f"\n[{tag} seed{a.seed}] {a.encoder} D={a.dim} epochs={a.epochs} offset={offset:+.4f}")
    print(f"  size   : item {res['size']['item_memory_bits']/8/1024:.1f} KiB + class "
          f"{res['size']['class_memory_bits']/8/1024:.2f} KiB = {res['size']['total_kib']:.1f} KiB")
    print(f"  TEST   : acc={100*t['accuracy']:.2f}%  sens={100*t['sensitivity']:.2f}%  "
          f"spec={100*t['specificity']:.2f}%")
    print(f"           Fb_pooled={t['fbeta_pooled']:.4f}  Fb_subject={t['fbeta_per_subject']:.4f}  "
          f"F1={t['f1']:.4f}")
    print(f"  timing : encode {res['per_segment_encode_ms']:.3f} ms/segment, fit {t_fit:.1f}s")
    print("wrote", path)


if __name__ == "__main__":
    main()
