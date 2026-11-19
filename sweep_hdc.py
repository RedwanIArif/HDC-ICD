import argparse
import glob
import json
import os
import subprocess
import sys
import numpy as np

PY = sys.executable
CNN_REF = dict(name="1D-CNN (single-task)", kib=None, fb_pooled=None)
TRIVIAL_FB = None   # F-beta of "always predict VA" on the test set, computed from the labels


def grid(quick):
    """(tag, args). Encoder comparison at matched D, then a D sweep, then two ablations."""
    dims_enc = [2048] if quick else [2048]
    jobs = []
    for enc in ["record", "ngram", "feature", "sinusoid"]:
        for d in dims_enc:
            jobs.append((f"enc_{enc}_d{d}", ["--encoder", enc, "--dim", str(d), "--epochs", "20"]))
    dims = [256, 1024, 4096] if quick else [128, 256, 512, 1024, 2048, 4096, 8192]
    for d in dims:
        jobs.append((f"dim_feature_d{d}", ["--encoder", "feature", "--dim", str(d), "--epochs", "20"]))
    if not quick:
        for d in [1024, 4096]:
            jobs.append((f"dim_sinusoid_d{d}", ["--encoder", "sinusoid", "--dim", str(d), "--epochs", "20"]))
    for ep in [0, 1, 5, 20, 50]:
        jobs.append((f"abl_epochs{ep}", ["--encoder", "feature", "--dim", "2048", "--epochs", str(ep)]))
    jobs.append(("abl_notune", ["--encoder", "feature", "--dim", "2048", "--epochs", "20", "--no_tune"]))
    jobs.append(("abl_balance", ["--encoder", "feature", "--dim", "2048", "--epochs", "20", "--balance"]))
    import features as FT
    for g in FT.FEATURE_GROUPS:
        jobs.append((f"grp_drop_{g}", ["--encoder", "feature", "--dim", "2048", "--epochs", "20",
                                       "--drop_group", g]))
        jobs.append((f"grp_only_{g}", ["--encoder", "feature", "--dim", "2048", "--epochs", "20",
                                       "--only_group", g]))
    if not quick:
        for lv in [8, 16, 32, 128]:
            jobs.append((f"abl_levels{lv}", ["--encoder", "feature", "--dim", "2048",
                                             "--epochs", "20", "--levels", str(lv)]))
    return jobs


def run(tag, extra, seed, outdir, cache):
    path = os.path.join(outdir, f"{tag}_s{seed}.json")
    if os.path.exists(path):
        print(f"  skip {tag} s{seed}")
        return
    cmd = [PY, "train_hdc.py", "--tag", tag, "--seed", str(seed),
           "--outdir", outdir, "--cache", cache] + extra
    print("  " + " ".join(cmd[2:]))
    subprocess.run(cmd, check=True)


def collect(outdir, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(outdir, f"{tag}_s*.json"))):
        if f.endswith("_scores.npz"):
            continue
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def ms(v, fmt="{:.3f}"):
    if not v:
        return "--"
    v = np.asarray(v, float)
    return fmt.format(v.mean()) if len(v) == 1 else (fmt + "$\\pm$" + fmt).format(v.mean(), v.std())


def pick(runs, *keys):
    out = []
    for r in runs:
        c = r
        ok = True
        for k in keys:
            if isinstance(c, dict) and k in c:
                c = c[k]
            else:
                ok = False
                break
        if ok and c is not None:
            out.append(c)
    return out


def rows_for(outdir, tags):
    table = []
    for tag in tags:
        r = collect(outdir, tag)
        if not r:
            continue
        cfg = r[0]["config"]
        table.append(dict(
            tag=tag, encoder=cfg["encoder"], dim=cfg["dim"], epochs=cfg["epochs"],
            levels=cfg["levels"], tuned=not cfg["no_tune"], balance=cfg.get("balance", False),
            kib=r[0]["size"]["total_kib"], item_kib=r[0]["size"]["item_memory_bits"] / 8 / 1024,
            enc_ms=np.mean(pick(r, "per_segment_encode_ms")),
            acc=pick(r, "test", "accuracy"), sens=pick(r, "test", "sensitivity"),
            spec=pick(r, "test", "specificity"),
            fbp=pick(r, "test", "fbeta_pooled"), fbs=pick(r, "test", "fbeta_per_subject"),
            val_fbp=pick(r, "val", "fbeta_pooled"),
            n=len(r)))
    return table


def md_table(rows, cols, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c(r) for c in cols) + " |")
    return "\n".join(out)


def figures(outdir, all_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- accuracy vs memory, with the ICD flash budget marked
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for enc, mk in [("feature", "o"), ("sinusoid", "s"), ("record", "^"), ("ngram", "v")]:
        pts = [(r["kib"], np.mean(r["fbp"])) for r in all_rows
               if r["encoder"] == enc and r["fbp"] and r["tag"].startswith(("dim_", "enc_"))]
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", label=f"HDC, {enc}")
    if CNN_REF["kib"] is not None and CNN_REF["fb_pooled"] is not None:
        ax.scatter([CNN_REF["kib"]], [CNN_REF["fb_pooled"]], marker="*", s=220, zorder=5,
                   color="k", label=f"{CNN_REF['name']}, {CNN_REF['kib']} KiB")
    if TRIVIAL_FB is not None:
        ax.axhline(TRIVIAL_FB, ls=":", lw=1.2, color="r",
                   label=f"always predict VA ($F_\\beta$={TRIVIAL_FB:.3f})")
    ax.axvline(64, ls="--", lw=1, color="0.4")
    ax.text(64, ax.get_ylim()[0], " STM32F303K8 flash (64 KiB)", rotation=90,
            va="bottom", fontsize=8, color="0.3")
    ax.set_xscale("log"); ax.set_xlabel("Model memory (KiB, item + class)")
    ax.set_ylabel(r"Test $F_\beta$ ($\beta$=2), pooled")
    ax.set_title("VA vs. non-VA on TDC'22: accuracy vs. memory")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_accuracy_vs_memory.png"), dpi=160)

    pts = sorted((r["dim"], np.mean(r["fbp"]), np.std(r["fbp"]) if r["n"] > 1 else 0)
                 for r in all_rows if r["tag"].startswith("dim_feature") and r["fbp"])
    if pts:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    marker="o", capsize=3)
        if CNN_REF["fb_pooled"] is not None:
            ax.axhline(CNN_REF["fb_pooled"], ls="--", color="k", lw=1,
                       label=f"1D-CNN baseline ({CNN_REF['fb_pooled']})")
        ax.set_xscale("log", base=2); ax.set_xlabel("Hypervector dimension $D$")
        ax.set_ylabel(r"Test $F_\beta$ ($\beta$=2)")
        ax.set_title("Effect of hypervector dimension (feature encoder)")
        ax.grid(alpha=.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_dimension.png"), dpi=160)

    pts = sorted((r["epochs"], np.mean(r["fbp"])) for r in all_rows
                 if r["tag"].startswith("abl_epochs") and r["fbp"])
    if pts:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
        ax.set_xlabel("Error-driven retraining epochs (0 = single-pass centroid)")
        ax.set_ylabel(r"Test $F_\beta$")
        ax.set_title("Single-pass HDC underfits this dataset")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_retraining.png"), dpi=160)

    best = max((r for r in all_rows if r["val_fbp"]), key=lambda r: np.mean(r["val_fbp"]), default=None)
    f = os.path.join(outdir, f"{best['tag']}_s0_scores.npz") if best else None
    if f and os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        with open(os.path.join(outdir, f"{best['tag']}_s0.json")) as fh:
            off = json.load(fh)["offset"]
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.hist(z["s_test"][z["y_test"] == 0], bins=60, alpha=.6, label="non-VA")
        ax.hist(z["s_test"][z["y_test"] == 1], bins=60, alpha=.6, label="VA")
        ax.axvline(off, color="k", ls="--", lw=1.2, label=f"tuned offset ({off:+.3f})")
        ax.axvline(0, color="0.5", ls=":", lw=1, label="plain argmax (0)")
        ax.set_xlabel("similarity margin toward VA"); ax.set_ylabel("segments")
        ax.set_title(f"Decision margin, {best['tag']}")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_margin.png"), dpi=160)
    print("figures written to", outdir)


def report(outdir):
    tags = sorted({os.path.basename(f).rsplit("_s", 1)[0]
                   for f in glob.glob(os.path.join(outdir, "*.json"))})
    rows = rows_for(outdir, tags)
    if not rows:
        print("no results found in", outdir)
        return
    global TRIVIAL_FB
    npz = glob.glob(os.path.join(outdir, "*_scores.npz"))
    if npz:
        y = np.load(npz[0], allow_pickle=True)["y_test"]
        ppv = y.mean()
        TRIVIAL_FB = 5 * ppv / (4 * ppv + 1)
    enc_rows = [r for r in rows if r["tag"].startswith("enc_")]
    dim_rows = sorted([r for r in rows if r["tag"].startswith("dim_")], key=lambda r: r["dim"])
    abl_rows = [r for r in rows if r["tag"].startswith("abl_")]
    grp_rows = [r for r in rows if r["tag"].startswith("grp_")]

    def C(k, fmt="{:.3f}", scale=1.0):
        return lambda r: (ms([v * scale for v in r[k]], fmt) if isinstance(r[k], list)
                          else fmt.format(r[k] * scale))
    hdr = ["Encoder", "D", "Mem (KiB)", "Enc (ms/seg)", "Acc (%)", "Sens (%)", "Spec (%)",
           "Fb val", "Fb pooled", "Fb subject"]
    cols = [lambda r: r["encoder"], lambda r: str(r["dim"]), C("kib", "{:.1f}"),
            C("enc_ms", "{:.2f}"), C("acc", "{:.1f}", 100), C("sens", "{:.1f}", 100),
            C("spec", "{:.1f}", 100), C("val_fbp"), C("fbp"), C("fbs")]

    lines = ["# HDC for VA vs. non-VA detection on TDC'22", "",
             "Protocol: official patient-wise train/test split; ten training subjects held out",
             "for validation; the decision offset is tuned on validation only; the test set is",
             "read once. Metric is $F_\\beta$ with $\\beta$=2 (recall-weighted), reported both",
             "pooled over segments and averaged per subject (the contest convention).", "",
             "## 1. Encoder comparison (matched D)", "",
             md_table(enc_rows, cols, hdr), "",
             "## 2. Effect of hypervector dimension", "",
             md_table(dim_rows, cols, hdr), "",
             "## 3. Ablations", "",
             md_table(abl_rows, [lambda r: r["tag"].replace("abl_", "")] + cols[1:],
                      ["Variant"] + hdr[1:]), "",
             "## 4. Feature-group ablation (feature encoder, D=2048)", "",
             md_table(grp_rows, [lambda r: r["tag"].replace("grp_", "")] + cols[1:],
                      ["Variant"] + hdr[1:]), "",
             "## 5. Reference points", "",
             (f"Always predicting VA scores pooled $F_\\beta$ {TRIVIAL_FB:.3f} on this test set; "
              "that, not zero, is the floor for this metric." if TRIVIAL_FB else ""), "",
             (f"Single-task 1D-CNN: {CNN_REF['kib']} KiB of weights, pooled $F_\\beta$ "
              f"{CNN_REF['fb_pooled']}." if CNN_REF["kib"] is not None
              else "No CNN reference supplied (use --cnn_kib and --cnn_fb)."),
             "",
             "Caveat: HDC memory is computed from hypervector counts, not read from a deployed",
             "MCU build, and encode time is desktop-CPU time, not on-device latency.", ""]
    # Selected on VALIDATION F-beta. Choosing the best row by test F-beta would be test-set
    # model selection.
    best = max(rows, key=lambda r: np.mean(r["val_fbp"]) if r["val_fbp"] else -1)
    lines += ["## 6. Headline numbers (configuration chosen on validation)", "",
              f"- Best by validation: **{best['tag']}** ({best['encoder']}, D={best['dim']}), "
              f"pooled $F_\\beta$ {ms(best['fbp'])}, per-subject {ms(best['fbs'])}, "
              f"{best['kib']:.1f} KiB.",
              f"- Item memory is {best['item_kib']:.1f} KiB of that {best['kib']:.1f} KiB, i.e. "
              f"{100*best['item_kib']/best['kib']:.0f}%: in HDC the encoder, not the classifier, "
              f"is what has to fit on the device.", ""]
    with open(os.path.join(outdir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    figures(outdir, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--outdir", default="./results_hdc")
    p.add_argument("--cache", default="cache_iegm.npz")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--report_only", action="store_true")
    p.add_argument("--cnn_kib", type=float, default=None, help="weight ROM of a reference CNN, in KiB")
    p.add_argument("--cnn_fb", type=float, default=None, help="pooled F-beta of a reference CNN")
    a = p.parse_args()
    CNN_REF["kib"], CNN_REF["fb_pooled"] = a.cnn_kib, a.cnn_fb
    if not a.report_only:
        os.makedirs(a.outdir, exist_ok=True)
        for tag, extra in grid(a.quick):
            for s in range(a.seeds):
                run(tag, extra, s, a.outdir, a.cache)
    report(a.outdir)


if __name__ == "__main__":
    main()
