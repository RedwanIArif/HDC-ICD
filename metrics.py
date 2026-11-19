import numpy as np


def _rates(tp, fn, fp, tn):
    ppv = 1.0 if tp + fn == 0 else (0.0 if tp + fp == 0 else tp / (tp + fp))
    npv = 1.0 if tn + fp == 0 else (0.0 if tn + fn == 0 else tn / (tn + fn))
    sen = 1.0 if tp + fn == 0 else tp / (tp + fn)
    spe = 1.0 if tn + fp == 0 else tn / (tn + fp)
    return ppv, npv, sen, spe


def _fbeta(ppv, sen, beta=2.0):
    b2 = beta * beta
    return 0.0 if ppv + sen == 0 else (1 + b2) * ppv * sen / (b2 * ppv + sen)


def _counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fn, fp, tn


def primary_report(y_true, y_pred, subject, beta=2.0):
    tp, fn, fp, tn = _counts(y_true, y_pred)
    ppv, npv, sen, spe = _rates(tp, fn, fp, tn)
    total = tp + fn + fp + tn
    out = dict(
        TP=tp, FN=fn, FP=fp, TN=tn,
        accuracy=(tp + tn) / total if total else 0.0,
        sensitivity=sen, specificity=spe, ppv=ppv, npv=npv,
        f1=_fbeta(ppv, sen, 1.0), fbeta_pooled=_fbeta(ppv, sen, beta),
        bac=(sen + spe) / 2,
    )
    per = []
    subject = np.asarray(subject)
    for s in np.unique(subject):
        m = subject == s
        t, f_, p, n = _counts(np.asarray(y_true)[m], np.asarray(y_pred)[m])
        pv, _, se, _ = _rates(t, f_, p, n)
        per.append(_fbeta(pv, se, beta))
    out["fbeta_per_subject"] = float(np.mean(per))
    out["n_subjects"] = int(len(per))
    return out


def subtype_report(y_true, y_pred, names):
    """Per-class precision/recall/F1 + macro-F1, on VA rows only (y_true >= 0)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = y_true >= 0
    y_true, y_pred = y_true[keep], y_pred[keep]
    k = len(names)
    cm = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    rec = np.divide(np.diag(cm), np.maximum(cm.sum(1), 1), dtype=float)
    prec = np.divide(np.diag(cm), np.maximum(cm.sum(0), 1), dtype=float)
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(rec + prec, 1e-12), 0.0)
    out = {n: dict(recall=float(r), precision=float(p), f1=float(f), support=int(cm[i].sum()))
           for i, (n, r, p, f) in enumerate(zip(names, rec, prec, f1))}
    out["macro_f1"] = float(f1.mean())
    out["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    out["balanced_accuracy"] = float(rec.mean())
    out["confusion"] = cm.tolist()
    return out


def plot_confusion_pair(cm_left, cm_right, names, titles, path="fig_confusion.pdf"):
    """Figure 2: row-normalised confusion matrices, greyscale-safe, single-column width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.85))
    for ax, cm, title in zip(axes, (cm_left, cm_right), titles):
        cm = np.asarray(cm, float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7); ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("True", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    print("wrote", path)
    return path


if __name__ == "__main__":
    r = subtype_report([0] * 2005 + [1] * 289 + [2] * 95, [0] * 2389, ("VT", "VFb", "VFt"))
    print(f"collapse check: acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")   # 0.8393 / 0.3042
