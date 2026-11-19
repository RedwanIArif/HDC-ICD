"""
make_robustness.py -- Fig. 2: test F-beta against the bit error rate of the stored model.

Reads results_robustness/robustness_no_fft.json if it is there (written by robustness.py), otherwise
uses the values of the summary table below (E7 run, no_fft features, 3 seeds x 10 fault patterns).
Writes ../robustness.pdf (vector, single-column width).

  python make_robustness.py [path/to/robustness_no_fft.json]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BERS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
FLOOR = 0.787
# model: (mean per BER, 10th percentile per BER)  -- from robustness_no_fft.md, used when no JSON is given
TABLE = {   # 3 seeds x 100 fault patterns
    "HDC binary":                 ([0.961, 0.961, 0.962, 0.962, 0.961, 0.954, 0.879], [0.957, 0.957, 0.957, 0.957, 0.957, 0.941, 0.828]),
    "HDC binary, rebuilt levels": ([0.961, 0.961, 0.961, 0.962, 0.961, 0.954, 0.871], [0.957, 0.957, 0.957, 0.957, 0.956, 0.939, 0.797]),
    "LogReg float32":             ([0.955, 0.955, 0.955, 0.937, 0.728, 0.545, 0.332], [0.955, 0.955, 0.955, 0.954, 0.000, 0.000, 0.000]),
    "LogReg int8":                ([0.955, 0.955, 0.954, 0.949, 0.882, 0.822, 0.616], [0.955, 0.955, 0.955, 0.955, 0.815, 0.550, 0.016]),
    "MLP float32":                ([0.959, 0.959, 0.947, 0.782, 0.212, 0.056, 0.010], [0.953, 0.953, 0.953, 0.000, 0.000, 0.000, 0.000]),
    "MLP int8":                   ([0.959, 0.959, 0.958, 0.947, 0.865, 0.710, 0.505], [0.953, 0.953, 0.953, 0.923, 0.757, 0.312, 0.000]),
    "Tree float32":               ([0.950, 0.950, 0.950, 0.943, 0.871, 0.719, 0.471], [0.949, 0.949, 0.949, 0.949, 0.786, 0.177, 0.003]),
}
# one hue per model family; float32 solid, int8 dashed; every series has its own marker (greyscale-safe).
# Keys are the model names written by robustness.py; labels are the names used in the paper.
STYLE = {
    "HDC binary":                 dict(color="#2a78d6", ls="-",  marker="o", lw=1.8, ms=4.0, zorder=5, label="HDC, stored levels (ours)"),
    "HDC binary, rebuilt levels": dict(color="#2a78d6", ls="--", marker="D", lw=1.3, ms=3.2, zorder=5, label="HDC, regenerated levels (ours)"),
    "LogReg float32":             dict(color="#eb6834", ls="-",  marker="s", lw=1.1, ms=3.2, label="Logistic regression, float32"),
    "LogReg int8":                dict(color="#eb6834", ls="--", marker="s", lw=1.1, ms=3.2, mfc="white", label="Logistic regression, int8"),
    "MLP float32":                dict(color="#1baf7a", ls="-",  marker="^", lw=1.1, ms=3.4, label="Neural network, float32"),
    "MLP int8":                   dict(color="#1baf7a", ls="--", marker="^", lw=1.1, ms=3.4, mfc="white", label="Neural network, int8"),
    "Tree float32":               dict(color="#eda100", ls="-",  marker="v", lw=1.1, ms=3.4, label="Decision tree, float32"),
}


def load(path):
    """model -> (mean, p10) over the BERS above, from the JSON written by robustness.py."""
    with open(path) as f:
        d = json.load(f)
    out = {}
    for model, per_ber in d["results"].items():
        vals = {float(k): np.asarray(v, float) for k, v in per_ber.items()}
        out[model] = ([float(vals[b].mean()) for b in BERS], [float(np.percentile(vals[b], 10)) for b in BERS])
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else "results_robustness/robustness_no_fft.json"
    data = load(path) if os.path.exists(path) else TABLE
    print("source:", path if os.path.exists(path) else "built-in table")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.axhline(FLOOR, color="#8a8a86", lw=0.8, ls=":", zorder=1)
    ax.text(1.15e-6, FLOOR - 0.012, "always predict VA (0.787)", color="#52514e", fontsize=6, va="top")
    for model, st in STYLE.items():
        if model not in data:
            continue
        mean, p10 = data[model]
        ax.plot(BERS, mean, **st)
        if model.startswith("HDC binary") and "rebuilt" not in model:
            ax.fill_between(BERS, p10, mean, color=st["color"], alpha=0.15, lw=0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(8e-7, 1.3e-1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Bit error rate of the stored model")
    ax.set_ylabel(r"Test F$_\beta$")
    ax.grid(True, axis="y", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), frameon=False, fontsize=6, ncol=1, handlelength=2.6,
              labelspacing=0.25, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    out = os.path.join(here, "..", "robustness.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
