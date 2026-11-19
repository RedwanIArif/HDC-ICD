"""
make_configs.py -- Fig. 3: test F-beta against the hypervector dimension, for binary prototypes (1 bit per
element, Hamming distance) and integer prototypes (15 bits per element, cosine similarity).

Reads the selection summary written by select_config.py (results_select/select_summary.md, a copy is kept
next to this script) and writes ../configs.pdf (vector, single-column width).

  python make_configs.py [path/to/select_summary.md]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_SET, EPOCHS = "no_fft", 20          # the selected feature set and number of retraining epochs
SELECTED = (4096, "binary")


def pm(cell):
    a, b = cell.split("±")
    return float(a), float(b)


def load(path):
    """(D, representation) -> dict(mem, fb, fb_std, sens, spec) for the selected feature set."""
    out = {}
    for line in open(path, encoding="utf-8"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 12 or c[0] != FEATURE_SET or int(c[2]) != EPOCHS:
            continue
        fb, fb_std = pm(c[8])
        out[(int(c[3]), c[4])] = dict(mem=float(c[5]), fb=fb, fb_std=fb_std, sens=c[10], spec=c[11])
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results_selection.md")
    data = load(path)
    dims = sorted({d for d, _ in data})
    for rep in ("bipolar", "binary"):
        print(rep, [(d, data[(d, rep)]["mem"], data[(d, rep)]["fb"], data[(d, rep)]["fb_std"]) for d in dims])

    # Times New Roman, as in the paper. Liberation Serif has the same letter widths and is used where
    # Times New Roman is not installed. Math (the F-beta label) uses STIX, which is also a Times design.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    style = {"binary": dict(color="#2a78d6", marker="o", ls="-", lw=1.8, ms=4.0, label="Binary prototypes, 1 bit (ours)"),
             "bipolar": dict(color="#eb6834", marker="s", ls="--", lw=1.1, ms=3.2, mfc="white", label="Integer prototypes, 15 bits")}
    for rep in ("binary", "bipolar"):
        y = [data[(d, rep)]["fb"] for d in dims]
        e = [data[(d, rep)]["fb_std"] for d in dims]
        ax.errorbar(dims, y, yerr=e, capsize=2, elinewidth=0.7, capthick=0.7, zorder=3, **style[rep])

    # model size in KiB next to every point (stored levels, as in Table II): integer above its error bar,
    # binary below its error bar, so that no label sits on a bar or a line
    # 4096 and 5000 sit close together, so their labels lean apart; the binary label at 5000 goes to the
    # right of the point, below the rising line
    lean = {4096: ("right", 2.0), 5000: ("left", -2.0)}
    for rep, sign in (("bipolar", +1), ("binary", -1)):
        for d in dims:
            r = data[(d, rep)]
            ha, dx = lean.get(d, ("center", 0.0))
            xy, off, va = (d, r["fb"] + sign * r["fb_std"]), (dx, sign * 2.0), "bottom" if sign > 0 else "top"
            if rep == "binary" and d == 5000:
                xy, off, ha, va = (d, r["fb"]), (4.0, -2.5), "left", "top"
            ax.annotate(f"{r['mem']:.1f}", xy=xy, xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=6.5, color="#52514e", zorder=4)

    # deployed configuration: a ring around its point, explained in the legend
    d, rep = SELECTED
    ax.plot([d], [data[(d, rep)]["fb"]], ls="none", marker="o", ms=8.5, mfc="none", mec="#0b0b0b", mew=0.8,
            zorder=5, label="Deployed ($D$ = 4096)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    for lab, d in zip(ax.get_xticklabels(), dims):     # 4096 and 5000 sit close together on a log axis
        if d == 4096:
            lab.set_horizontalalignment("right")
        elif d == 5000:
            lab.set_horizontalalignment("left")
    ax.minorticks_off()
    ax.set_xlim(dims[0] * 0.8, dims[-1] * 1.2)
    ax.set_ylim(0.926, 0.9745)
    ax.set_xlabel("Hypervector dimension $D$")
    ax.set_ylabel(r"Test $F_\beta$")
    ax.grid(True, axis="y", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in ("Binary prototypes, 1 bit (ours)", "Integer prototypes, 15 bits", "Deployed ($D$ = 4096)")]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", frameon=False, fontsize=7, handlelength=2.6, labelspacing=0.25,
              title="Labels: model size in KiB", title_fontsize=6.5)
    fig.tight_layout(pad=0.3)
    out = os.path.join(here, "configs.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
