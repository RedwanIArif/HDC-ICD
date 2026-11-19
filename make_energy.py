"""
make_energy.py -- Fig. 3: estimated energy of the HDC stages (encoding and classification) per
window on dedicated HDC hardware, against the dimension D.

Energy model (per window, dynamic energy, 45 nm, 0.9 V, Horowitz ISSCC 2014):
  * SRAM read of 64 bits costs 10 pJ for an 8 KB array, 20 pJ for 32 KB, 100 pJ for 1 MB,
    i.e. about 10 pJ * sqrt(size / 8 KB). We use this square-root fit, with 8 KB as the floor,
    and halve it for a 32-bit word.
  * every 32-bit logic operation (XOR, AND, population count, accumulation) is charged as one
    32-bit integer addition, 0.1 pJ (an upper bound for XOR and AND).
Operation counts per 32-element word position (D/32 positions per window), n = 15, b = 6:
  stored levels      reads: n keys + n levels + 2 prototypes          = 2n + 2      words
  regenerated levels reads: n keys + A + B + b planes + 2 prototypes  = n + 2 + b + 2 words
  binding n XOR, bundling 8 ops per input (4-bit bit-sliced counter), majority = counter bit 3,
  distances 2 x (XOR + popcount + accumulate) = 6, regeneration 2b ops per level word.
Voltage-scaled memory: 30 % less SRAM read energy (Stutz et al., MLSys 2021).
"""
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n, M, b = 15, 64, 6
E_ADD32 = 0.1e-12             # J, 32-bit integer add, 45 nm
E_SRAM64_8KB = 10e-12         # J, 64-bit read from an 8 KB SRAM, 45 nm
VS_SAVING = 0.30              # SRAM energy saved by voltage scaling


def model_bits(D, regen):
    return ((n + 2 + b + 2) if regen else (n + M + 2)) * D + 64 * n + 32


def e_read32(size_bytes):
    return 0.5 * E_SRAM64_8KB * math.sqrt(max(size_bytes, 8192) / 8192)


def energy(D, regen, vscale=False):
    pos = D // 32
    reads = (n + 2 + b + 2) if regen else (2 * n + 2)
    ops = n + 8 * n + 6 + (2 * b * n if regen else 0)
    e_mem = pos * reads * e_read32(model_bits(D, regen) / 8)
    if vscale:
        e_mem *= 1 - VS_SAVING
    e_log = pos * ops * E_ADD32
    return e_mem, e_log


dims = [1024, 2048, 4096, 8192, 16384]
rows = {}
for name, regen, vs in [("stored", False, False), ("regen", True, False), ("regen_vs", True, True)]:
    rows[name] = [sum(energy(D, regen, vs)) * 1e9 for D in dims]
for D in dims:
    em, el = energy(D, True)
    ems, els = energy(D, False)
    print(f"D={D:5d}  regen: mem {em*1e9:6.1f} nJ logic {el*1e9:5.1f} nJ total {(em+el)*1e9:6.1f} nJ"
          f" | stored total {(ems+els)*1e9:6.1f} nJ | regen+VS {sum(energy(D, True, True))*1e9:6.1f} nJ"
          f" | model {model_bits(D, True)/8/1024:.1f} / {model_bits(D, False)/8/1024:.1f} KiB")
cyc = 3.4e-3 * 64e6
print(f"programmable core, 45 nm, 70 pJ/instruction, <= {cyc:.0f} instructions: {cyc*70e-12*1e6:.1f} uJ")

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.45, 1.45))
ax.plot(dims, rows["stored"], color="#eb6834", marker="s", ms=3.2, mfc="white", lw=1.1, ls="--",
        label="Stored levels")
ax.plot(dims, rows["regen"], color="#2a78d6", marker="o", ms=4.0, lw=1.8, label="Regenerated levels (ours)")
ax.plot(dims, rows["regen_vs"], color="#1a9e77", marker="^", ms=3.6, lw=1.1, ls="-.",
        label="Regenerated, voltage-scaled memory")
ax.plot([4096], [rows["regen"][2]], ls="none", marker="o", ms=8.5, mfc="none", mec="#0b0b0b", mew=0.8,
        label="Deployed ($D$ = 4096)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter
ax.yaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300]))
ax.yaxis.set_major_formatter(FixedFormatter(["3", "10", "30", "100", "300"]))
ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims]); ax.minorticks_off()
ax.set_ylim(2.5, 500)
ax.set_xlim(dims[0] * 0.8, dims[-1] * 1.25)
ax.set_xlabel("Hypervector dimension $D$")
ax.set_ylabel("Energy per window (nJ)")
ax.grid(True, axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(width=0.6, length=2.5)
ax.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=2.4, labelspacing=0.2)
fig.tight_layout(pad=0.3)
fig.savefig("energy.pdf"); fig.savefig("energy.png", dpi=300)
print("wrote energy.pdf")
