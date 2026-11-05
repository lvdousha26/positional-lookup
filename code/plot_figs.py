"""plot_figs.py -- publication figures.

Style base: SciencePlots `science.mplstyle` (garrettj403/SciencePlots, MIT),
downloaded to ../styles/science.mplstyle. We disable text.usetex because
ghostscript is unavailable on this machine; STIX is used instead so the figures
still match the Times body text of the paper.

Three explicit design decisions, made after looking at the first draft:
  1. The heatmap uses a white-anchored Blues ramp so that zero cells recede into
     the page instead of dominating it (viridis made 21/25 cells look like a
     single black slab).
  2. learned / sin / RoPE are drawn as ONE curve because they are numerically
     indistinguishable (1.00 at L_tr, 0.00 everywhere beyond). Three overlapping
     lines would suggest three independent measurements where there is one fact.
  3. The cliff probe gets its own figure: it is the only evidence that the
     collapse follows L_tr rather than an absolute position range.
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from arlib import COUPLED, PES, TASKS

LENGTHS = (8, 10, 12, 16, 20)
FIGDIR = "../figures"

LABEL = {"copy": "copy", "prefix_xor": "prefix-XOR", "sort": "sort",
         "swap_halves": "swap-halves", "reverse": "reverse"}
PE_LABEL = {"nope": "NoPE", "learned": "learned", "sin": "sin",
            "rope": "RoPE", "alibi": "ALiBi"}
ORDER = ["copy", "prefix_xor", "sort", "swap_halves", "reverse"]

# SciencePlots palette + near-black for the reference condition
C_NOPE = "#111111"
C_LEARNED = "#FF2C00"
C_SIN = "#00B945"
C_ROPE = "#FF9500"
C_ALIBI = "#845B97"
C_COLLAPSED = "#FF2C00"      # learned/sin/rope coincide -> one colour, one line


def setup():
    plt.style.use("../styles/science.mplstyle")
    plt.rcParams.update({
        "text.usetex": False,                      # no ghostscript on this box
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "savefig.dpi": 400,
        "figure.dpi": 150,
    })


def load():
    df = pd.read_csv("../results/runs.csv")
    df["coupled"] = df["task"].isin(COUPLED)
    return df


# ---------------------------------------------------------------- fig: heatmap
def fig_heatmap(df, summary):
    piv = summary.pivot(index="task", columns="pe", values="mean")
    piv = piv.reindex(ORDER)[[p for p in PES if p in piv.columns]]

    cmap = LinearSegmentedColormap.from_list(
        "paper_blues", ["#ffffff", "#d7e6f4", "#7fb0d8", "#2f6fa8", "#123f66"])

    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=0.8, aspect="auto")

    ax.set_xticks(range(piv.shape[1]), [PE_LABEL.get(p, p) for p in piv.columns], rotation=0)
    ax.set_yticks(range(piv.shape[0]), [LABEL[t] + ("$^{\\dagger}$" if t in COUPLED else "")
                                       for t in piv.index])
    ax.tick_params(top=False, right=False)
    # grid lines so that white (zero) cells are still visible as cells
    ax.set_xticks(np.arange(-.5, piv.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, piv.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#c9d3dc", lw=0.6)
    ax.tick_params(which="minor", length=0)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#1b1b1b" if v < 0.42 else "white",
                    fontweight="bold" if v >= 0.05 else "normal")

    # separator between decoupled/position-free and coupled tasks
    ax.axhline(2.5, color="#444444", lw=1.0, ls=(0, (4, 2)))

    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02, ticks=[0, .2, .4, .6, .8])
    cb.set_label("extrapolation score", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.4)

    ax.set_xlabel("positional encoding", fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig2_heatmap.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig: curves
def fig_curves(df):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.1), sharey=True)
    axes = axes.flatten()

    for ax, task in zip(axes, ORDER):
        sub = df[df.task == task]

        def series(pe_list):
            s = sub[sub.pe.isin(pe_list)]
            m = s[[f"L{L}" for L in LENGTHS]].mean().values
            sd = s[[f"L{L}" for L in LENGTHS]].std().fillna(0).values
            return m, sd

        m, sd = series(["nope"])
        ax.fill_between(LENGTHS, m - sd, m + sd, color=C_NOPE, alpha=0.13, lw=0)
        ax.plot(LENGTHS, m, "-o", color=C_NOPE, lw=1.4, ms=3.2,
                mfc="white", mew=0.9, label="NoPE", zorder=5)

        m, sd = series(["alibi"])
        ax.plot(LENGTHS, m, "--s", color=C_ALIBI, lw=1.1, ms=2.8, label="ALiBi")

        m, sd = series(["learned", "sin", "rope"])
        ax.plot(LENGTHS, m, ":^", color=C_COLLAPSED, lw=1.1, ms=2.8,
                label="learned / sin / RoPE")

        ax.axvspan(8, 20.5, color="#f2f2f2", zorder=0)
        ax.axvline(8, color="#C0504D", lw=1.0, ls=(0, (4, 2)), zorder=1)
        ax.set_title(LABEL[task] + ("$^{\\dagger}$" if task in COUPLED else ""))
        ax.set_xticks([8, 12, 16, 20])
        ax.set_ylim(-0.05, 1.06)
        ax.set_yticks([0, .25, .5, .75, 1.0])
        if task in ("swap_halves", "reverse"):
            ax.set_xlabel("test length $L$")

    for ax in axes[len(ORDER):]:
        ax.axis("off")
    for ax in axes[::3]:
        ax.set_ylabel("exact-match acc.")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[len(ORDER)].legend(handles, labels, loc="center", frameon=False, fontsize=7.5,
                            handlelength=1.8, labelspacing=0.7)
    axes[len(ORDER)].text(0.5, 0.22, "dotted line:\n$L_{\\mathrm{tr}}=8$",
                          ha="center", va="top", fontsize=6.5, color="#666666",
                          transform=axes[len(ORDER)].transAxes)

    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig1_curves.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig: spread
def fig_spread(summary):
    spread = summary.groupby("task")["mean"].agg(lambda s: s.max() - s.min())
    spread = spread.reindex(ORDER)

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    ypos = np.arange(len(spread))[::-1]
    colors = ["#C0504D" if t in COUPLED else "#2F6FA8" for t in spread.index]
    ax.barh(ypos, spread.values, color=colors, height=0.6, edgecolor="none")

    for y, v in zip(ypos, spread.values):
        ax.text(v + 0.014, y, f"{v:.2f}", va="center", fontsize=6.5, color="#333333")

    ax.set_yticks(ypos, [LABEL[t] for t in spread.index])
    ax.set_xlabel("spread in extrapolation score (best $-$ worst)")
    ax.set_xlim(0, max(spread.values) * 1.24)
    ax.set_ylim(-0.6, len(spread) - 0.4)
    ax.legend(handles=[Patch(facecolor="#2F6FA8", label="decoupled / position-free"),
                       Patch(facecolor="#C0504D", label="length-coupled")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=6.5, frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig3_spread.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig: cliff
def fig_cliff():
    runs = {}
    for L in (8, 12):
        for pe in ("nope", "sin"):
            tag = "probe_L12_copy_%s.json" % pe if L == 12 else None
            if tag is None:
                continue
            runs[(L, pe)] = json.load(open(os.path.join("../results", tag)))

    # L_tr=8 comes from the main grid (seed-averaged)
    df = load()
    base = df[(df.task == "copy")]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True)
    for ax, L_tr in zip(axes, (8, 12)):
        if L_tr == 8:
            xs = list(LENGTHS)
            nope = base[base.pe == "nope"][[f"L{L}" for L in LENGTHS]].mean().values
            coll = base[base.pe.isin(["learned", "sin", "rope"])][
                [f"L{L}" for L in LENGTHS]].mean().values
        else:
            c = runs[(12, "nope")]["curve"][-1]
            xs = sorted(int(k[1:]) for k in c if k.startswith("L"))
            nope = np.array([c["L%d" % L] for L in xs], float)
            c2 = runs[(12, "sin")]["curve"][-1]
            coll = np.array([c2["L%d" % L] for L in xs], float)

        ax.plot(xs, nope, "-o", color=C_NOPE, lw=1.4, ms=3.2, mfc="white",
                mew=0.9, label="NoPE")
        ax.plot(xs, coll, ":^", color=C_COLLAPSED, lw=1.2, ms=3.0,
                label="explicit PE")
        ax.axvline(L_tr, color="#999999", lw=0.8, ls=(0, (2, 2)))
        ax.set_title("$L_{\\mathrm{tr}} = %d$" % L_tr)
        ax.set_xlabel("test length $L$")
        ax.set_ylim(-0.05, 1.06)
    axes[0].set_ylabel("exact-match acc.")
    axes[0].legend(loc="upper right", frameon=False, handlelength=1.6)

    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig4_cliff.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig: teaser
def fig_teaser(summary):
    """Figure 1. Left: what the hypothesis predicts (schematic, drawn by hand).
    Right: what we measured. The visual argument is the missing crossover."""
    dec = ["copy", "prefix_xor", "sort"]
    cou = ["swap_halves", "reverse"]

    def means(pe):
        s = summary[summary.pe == pe]
        return (s[s.task.isin(dec)]["mean"].mean(),
                s[s.task.isin(cou)]["mean"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
    xs = [0, 1]

    # ---- (a) schematic prediction ----
    ax = axes[0]
    ax.plot(xs, [0.66, 0.20], "--", color=C_NOPE, lw=1.4, marker="o", ms=4,
            mfc="white", mew=0.9)
    ax.plot(xs, [0.20, 0.62], "--", color=C_SIN, lw=1.4, marker="s", ms=3.6,
            mfc="white", mew=0.9)
    ax.plot(xs, [0.40, 0.40], "--", color="#999999", lw=1.1, marker="^", ms=3.4,
            mfc="white", mew=0.8)
    ax.text(0.5, 0.70, "schematic", ha="center", va="center", fontsize=7,
            style="italic", color="#b03030")
    ax.set_title("(a) what the hypothesis predicts", fontsize=8)
    ax.set_xticks(xs, ["decoupled", "coupled"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.0)
    ax.set_facecolor("#f4f4f4")          # marks this panel as a schematic

    # ---- (b) observed ----
    ax = axes[1]
    for pe, col, mk, lab in [("nope", C_NOPE, "o", "NoPE"),
                             ("sin", C_SIN, "s", "sin"),
                             ("alibi", C_ALIBI, "D", "ALiBi")]:
        a, b = means(pe)
        ax.plot(xs, [a, b], "-", color=col, lw=1.5, marker=mk, ms=3.6,
                mfc="white", mew=0.9, label=lab)
    # learned / rope collapse to the same zero line
    a, b = means("learned")
    ax.plot(xs, [a, b], ":", color=C_COLLAPSED, lw=1.3, marker="^", ms=3.4,
            label="learned / RoPE")
    ax.set_title("(b) what we observe", fontsize=8)
    ax.set_xticks(xs, ["decoupled", "coupled"])
    ax.set_xlim(-0.25, 1.25)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=1.5,
              labelspacing=0.45)

    for ax in axes:
        ax.set_xlabel("task class")

    axes[0].set_ylabel("extrapolation score")
    fig.tight_layout(w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig0_teaser.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig: padded
def fig_padded():
    """The discriminating test: total length fixed, so every position is seen."""
    import glob

    runs = {}
    for p in glob.glob("../results/padded_copy_*.json"):
        r = json.load(open(p))
        runs[r["pe"]] = r["curve"][-1]

    Ls = [8, 10, 12, 16]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style = {"nope": (C_NOPE, "o", "NoPE"),
             "alibi": (C_ALIBI, "D", "ALiBi"),
             "sin": (C_SIN, "s", "sin"),
             "learned": (C_LEARNED, "^", "learned")}

    # shading convention borrowed from Delétang et al. (ICLR 2023, Fig. 2):
    # the region right of the training range is where generalisation is measured.
    ax.axvspan(8, max(Ls) + 0.4, color="#f2f2f2", zorder=0)
    ax.axvline(8, color="#C0504D", lw=1.1, ls=(0, (4, 2)), zorder=1)
    ax.text(max(Ls) - 0.2, 1.06, "generalisation", fontsize=6.2,
            color="#8a8a8a", va="top", ha="right")

    for pe, (col, mk, lab) in style.items():
        if pe not in runs:
            continue
        ys = [runs[pe].get("L%d" % L, float("nan")) for L in Ls]
        ax.plot(Ls, ys, "-", color=col, marker=mk, ms=3.8, lw=1.5,
                mfc="white", mew=0.9, label=lab, zorder=3)

    ax.set_xlabel("effective length $L$")
    ax.set_ylabel("exact-match acc.")
    ax.set_xticks(Ls)
    ax.set_xlim(7.2, max(Ls) + 0.4)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(frameon=False, fontsize=7, loc="center right", handlelength=1.5,
              labelspacing=0.35)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig0_padded.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    df = load()
    summary = pd.read_csv("../results/summary.csv")
    summary["coupled"] = summary["task"].isin(COUPLED)

    fig_padded()
    fig_curves(df)
    fig_heatmap(df, summary)
    fig_spread(summary)
    fig_cliff()
    print("figures ->", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        if f.endswith(".pdf"):
            print("  ", f)


if __name__ == "__main__":
    main()
