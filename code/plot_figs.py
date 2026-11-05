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

# Notebook-local copies of the task list: importing arlib would pull in torch and
# compete with the training jobs for the (small) GPU for no reason.
TASKS = ("copy", "prefix_xor", "sort", "swap_halves", "reverse")
PES = ("nope", "learned", "sin", "rope", "alibi")
COUPLED = {"swap_halves", "reverse"}

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
def _padded_runs(task="copy"):
    """Mean over seeds of the fixed-length protocol, keyed by encoding."""
    import glob
    acc = {}
    for p in glob.glob(f"../results/padded/{task}_*_s*.json"):
        r = json.load(open(p))
        acc.setdefault(r["pe"], []).append(r["curve"][-1])
    out = {}
    for pe, runs in acc.items():
        keys = [k for k in runs[0] if k.startswith("L")]
        out[pe] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        out[pe]["_sd"] = {k: float(np.std([r[k] for r in runs])) for k in keys}
    return out


def fig_padded():
    """Every position is seen in training; only the effective length varies."""
    runs = _padded_runs("copy")
    Ls = [8, 10, 12, 16]

    # sin and qk_abs are the same code at two different sites; if they coincide
    # to within noise they are drawn as one line, because there is one fact.
    same = ("sin" in runs and "qk_abs" in runs and
            all(abs(runs["sin"][k] - runs["qk_abs"][k]) < 0.02 for k in runs["sin"]
                if k.startswith("L")))

    style = [("nope", C_NOPE, "o", "NoPE"),
             ("alibi", C_ALIBI, "D", "ALiBi (relative, logits)"),
             ("rope", C_ROPE, "^", "RoPE (relative, $q/k$)")]
    if same:
        style.append(("sin", C_COLLAPSED, "s", "absolute (repr. or $q/k$)"))
    else:
        style += [("sin", C_SIN, "s", "sin (absolute, rep.)"),
                  ("qk_abs", C_LEARNED, "v", "absolute, $q/k$")]

    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    ax.axvspan(8, max(Ls) + 0.4, color="#f2f2f2", zorder=0)
    ax.axvline(8, color="#C0504D", lw=1.1, ls=(0, (4, 2)), zorder=1)

    for pe, col, mk, lab in style:
        if pe not in runs:
            continue
        ys = [runs[pe]["L%d" % L] for L in Ls]
        sd = [runs[pe]["_sd"]["L%d" % L] for L in Ls]
        ax.errorbar(Ls, ys, yerr=sd, color=col, marker=mk, ms=3.6, lw=1.5,
                    mfc="white", mew=0.9, capsize=1.6, elinewidth=0.6,
                    label=lab, zorder=3)

    ax.set_xlabel("effective length $L$")
    ax.set_ylabel("exact-match acc.")
    ax.set_xticks(Ls)
    ax.set_xlim(7.2, max(Ls) + 0.4)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(frameon=False, fontsize=6.6, loc="center right", handlelength=1.4,
              labelspacing=0.32)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig0_padded.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_injection():
    """Same site, different kind -- and the cell that cannot exist."""
    runs = _padded_runs("copy")
    sites = [("representation", "rep", "sin"),
             ("$q/k$", "qk", "qk_abs"),
             ("attn. logits", "logits", None)]
    rel_at = {"rep": None, "qk": "rope", "logits": "alibi"}

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    w = 0.34
    xs = np.arange(len(sites))

    for i, (lab, key, abs_pe) in enumerate(sites):
        if abs_pe and abs_pe in runs:
            v = runs[abs_pe]["L10"]
            ax.bar(i - w / 2, v, w, color="#C0504D", edgecolor="none")
            ax.text(i - w / 2, v + 0.035, f"{v:.2f}", ha="center", fontsize=6.2,
                    color="#7a2b28")
        else:
            # an additive bias that is constant along the key axis is removed by
            # the softmax, so no absolute code can act at this site
            ax.bar(i - w / 2, 0.015, w, facecolor="none", edgecolor="#9a9a9a",
                   hatch="////", lw=0.7)
            ax.text(i - w / 2, 0.10, "n/a", ha="center", fontsize=6.2,
                    color="#7a7a7a", style="italic")

        rel_pe = rel_at[key]
        if rel_pe and rel_pe in runs:
            v = runs[rel_pe]["L10"]
            ax.bar(i + w / 2, v, w, color="#2F6FA8", edgecolor="none")
            ax.text(i + w / 2, v + 0.035, f"{v:.2f}", ha="center", fontsize=6.2,
                    color="#1d4568")

    ax.set_xticks(xs, [s[0] for s in sites])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy at $L{=}10$")
    ax.set_xlabel("where the code enters")
    ax.legend(handles=[Patch(facecolor="#C0504D", label="absolute code"),
                       Patch(facecolor="#2F6FA8", label="relative code")],
              frameon=False, fontsize=6.6, loc="upper left", ncol=1,
              handlelength=1.1, labelspacing=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_site.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_multilen():
    """Failure tracks the set of trained lengths, not the largest position."""
    import glob
    runs = {}
    for p in glob.glob("../results/multilen/*.json"):
        r = json.load(open(p))
        name = "single" if len(r["lens"]) == 1 else "multi"
        runs.setdefault((r["pe"], name), []).append(r["curve"][-1])
    if not runs:
        return

    Ls = [4, 6, 8, 10, 12, 16]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    style = {("sin", "single"): (C_COLLAPSED, "s", "absolute, $L_{tr}{=}8$"),
             ("sin", "multi"): (C_COLLAPSED, "D", "absolute, $L_{tr}\\in\\{6,8,10\\}$"),
             ("nope", "single"): (C_NOPE, "o", "NoPE, $L_{tr}{=}8$"),
             ("nope", "multi"): (C_NOPE, "P", "NoPE, $L_{tr}\\in\\{6,8,10\\}$")}
    for key, (col, mk, lab) in style.items():
        if key not in runs:
            continue
        rs = runs[key]
        ys = [float(np.mean([r["L%d" % L] for r in rs])) for L in Ls]
        dashed = key[1] == "multi"
        ax.plot(Ls, ys, "-" if not dashed else "--", color=col, marker=mk, ms=3.4,
                lw=1.4, mfc="white", mew=0.9, label=lab, zorder=3)

    ax.axvspan(10, max(Ls) + 0.5, color="#f2f2f2", zorder=0)
    ax.set_xlabel("test length $L$")
    ax.set_ylabel("exact-match acc.")
    ax.set_xticks(Ls)
    ax.set_xlim(3.4, max(Ls) + 0.5)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(frameon=False, fontsize=6.2, loc="center left", handlelength=1.5,
              labelspacing=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_multilen.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _eval_runs(task="copy"):
    """Mean over seeds of the per-position analysis, keyed by encoding."""
    import glob
    acc = {}
    for p in glob.glob(f"../results/padded_eval/{task}_*_s*.json"):
        r = json.load(open(p))
        acc.setdefault(r["pe"], []).append(r)
    out = {}
    for pe, rs in acc.items():
        out[pe] = {}
        for L in rs[0]["L"]:
            out[pe][L] = {k: float(np.mean([r["L"][L][k] for r in rs]))
                          for k in ("seq", "tok", "tok_oracle", "stop")}
            out[pe][L]["per_pos"] = np.mean(
                [r["L"][L]["per_pos"] for r in rs], axis=0)
            out[pe][L]["eos"] = np.mean([r["L"][L]["eos"] for r in rs], axis=0)
    return out


def fig_profile():
    """(a) accuracy by output index at L=10; (b) emitted EOS; (c) the step moves."""
    ev = _eval_runs("copy")
    L = "10"
    n = 10
    xs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))

    # (a) all five codes at the same test length
    ax = axes[0]
    style = [("nope", C_NOPE, "o", "NoPE"), ("alibi", C_ALIBI, "D", "ALiBi"),
             ("sin", C_COLLAPSED, "s", "sin"), ("qk_abs", "#FF9500", "v", "qk-abs"),
             ("rope", C_ROPE, "^", "RoPE")]
    for pe, col, mk, lab in style:
        if pe not in ev:
            continue
        ax.plot(xs, ev[pe][L]["per_pos"], "-", color=col, marker=mk, ms=2.6,
                lw=1.2, mfc="white", mew=0.7, label=lab)
    ax.axvline(8.5, color="#cccccc", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("output position $k$")
    ax.set_ylabel("accuracy at $L{=}10$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([1, 5, 10])
    ax.legend(frameon=False, fontsize=5.6, loc="lower left", handlelength=1.2,
              labelspacing=0.25, ncol=2, columnspacing=0.8)

    # (b) the mechanism: EOS probability at the same positions
    ax = axes[1]
    for pe, col, mk, lab in [("sin", C_COLLAPSED, "s", "sin"),
                             ("nope", C_NOPE, "o", "NoPE"),
                             ("alibi", C_ALIBI, "D", "ALiBi")]:
        if pe not in ev:
            continue
        ax.plot(xs, ev[pe][L]["eos"], "-", color=col, marker=mk, ms=2.6, lw=1.2,
                mfc="white", mew=0.7, label=lab)
    ax.axvline(8.5, color="#cccccc", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("output position $k$")
    ax.set_ylabel("$P(\\mathrm{EOS})$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([1, 5, 10])
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.2,
              labelspacing=0.25)

    # (c) move the training length, move the step
    ax = axes[2]
    xs16 = np.arange(1, 17)
    if "sin" in ev:
        ax.plot(xs16, ev["sin"]["16"]["per_pos"], "-s", color=C_COLLAPSED, ms=2.6,
                lw=1.2, mfc="white", mew=0.7, label="sin, $L_{tr}{=}8$")
    rows = sorted(__import__("glob").glob("../results/rows/copy_sin_L12_s0.json"))
    if rows:
        r = json.load(open(rows[0]))
        ax.plot(xs16, r["L"]["16"]["per_pos"], ":^", color="#7a2b28", ms=2.6,
                lw=1.1, mfc="white", mew=0.7, label="sin, $L_{tr}{=}12$")
    ax.axvline(8.5, color="#cccccc", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("output position $k$")
    ax.set_ylabel("accuracy at $L{=}16$")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([1, 8, 16])
    ax.legend(frameon=False, fontsize=5.8, loc="lower left", handlelength=1.2,
              labelspacing=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_profile.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_decomp():
    """Exact match, token accuracy, and forced-length accuracy at L=10."""
    ev = _eval_runs("copy")
    codes = [("nope", "NoPE"), ("alibi", "ALiBi"), ("sin", "sin"),
             ("qk_abs", "qk-abs"), ("rope", "RoPE")]
    codes = [(c, lab) for c, lab in codes if c in ev]
    L = "10"

    fig, ax = plt.subplots(figsize=(4.0, 2.25))
    w = 0.26
    xs = np.arange(len(codes))
    cols = {"seq": "#c9c9c9", "tok": "#7fa8c9", "tok_oracle": "#2F6FA8"}
    labels = {"seq": "exact match", "tok": "token accuracy",
              "tok_oracle": "forced length"}

    for i, (k, off) in enumerate([("seq", -w), ("tok", 0.0), ("tok_oracle", w)]):
        vals = [ev[c][L][k] for c, _ in codes]
        ax.bar(xs + off, vals, w, color=cols[k], edgecolor="none", label=labels[k])
        for x, v in zip(xs + off, vals):
            if v < 0.005:      # a measured zero, not a missing measurement
                ax.plot([x - w / 2, x + w / 2], [0.008, 0.008], color="#333333",
                        lw=1.6, solid_capstyle="butt")
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=4.9,
                    va="bottom", color="#333333")

    ax.set_xticks(xs, [lab for _, lab in codes])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score at $L{=}10$")
    ax.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper center",
              handlelength=1.1, columnspacing=1.0, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_decomp.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_stopbound():
    """Where does generation stop, as a function of the test length?"""
    ev = _eval_runs("copy")
    Ls = [8, 10, 12]
    xs = np.array(Ls)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    style = {"nope": (C_NOPE, "o", "NoPE"), "alibi": (C_ALIBI, "D", "ALiBi"),
             "sin": (C_COLLAPSED, "s", "sin"), "qk_abs": ("#FF2C00", "v", "qk-abs"),
             "rope": (C_ROPE, "^", "RoPE")}
    ax.plot([8, 14], [8, 14], ":", color="#bbbbbb", lw=1.0, zorder=1)
    ax.text(12.5, 12.9, "correct stop", fontsize=6, color="#8a8a8a", rotation=42,
            ha="center", va="center")
    for pe, (col, mk, lab) in style.items():
        if pe not in ev:
            continue
        ys = [ev[pe][str(L)]["stop"] for L in Ls]
        ax.plot(Ls, ys, "-", color=col, marker=mk, ms=3.2, lw=1.3, mfc="white",
                mew=0.8, label=lab)
    ax.axhline(8, color="#cccccc", lw=0.6, ls=(0, (1, 2)))
    ax.text(8.05, 7.2, "$L_{tr}$", fontsize=6, color="#9a9a9a")
    ax.set_xlabel("test length $L$")
    ax.set_ylabel("mean stop position")
    ax.set_xticks(Ls)
    ax.set_xlim(7.6, 12.4)
    ax.set_ylim(0, 13.5)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", ncol=2,
              handlelength=1.3, columnspacing=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_stop.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_noeos():
    """Train without an EOS token: is the gap still there?"""
    import glob
    noe = {}
    for p in glob.glob("../results/noeos/*.json"):
        r = json.load(open(p))
        noe.setdefault(r["pe"], []).append(r)
    if not noe:
        return
    ev = _eval_runs("copy")
    Ls = [8, 10, 12, 16]
    fig, ax = plt.subplots(figsize=(3.6, 2.85))
    for pe, col, lab in [("nope", C_NOPE, "NoPE"), ("sin", C_COLLAPSED, "sin"),
                         ("rope", C_ROPE, "RoPE")]:
        if pe not in noe:
            continue
        ys = [float(np.mean([r["L"][str(L)]["seq"] for r in noe[pe]])) for L in Ls]
        ax.plot(Ls, ys, "-", color=col, marker="o", ms=3.0, lw=1.3, mfc="white",
                mew=0.7, label=f"{lab}, no EOS")
        if pe in ev:
            ys2 = [ev[pe][str(L)]["seq"] for L in Ls]
            ax.plot(Ls, ys2, "--", color=col, lw=0.9, alpha=0.5,
                    label=f"{lab}, with EOS")
    ax.set_xlabel("test length $L$")
    ax.set_ylabel("exact-match acc.")
    ax.set_xticks(Ls)
    ax.set_xlim(7.5, 16.5)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(frameon=False, fontsize=5.9, ncol=3, loc="upper center",
              handlelength=1.3, columnspacing=0.9, labelspacing=0.3,
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_noeos.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    df = load()
    summary = pd.read_csv("../results/summary.csv")
    summary["coupled"] = summary["task"].isin(COUPLED)

    # The paper uses three figures. The other functions below document analyses
    # from earlier drafts (the 5x5 standard-protocol grid, the length-coupling
    # dispersion, the cliff probe); they are kept for reproducibility of the
    # intermediate steps but are not regenerated by default, because two of them
    # illustrate claims the current draft withdraws.
    fig_profile()
    fig_decomp()
    fig_noeos()
    print("figures ->", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)):
        if f.endswith(".pdf"):
            print("  ", f)


if __name__ == "__main__":
    main()
