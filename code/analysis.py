"""analysis.py -- aggregate the grid, test the interaction, build paper figures.

Core quantity: PE effects are only interesting *conditionally*. We therefore
report extrapolation accuracy per (task, PE) and ask whether the ranking of PEs
reverses between length-decoupled and length-coupled tasks.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from arlib import COUPLED, PES, TASKS

LENGTHS = (8, 10, 12, 16, 20)


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        r = json.load(open(p))
        final = r["curve"][-1]
        row = {"task": r["task"], "pe": r["pe"], "seed": r["seed"], "sec": r["sec"]}
        for L in LENGTHS:
            row[f"L{L}"] = final.get(f"L{L}", np.nan)
        rows.append(row)
    if not rows:
        raise SystemExit(f"no runs matched {pattern}")
    df = pd.DataFrame(rows)
    df["ES"] = df[[f"L{L}" for L in LENGTHS[1:]]].mean(axis=1)
    df["coupled"] = df["task"].isin(COUPLED)
    df["solved"] = df["L8"] >= 0.95
    return df


def bootstrap_ci(x, n=5000, alpha=0.05, rng=None):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return tuple(np.quantile(bs, [alpha / 2, 1 - alpha / 2]))


def summarise(df):
    g = df.groupby(["task", "pe"])["ES"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["lo"], out["hi"] = zip(*[
        bootstrap_ci(df[(df.task == t) & (df.pe == p)]["ES"]) for t, p in zip(out.task, out.pe)
    ])
    out["coupled"] = out["task"].isin(COUPLED)
    return out


def rank_table(df):
    """Mean rank of each PE within each task (1 = best extrapolation)."""
    piv = df.groupby(["task", "seed", "pe"])["ES"].mean().reset_index()
    piv["rank"] = piv.groupby(["task", "seed"])["ES"].rank(ascending=False, method="average")
    rt = piv.groupby(["task", "pe"])["rank"].mean().unstack()
    return rt.reindex(list(TASKS))[[p for p in PES if p in rt.columns]]


def paired_rank_test(df):
    """Compare NoPE rank on decoupled vs coupled tasks (Mann-Whitney U)."""
    from scipy.stats import mannwhitneyu

    piv = df.groupby(["task", "seed", "pe"])["ES"].mean().reset_index()
    piv["rank"] = piv.groupby(["task", "seed"])["ES"].rank(ascending=False, method="average")
    nope = piv[piv.pe == "nope"].copy()
    a = nope[~nope.task.isin(COUPLED)]["rank"].values
    b = nope[nope.task.isin(COUPLED)]["rank"].values
    if len(a) < 2 or len(b) < 2:
        return None
    stat, p = mannwhitneyu(a, b, alternative="less")
    return {"nope_rank_decoupled": float(np.mean(a)),
            "nope_rank_coupled": float(np.mean(b)), "U": float(stat), "p": float(p)}


def make_figures(summary, df, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    plt.rcParams["font.family"] = "DejaVu Sans"

    # ---- Fig 1: task x PE heatmap of extrapolation score ----
    piv = summary.pivot(index="task", columns="pe", values="mean").reindex(list(TASKS))
    piv = piv[[p for p in PES if p in piv.columns]]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=max(0.6, np.nanmax(piv.values)))
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]),
                  [f"{t} *" if t in COUPLED else t for t in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.55 * np.nanmax(piv.values) else "black")
    ax.axhline(2.5, color="w", lw=1.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("extrapolation acc. (L=10..20)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Extrapolation score by task and encoding", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig1_heatmap.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Fig 2: accuracy vs length, 2x3 grid ----
    n = len(TASKS)
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 4.0), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    colors = {"nope": "#111111", "learned": "#d62728", "sin": "#1f77b4",
              "rope": "#2ca02c", "alibi": "#ff7f0e"}
    for ax, task in zip(axes, TASKS):
        sub = df[df.task == task]
        for pe in [p for p in PES if p in sub.pe.unique()]:
            s = sub[sub.pe == pe]
            m = s[[f"L{L}" for L in LENGTHS]].mean()
            sd = s[[f"L{L}" for L in LENGTHS]].std().fillna(0)
            ax.errorbar(LENGTHS, m.values, yerr=sd.values, marker="o", ms=2.6, lw=1.2,
                        capsize=1.6, elinewidth=0.6, color=colors.get(pe, None), label=pe)
        ax.axvline(8, ls=":", c="gray", lw=0.9)
        ax.set_title(("* " if task in COUPLED else "") + task, fontsize=8)
        ax.set_xlabel("test length", fontsize=7.5)
        ax.set_xticks([8, 12, 16, 20])
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("exact-match acc.", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[n].legend(handles, labels, fontsize=7.5, frameon=False, loc="center")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig2_curves.{ext}"), dpi=300)
    plt.close(fig)

    # ---- Fig 3: PE spread per task, split by coupling ----
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    spread = summary.groupby("task")["mean"].agg(lambda s: s.max() - s.min()).reindex(list(TASKS))
    cols = ["#d62728" if t in COUPLED else "#4c72b0" for t in spread.index]
    ax.bar(range(len(spread)), spread.values, color=cols, width=0.62)
    ax.set_xticks(range(len(spread)), spread.index, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("PE spread (max$-$min acc.)", fontsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c72b0", label="L-decoupled"),
                       Patch(color="#d62728", label="L-coupled")],
              fontsize=6, frameon=False)
    ax.set_title("Heterogeneity of the PE effect", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"fig3_spread.{ext}"), dpi=300)
    plt.close(fig)


def two_way_anova(df, y="ES", a="task", b="pe"):
    """Two-way ANOVA via nested least squares; the interaction term is the target.

    The interaction SS is order-free (additive model vs. full model), so it does
    not depend on how the two factors are entered. Main effects are reported
    sequentially (type I) and flagged as such.
    """
    from scipy.stats import f as fdist

    d = df.dropna(subset=[y])
    n = len(d)
    if n < 10:
        return None
    Y = d[y].to_numpy(float)
    A = pd.get_dummies(d[a], drop_first=True).to_numpy(float)
    B = pd.get_dummies(d[b], drop_first=True).to_numpy(float)
    AB = (A[:, :, None] * B[:, None, :]).reshape(n, -1)
    one = np.ones((n, 1))

    def rss(X):
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        r = Y - X @ beta
        return float(r @ r), X.shape[1]

    X0 = one
    XA = np.hstack([one, A])
    rss0, k0 = rss(X0)
    rssA, kA = rss(XA)
    Xadd = np.hstack([one, A, B])
    rssAdd, kAdd = rss(Xadd)
    Xfull = np.hstack([one, A, B, AB])
    rssFull, kFull = rss(Xfull)

    dfe = n - kFull
    if dfe <= 0:
        return None
    mse = rssFull / dfe
    ssE = rssFull

    def row(name, ss, dfn):
        if dfn <= 0:
            return {"term": name, "SS": np.nan, "df": dfn, "F": np.nan, "p": np.nan,
                    "eta2p": np.nan}
        Fv = (ss / dfn) / mse
        return {"term": name, "SS": ss, "df": dfn, "F": Fv,
                "p": float(fdist.sf(Fv, dfn, dfe)),
                "eta2p": ss / (ss + ssE)}

    return [
        row("task (type I)", rss0 - rssA, kA - k0),
        row("pe (type I | task)", rssA - rssAdd, kAdd - kA),
        row("task x pe (interaction)", rssAdd - rssFull, kFull - kAdd),
        {"term": "residual", "SS": ssE, "df": dfe, "F": np.nan, "p": np.nan, "eta2p": np.nan},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="../results/grid/*.json")
    ap.add_argument("--figdir", default="../figures")
    ap.add_argument("--outdir", default="../results")
    a = ap.parse_args()

    os.makedirs(a.figdir, exist_ok=True)
    df = load(a.glob)
    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(os.path.join(a.outdir, "runs.csv"), index=False)

    print(f"\n{len(df)} runs, {df.task.nunique()} tasks x {df.pe.nunique()} PEs "
          f"x {df.seed.nunique()} seeds\n")

    summary = summarise(df)
    summary.to_csv(os.path.join(a.outdir, "summary.csv"), index=False)
    piv = summary.pivot(index="task", columns="pe", values="mean").reindex(list(TASKS))
    piv = piv[[p for p in PES if p in piv.columns]]
    print("Extrapolation score (mean acc over L=10,12,16,20):")
    print(piv.round(3).to_string())
    print("\nIn-distribution accuracy at L=8:")
    print(df.pivot_table(index="task", columns="pe", values="L8", aggfunc="mean")
          .reindex(list(TASKS))[[p for p in PES if p in df.pe.unique()]].round(3).to_string())

    print("\nMean PE rank within task (1=best):")
    rt = rank_table(df)
    print(rt.round(2).to_string())

    spread = summary.groupby("task")["mean"].agg(lambda s: s.max() - s.min()).reindex(list(TASKS))
    print("\nPE spread per task (max-min):")
    print(spread.round(3).to_string())
    print(f"  decoupled mean spread: {spread[[t for t in TASKS if t not in COUPLED]].mean():.3f}")
    print(f"  coupled   mean spread: {spread[[t for t in TASKS if t in COUPLED]].mean():.3f}")

    # two-way interaction on extrapolation score (hand-rolled, no statsmodels)
    res = two_way_anova(df[df.solved])
    if res is not None:
        print("\nTwo-way ANOVA on ES (solved runs only):")
        print(pd.DataFrame(res).round(4).to_string(index=False))
        pd.DataFrame(res).to_csv(os.path.join(a.outdir, "anova.csv"), index=False)

    pt = paired_rank_test(df)
    if pt:
        print(f"\nNoPE rank: decoupled={pt['nope_rank_decoupled']:.2f} "
              f"coupled={pt['nope_rank_coupled']:.2f}  (U={pt['U']:.1f}, p={pt['p']:.4f})")
        json.dump(pt, open(os.path.join(a.outdir, "nope_rank_test.json"), "w"), indent=1)

    make_figures(summary, df, a.figdir)
    print(f"\nfigures -> {a.figdir}")


if __name__ == "__main__":
    main()
