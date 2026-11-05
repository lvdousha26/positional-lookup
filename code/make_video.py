"""make_video.py -- a silent explainer video built from the paper's own data.

Frames are rendered with matplotlib and encoded with imageio/ffmpeg. Static
scenes are drawn once and held, so a 40-second video renders in a few seconds.

Output: ../video/lookup.mp4
"""
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt

FPS = 30
INK = "#1a1a1a"
MUTED = "#6b7683"
ACCENT = "#0C5DA5"
RED = "#C0504D"

SCENES = []


def scene(dur):
    def deco(fn):
        SCENES.append((fn, dur))
        return fn
    return deco


def clean(fig):
    fig.patch.set_facecolor("white")
    for s in fig.texts:
        s.set_color(INK)


def head(fig, text, y=0.86, size=15):
    fig.text(0.08, y, text, fontsize=size, color=MUTED, family="monospace")


@scene(3.2)
def s1_title(fig):
    fig.text(0.5, 0.58, "Explicit Positional Encodings Turn",
             ha="center", fontsize=30, weight="bold")
    fig.text(0.5, 0.49, "Length Generalisation into Lookup",
             ha="center", fontsize=30, weight="bold")
    fig.text(0.5, 0.36, "a diagnostic study, reproduced from one script",
             ha="center", fontsize=15, color=MUTED, style="italic")


@scene(4.0)
def s2_problem(fig):
    head(fig, "the phenomenon")
    fig.text(0.5, 0.68, "train on length 8", ha="center", fontsize=24)
    fig.text(0.5, 0.58, "\u2193", ha="center", fontsize=22, color=MUTED)
    fig.text(0.5, 0.48, "test on length 16", ha="center", fontsize=24)
    fig.text(0.5, 0.36, "accuracy: 0.00", ha="center", fontsize=24,
             color=RED, weight="bold")
    fig.text(0.5, 0.20, "even for copying", ha="center", fontsize=16,
             color=MUTED, style="italic")


@scene(4.5)
def s3_stories(fig):
    head(fig, "two explanations")
    fig.text(0.26, 0.62, "POSITION STORY", ha="center", fontsize=16,
             color=ACCENT, weight="bold")
    fig.text(0.26, 0.50, "the model meets absolute\npositions it never saw",
             ha="center", fontsize=18, linespacing=1.5)
    fig.text(0.74, 0.62, "LOOKUP STORY", ha="center", fontsize=16,
             color=RED, weight="bold")
    fig.text(0.74, 0.50, "the model learned a table\nindexed by length",
             ha="center", fontsize=18, linespacing=1.5)
    fig.text(0.5, 0.28, "both predict collapse. only one predicts WHERE.",
             ha="center", fontsize=16, color=MUTED, style="italic")


@scene(5.0)
def s4_confound(fig):
    head(fig, "why the usual protocol cannot tell them apart")
    fig.text(0.5, 0.70, "a sequence of L tokens occupies 2L + 1 positions",
             ha="center", fontsize=19)
    fig.text(0.30, 0.48, "L\u209c\u1d63 = 8", ha="center", fontsize=17, color=MUTED)
    fig.text(0.30, 0.40, "max position 16", ha="center", fontsize=19)
    fig.text(0.70, 0.48, "L = 10", ha="center", fontsize=17, color=MUTED)
    fig.text(0.70, 0.40, "max position 20", ha="center", fontsize=19)
    fig.text(0.5, 0.22, "position range and length run out at the same moment",
             ha="center", fontsize=17, color=RED)


@scene(5.0)
def s5_decouple(fig):
    head(fig, "the discriminating experiment")
    fig.text(0.5, 0.68, "fix the total to 41 tokens. pad every example.",
             ha="center", fontsize=19)
    fig.text(0.5, 0.56, "all 41 positions occur during training",
             ha="center", fontsize=19, color=ACCENT)
    fig.text(0.5, 0.44, "only the effective length varies",
             ha="center", fontsize=19, color=ACCENT)
    fig.text(0.5, 0.26, "if the position story is right, the failure disappears",
             ha="center", fontsize=16, color=MUTED, style="italic")


@scene(7.0)
def s6_result(fig):
    head(fig, "result: it does not disappear")
    runs = {}
    for p in glob.glob("../results/padded_copy_*.json"):
        r = json.load(open(p))
        runs[r["pe"]] = r["curve"][-1]
    Ls = [8, 10, 12, 16]
    ax = fig.add_axes([0.12, 0.16, 0.76, 0.58])
    ax.axvspan(8, 16.4, color="#f2f2f2", zorder=0)
    ax.axvline(8, color=RED, lw=1.4, ls=(0, (4, 2)), zorder=1)
    style = {"nope": ("#111111", "o", "NoPE"), "alibi": ("#845B97", "D", "ALiBi"),
             "sin": ("#00B945", "s", "sin"), "learned": ("#FF2C00", "^", "learned")}
    for pe, (c, m, lab) in style.items():
        if pe not in runs:
            continue
        ys = [runs[pe].get("L%d" % L, float("nan")) for L in Ls]
        ax.plot(Ls, ys, "-", color=c, marker=m, ms=7, lw=2.2, mfc="white",
                mew=1.2, label=lab)
    ax.set_xlabel("effective length L", fontsize=14)
    ax.set_ylabel("exact-match accuracy", fontsize=14)
    ax.set_xticks(Ls)
    ax.set_ylim(-0.05, 1.08)
    ax.tick_params(labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="center right")
    ax.set_title("every position seen in training; only the length is new",
                 fontsize=13, color=MUTED, pad=10)


@scene(4.0)
def s7_conclusion(fig):
    head(fig, "what follows")
    fig.text(0.5, 0.66, "positions are not the variable",
             ha="center", fontsize=26, weight="bold")
    fig.text(0.5, 0.54, "the collapse is indexed by length",
             ha="center", fontsize=26, weight="bold", color=ACCENT)
    fig.text(0.5, 0.36, "a more extrapolable absolute code cannot fix this",
             ha="center", fontsize=16, color=MUTED, style="italic")
    fig.text(0.5, 0.16, "code, proofs and results: github.com/lvdousha26/positional-lookup",
             ha="center", fontsize=12, color=MUTED, family="monospace")


def main():
    os.makedirs("../video", exist_ok=True)
    frames = []
    for fn, dur in SCENES:
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        clean(fig)
        fn(fig)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.extend([buf] * int(dur * FPS))
        plt.close(fig)
        print(f"  {fn.__name__}: {int(dur*FPS)} frames")

    out = "../video/lookup.mp4"
    imageio.mimsave(out, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"\n{out}  ({len(frames)/FPS:.1f}s, {len(frames)} frames)")


if __name__ == "__main__":
    main()
