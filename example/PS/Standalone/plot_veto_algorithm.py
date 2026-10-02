#!/usr/bin/env python3
"""The four steps of the Sudakov veto algorithm, drawn with the numbers of our event.

One panel per step of the derivation in the book (chapter 2, "From a form
factor to a random number"), all for the mother's first proposal in round 1
of the seed-1 shower, i.e. the b quark radiating against the incoming u:

  step 1  overestimate the kernel     P_qq(z) = C_F (1+z^2)/(1-z) under 2 C_F/(1-z)
  step 2  overestimate the coupling   alpha_s at second order under the first-order form
  step 3  invert                      Delta_over(pT_start, pT) = R, three draws in a row,
                                      each continued from the rejected scale
  step 4  choose z and correct        the four factors of the acceptance weight against R'

The coupling and Lambda_nf are taken from the shower script itself
(standalone_parton_showering.AlphaStrong), the random numbers R and the
kinematic numbers from its -vv log for seed 1, so that every line of the figure
can be found in that log.

    python3 plot_veto_algorithm.py                       # -> veto_algorithm.pdf
    python3 plot_veto_algorithm.py -o veto.png --dpi 200

Needs numpy and matplotlib (on this Mac: /usr/bin/python3).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import standalone_parton_showering as core          # noqa: E402

CF = 4.0 / 3.0
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
COL_B = "#ff0000"           # the b quark's family colour, as in plot_journey_PS.py
COL_OVER = "#52514e"        # overestimates: neutral, dashed
COL_REJ = "#f3c6c6"         # the rejected part of the overestimate
COL_BBAR = "#0000ff"

# --- numbers of the mother's round-1 proposal (seed 1, -vv log) --------------------------
PTMAX = 113.3227                      # starting scale of the shower
C_OVER = 23.2841                      # 2 C_F ln((1-zmin)/(1-zmax)) for the nf = 5 segment
B0_5 = 23.0 / 6.0                     # alpha_s^(1) = 2 pi / (b0 ln(pT^2/Lambda^2))
DRAWS = [                              # (R, pT it gives, why it ended)
    (0.51607, "rejected: the incoming $u$ would need $x' = 2.65 > 1$"),
    (0.77668, "rejected: $x' = 19$"),
    (0.03959, "kept: on to step 4"),
]
Z_ACCEPTED = 0.9819
WEIGHTS = [                            # the four factors of the acceptance weight, in the log's order
    (0.9818, r"$(1+z^2)/2$ (dead cone $-0.0003$)"),
    (0.8076, r"$\alpha_s^{(2)}/\alpha_s^{(1)} = 0.2106/0.2608$"),
    (0.2747, r"$w_{\rm damp} = p_{T,\rm rad}\,p_T/(p_{T,\rm rad}\,p_T + m_a^2)$"),
    (0.9810, r"$xf_u(0.3944)/xf_u(0.3901)$"),
]
R_PRIME = 0.0623
BBAR_TRIALS = [59.22, 22.41, 14.56, 7.26, 2.73, 3.47, 2.36, 2.16]   # the bbar's eight draws of the same round
Z_EVENT = [(0.6013, "branching 5\n$z = 0.601$"), (Z_ACCEPTED, "round-1 proposal\n$z = 0.982$")]


def style(ax, fontsize=8.5):
    ax.tick_params(colors=TEXT2, labelsize=fontsize)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=10.5, color=TEXT, weight="bold", pad=8)


def delta_over(pt_start, pt, lam2, b0, c):
    """Overestimated no-emission probability between pt_start and pt (first-order coupling, constant kernel)."""
    return (np.log(pt ** 2 / lam2) / math.log(pt_start ** 2 / lam2)) ** (c / b0)


def invert(pt_start, R, lam2, b0, c):
    """Eq. (veto_solution): the scale at which Delta_over(pt_start, pt) = R."""
    return math.sqrt(lam2 * (pt_start ** 2 / lam2) ** (R ** (b0 / c)))


def panel_kernel(ax):
    z = np.linspace(0.01, 0.9995, 2000)
    p_true = CF * (1 + z ** 2) / (1 - z)
    p_over = 2 * CF / (1 - z)
    ax.fill_between(z, p_true, p_over, color=COL_REJ, lw=0, label="rejected in step 4")
    ax.plot(z, p_over, color=COL_OVER, lw=1.6, ls="--", label=r"overestimate $2C_F/(1-z)$")
    ax.plot(z, p_true, color=COL_B, lw=1.8, label=r"$P_{qq}(z) = C_F\,(1+z^2)/(1-z)$")
    for zz, lab in Z_EVENT:
        pt, po = CF * (1 + zz ** 2) / (1 - zz), 2 * CF / (1 - zz)
        ax.plot([zz, zz], [pt, po], color=TEXT, lw=0.8)
        ax.plot(zz, pt, "o", color=COL_B, ms=5, mec=TEXT, mew=0.6)
        ax.annotate(lab + f"\nratio {pt / po:.3f}", (zz, pt), xytext=(-8, -2), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=TEXT)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 3000)
    ax.set_xlabel("energy fraction $z$ kept by the quark", color=TEXT, fontsize=9)
    ax.set_ylabel("kernel (a rate, not a probability)", color=TEXT, fontsize=9)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    title(ax, "step 1: overestimate the kernel")
    ax.text(0.02, 0.60, "the ratio $(1+z^2)/2 \\leq 1$ is the\nfirst factor of the weight in step 4",
            transform=ax.transAxes, fontsize=7.5, color=TEXT2, va="top")


def panel_coupling(ax, alpha):
    pts = np.logspace(math.log10(0.611), math.log10(PTMAX), 600)
    a2 = np.array([alpha.alphaS(p * p) for p in pts])
    a1 = np.array([alpha.alphaS_1loop(p * p, alpha.nf(p * p)) for p in pts])
    ax.fill_between(pts, a2, a1, where=a1 > a2, color=COL_REJ, lw=0, label="rejected in step 4")
    ax.plot(pts, a1, color=COL_OVER, lw=1.6, ls="--", label=r"first order, same $\Lambda_{n_f}$ (overestimate)")
    ax.plot(pts, a2, color=COL_B, lw=1.8, label=r"$\alpha_s$ at second order (what the shower uses)")
    ax.plot(pts, a2 / a1, color=TEXT, lw=1.0, ls=":", label=r"ratio $\alpha_s^{(2)}/\alpha_s^{(1)}$")
    for m, lab in ((1.5, "$m_c$"), (4.8, "$m_b$")):
        ax.axvline(m, color=GRID, lw=0.8)
        ax.text(m, 2.6, lab, ha="center", va="bottom", fontsize=8, color=TEXT2)
    ax.axvline(0.611, color=TEXT2, lw=0.6, ls=":")
    ax.text(0.63, 2.6, "cut-off", fontsize=7.5, color=TEXT2, va="bottom")
    for p in (34.2, 5.24, 1.0):
        r = alpha.alphaS(p * p) / alpha.alphaS_1loop(p * p, alpha.nf(p * p))
        ax.plot(p, r, "o", color=TEXT, ms=4)
        ax.annotate(f"{r:.2f}", (p, r), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, color=TEXT)
    # the crossing in the last octave
    i = np.argmax(a2 > a1)
    ax.annotate("the two curves cross just\nabove the cut-off: here the\n'overestimate' is not one", (pts[i], a2[i]),
                xytext=(7.0, 0.42), fontsize=7.5, color=TEXT2, arrowprops=dict(arrowstyle="-|>", color=TEXT2, lw=0.7))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 130)
    ax.set_ylim(0.06, 3.2)
    ax.set_xlabel("$p_T$ [GeV]", color=TEXT, fontsize=9)
    ax.set_ylabel(r"$\alpha_s(p_T^2)$ and its overestimate", color=TEXT, fontsize=9)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    title(ax, "step 2: overestimate the coupling")


def panel_invert(ax, alpha):
    lam2 = alpha.lambda2(5)
    pts = np.logspace(math.log10(2.0), math.log10(PTMAX), 800)
    start = PTMAX
    styles = ["-", "--", ":"]
    chain = []
    for n, ((R, why), ls) in enumerate(zip(DRAWS, styles)):
        mask = pts <= start
        ax.plot(pts[mask], delta_over(start, pts[mask], lam2, B0_5, C_OVER), color=COL_B, lw=1.8, ls=ls,
                label=rf"draw {n + 1}, from {start:.1f} GeV")
        pt_new = invert(start, R, lam2, B0_5, C_OVER)
        chain.append(pt_new)
        ax.plot([2.0, pt_new], [R, R], color=TEXT2, lw=0.7, ls="-")
        ax.plot([pt_new, pt_new], [0, R], color=TEXT2, lw=0.7, ls="-")
        ax.plot(pt_new, R, "o", color=COL_B, ms=5, mec=TEXT, mew=0.6)
        rx, ry, rha = ((6, -11, "left"), (6, 3, "left"), (-6, 4, "right"))[n]
        ax.annotate(f"$R = {R:.5f}$", (pt_new, R), xytext=(rx, ry), textcoords="offset points", ha=rha,
                    fontsize=7.5, color=TEXT)
        ha = ("left", "right", "center")[n]
        dx = (4, -4, 0)[n]
        dy = (4, 4, 30)[n]
        ax.annotate(f"{pt_new:.2f} GeV\n{why}", (pt_new, 0.0), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=7.3, color=TEXT)
        start = pt_new
    ax.plot(BBAR_TRIALS, [0.015] * len(BBAR_TRIALS), "v", color=COL_BBAR, ms=4, alpha=0.8, clip_on=False, ls="none",
            label=r"the $\bar b$'s 8 draws, same round")
    ax.set_xscale("log")
    ax.set_xlim(2.0, PTMAX * 1.05)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("$p_T$ [GeV]  (the evolution runs from right to left)", color=TEXT, fontsize=9)
    ax.set_ylabel(r"$\Delta_{\rm over}(p_{T,\rm start}, p_T)$", color=TEXT, fontsize=9)
    leg = ax.legend(loc="upper left", fontsize=7.3, frameon=False,
                    title="$c = 23.28,\\ b_0 = 23/6,\\ \\Lambda_5 = 0.226$ GeV", title_fontsize=7.5)
    leg._legend_box.align = "left"
    title(ax, "step 3: invert, and continue from the rejected scale")
    return chain


def panel_weight(ax):
    w = 1.0
    ys = []
    for n, (f, lab) in enumerate(WEIGHTS):
        y = len(WEIGHTS) - n
        ax.barh(y, w, color=COL_REJ, height=0.55, lw=0)
        ax.barh(y, w * f, color=COL_B, height=0.55, lw=0, alpha=0.85)
        ax.text(w * f + 0.012, y, rf"$\times\,{f:.4f}$  {lab}", va="center", fontsize=7.8, color=TEXT)
        ax.text(-0.01, y, f"{w * f:.4f}", va="center", ha="right", fontsize=7.8, color=TEXT)
        w *= f
        ys.append(y)
    ax.barh(0, w, color=COL_B, height=0.55, lw=0)
    ax.text(-0.01, 0, f"$w = {w:.4f}$", va="center", ha="right", fontsize=8.5, color=TEXT, weight="bold")
    ax.axvline(R_PRIME, color=TEXT, lw=1.0, ls="--", ymin=0.02, ymax=0.2)
    ax.annotate(rf"$R' = {R_PRIME}$ was drawn: $R' \leq w$, accept" "\n"
                rf"the proposal $p_T = {5.24}$ GeV, $z = {Z_ACCEPTED}$",
                (R_PRIME, 0), xytext=(0.30, -0.05), fontsize=7.8, color=TEXT, va="center",
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=0.7))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(WEIGHTS) + 0.7)
    ax.set_yticks([])
    ax.set_xlabel("acceptance weight after each factor (light red: the part thrown away)", color=TEXT, fontsize=9)
    ax.text(0.0, len(WEIGHTS) + 0.55, "start: 1.0000", fontsize=7.8, color=TEXT2, va="center")
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    title(ax, "step 4: choose $z$ and correct")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "veto_algorithm.pdf")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--alphas", type=float, default=0.118)
    args = p.parse_args()

    alpha = core.AlphaStrong(args.alphas, order=2)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    panel_kernel(axes[0, 0])
    panel_coupling(axes[0, 1], alpha)
    chain = panel_invert(axes[1, 0], alpha)
    panel_weight(axes[1, 1])
    for ax in axes.flat:
        style(ax)
    axes[1, 1].grid(False)
    fig.suptitle("The Sudakov veto algorithm step by step: the mother's first proposal in round 1 "
                 f"($b$ against the incoming $u$, from $p_T^{{\\max}} = {PTMAX:.1f}$ GeV)",
                 fontsize=11, color=TEXT, x=0.02, ha="left", y=0.985)
    fig.text(0.98, 0.008, "numbers from standalone_parton_showering.py -vv, seed 1", fontsize=7.5, color=TEXT2,
             ha="right", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    print(f"Lambda_5 = {math.sqrt(alpha.lambda2(5)):.4f} GeV; the three draws give pT = "
          + ", ".join(f"{c:.2f}" for c in chain) + " GeV (log: 59.67, 47.54, 5.24)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
