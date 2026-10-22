#!/usr/bin/env python3
"""The same 749 particles under three algorithms: anti-kT, Cambridge/Aachen and HOTVR, side by side.

The three toy scripts antiKt.py, caKt.py and hotVR.py follow nine particles of our event by hand; this
script runs the same three rules on the whole final state of chapter 6 (../../Decay/event_Decay.lhe, the
749 rows with a positive status) and draws the result in the (eta, phi) frame of the journey figures,
one panel per algorithm:

  (a) anti-kT, R = 0.4         the ak4GenJets of the record (standalone_genjets.py's Clustering, p = -1)
  (b) Cambridge/Aachen, R = 0.4   the same class with p = 0: angle-ordered, the boundaries follow the soft particles
  (c) HOTVR                    hotVR.py's decision sequence (Cambridge/Aachen ordering, d_iB = R_eff^2 with
                               R_eff = clip(rho/pT, R_min, R_max), the mass jump above mu, the pT_sub grooming)
                               with the paper's top-tagging defaults, re-implemented with numpy for 749
                               particles and checked against hotVR.py on its nine

Every jet above 3 GeV is a circle of its radius (R, or R_eff for HOTVR) around its axis; the b jet is red,
the bbar jet blue (the jet nearest to each last b-quark copy within its radius), the next six jets in colour,
the rest grey, particles in no jet above 3 GeV hollow, and the particles HOTVR groomed away grey crosses;
HOTVR subjet axes are hollow diamonds. The printout lists the leading jets of each algorithm.

    python3 plot_algorithms_GenJet.py                         # -> algorithms_genjets.pdf
    python3 plot_algorithms_GenJet.py --png --rho 40 --mu 10 --ptsub 5 -o algorithms_genjets_scaled.pdf

Needs numpy and matplotlib (on this Mac: /usr/bin/python3). About ten seconds (the HOTVR step loop).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "PS"))
sys.path.insert(0, str(HERE.parent.parent / "Had"))
import standalone_genjets as core  # noqa: E402
import hotVR  # noqa: E402
import plot_journey_PS as ps  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

COL_B, COL_BBAR, COL_OTHER, COL_UNCL, COL_GROOM = "#ff0000", "#0000ff", "#8a8a8a", "#b4b4b4", "#6a6a6a"
PALETTE = ["#00999a", "#9b59d6", "#3b9ae1", "#ff1493", "#2e9e2e", "#a3a300"]
TEXT, TEXT2 = ps.TEXT, ps.TEXT2
PT_MIN = core.JET_PT_MIN


# ---------------------------------------------------------------------------------------------
# HOTVR on many particles: hotVR.py's decisions with numpy arrays
# ---------------------------------------------------------------------------------------------
class HOTVR:
    """Sequential recombination with the HOTVR rule of hotVR.py (Lapsien, Kogler, Haller, Section 2) on a list of
    four-vectors (rows E, px, py, pz). Indices as in standalone_genjets.Clustering: the originals are 0..N-1,
    a merge makes N, N+1, ...; a discarded branch keeps its index and is listed in `discarded`. The history
    holds (action, i, j, k) with the actions of hotVR.py: merge_below_mu, merge_mass_jump, discard_mass_jump,
    discard_soft, finalize_jet."""

    def __init__(self, p4, config: hotVR.HOTVRConfig):
        p4 = np.asarray(p4, dtype=float)
        n = len(p4)
        self.n0, self.cfg = n, config
        cap = 2 * n + 2
        self.p4 = np.zeros((cap, 4))
        self.p4[:n] = p4
        self.pt = np.zeros(cap)
        self.y = np.zeros(cap)
        self.phi = np.zeros(cap)
        self.m = np.zeros(cap)
        for i in range(n):
            self._kin(i)
        self.alive = np.zeros(cap, dtype=bool)
        self.alive[:n] = True
        self.is_jet = np.zeros(cap, dtype=bool)
        self.parents, self.subjets, self.const = {}, {}, {}
        self.discarded = []            # (index, reason)
        self.history = []
        self.size = n
        self._run()

    def _kin(self, i):
        p = self.p4[i]
        self.pt[i] = math.hypot(p[1], p[2])
        self.y[i] = core.rapidity(p)
        self.phi[i] = math.atan2(p[2], p[1])
        self.m[i] = math.sqrt(max(core.mass2(p), 0.0))

    def r_eff(self, pt):
        return np.clip(self.cfg.rho / np.maximum(pt, 1e-300), self.cfg.r_min, self.cfg.r_max)

    def _subjets_of(self, i):
        return self.subjets[i] if i in self.subjets else [i]

    def _run(self):
        cfg = self.cfg
        while self.alive.any():
            idx = np.flatnonzero(self.alive)
            diB = self.r_eff(self.pt[idx]) ** 2
            y, ph = self.y[idx], self.phi[idx]
            dy = y[:, None] - y[None, :]
            dp = np.abs(ph[:, None] - ph[None, :])
            dp = np.where(dp > math.pi, 2 * math.pi - dp, dp)
            d = dy * dy + dp * dp
            iu = np.triu_indices(len(idx), k=1)
            a = int(np.argmin(diB))
            if iu[0].size == 0 or diB[a] <= d[iu].min():       # a tie goes to the beam, as in hotVR.py
                i = int(idx[a])
                self.alive[i] = False
                self.is_jet[i] = True
                self.history.append(("finalize_jet", i, -1, i))
                continue
            q = int(np.argmin(d[iu]))
            i, j = int(idx[iu[0][q]]), int(idx[iu[1][q]])
            p = self.p4[i] + self.p4[j]
            m_ij = math.sqrt(max(core.mass2(p), 0.0))
            if m_ij <= cfg.mu + hotVR.EPS:
                self._merge(i, j, p, "merge_below_mu", keep=(i in self.subjets or j in self.subjets))
                continue
            lhs, rhs = cfg.theta * m_ij, max(self.m[i], self.m[j])
            if not lhs > rhs + hotVR.EPS:
                loser = min((i, j), key=lambda x: (self.m[x], self.pt[x], -x))
                self.alive[loser] = False
                self.discarded.append((loser, "mass-jump condition failed; lower-mass branch removed"))
                self.history.append(("discard_mass_jump", i, j, loser))
                continue
            soft = [x for x in (i, j) if self.pt[x] <= cfg.pt_sub + hotVR.EPS]
            if soft:
                for x in soft:
                    self.alive[x] = False
                    self.discarded.append((x, "mass jump passed, but branch failed pT,sub"))
                self.history.append(("discard_soft", i, j, soft[0]))
                continue
            self._merge(i, j, p, "merge_mass_jump", keep=True)

    def _merge(self, i, j, p, action, keep):
        k = self.size
        self.size += 1
        self.p4[k] = p
        self._kin(k)
        self.parents[k] = (i, j)
        if keep:
            self.subjets[k] = self._subjets_of(i) + self._subjets_of(j)
        self.alive[i] = self.alive[j] = False
        self.alive[k] = True
        self.history.append((action, i, j, k))

    def constituents(self, i):
        if i < self.n0:
            return [i]
        if i not in self.const:
            a, b = self.parents[i]
            self.const[i] = self.constituents(a) + self.constituents(b)
        return self.const[i]

    def jets(self, ptmin):
        js = [i for i in range(self.size) if self.is_jet[i] and self.pt[i] >= ptmin]
        return sorted(js, key=lambda i: -self.pt[i])

    def kin(self, i):
        p = self.p4[i]
        return dict(pt=float(self.pt[i]), y=float(self.y[i]), eta=core.eta(p), phi=float(self.phi[i]), m=float(self.m[i]))


def check_against_toy(config: hotVR.HOTVRConfig):
    """The nine particles of hotVR.py must give the same decisions here."""
    toy = hotVR.build_toy_events()["Our event"]
    ref = hotVR.hotvr_cluster(toy, config)
    p4 = [[p.p4.energy, p.p4.px, p.p4.py, p.p4.pz] for p in toy]
    mine = HOTVR(p4, config)
    a_ref = [s.action for s in ref.history]
    a_mine = [h[0] for h in mine.history]
    if a_ref != a_mine:
        raise AssertionError(f"HOTVR on the nine toy particles differs from hotVR.py: {a_ref} vs {a_mine}")
    j_ref = sorted(sorted(j.leaves) for j in ref.jets)
    j_mine = sorted(sorted(toy[i].label for i in mine.constituents(j)) for j in mine.jets(0.0))
    if j_ref != j_mine:
        raise AssertionError(f"HOTVR jets differ from hotVR.py: {j_ref} vs {j_mine}")
    return len(a_ref)


# ---------------------------------------------------------------------------------------------
# the three clusterings of the event
# ---------------------------------------------------------------------------------------------
def cluster_all(ev, config: hotVR.HOTVRConfig, R=core.R_AK4):
    rows = core.input_rows(ev, nonu=False)
    p4 = np.array([ev.p[r].p for r in rows], dtype=float)
    out = {}
    for key, title, power in (("antikt", f"(a) anti-$k_T$, R = {R:g} (ak4GenJets)", -1), ("ca", f"(b) Cambridge/Aachen, R = {R:g}", 0)):
        cl = core.Clustering(p4, R, power)
        out[key] = dict(title=title, cl=cl, rows=rows, radius=lambda pt, R=R: R, discarded=[], subjets={})
    hv = HOTVR(p4, config)
    out["hotvr"] = dict(title=f"(c) HOTVR, $R_{{\\rm eff}}$ = clip({config.rho:g} GeV / $p_T$, {config.r_min:g}, {config.r_max:g})",
                        params=f"$\\mu$ = {config.mu:g} GeV, $\\theta$ = {config.theta:g}, $p_{{T,\\rm sub}}$ = {config.pt_sub:g} GeV",
                        cl=hv, rows=rows, radius=lambda pt: float(hv.r_eff(np.array([pt]))[0]),
                        discarded=[rows[i] for i, _ in hv.discarded if i < hv.n0] + [rows[c] for i, _ in hv.discarded if i >= hv.n0 for c in hv.constituents(i)],
                        subjets=hv.subjets)
    for key, c in out.items():
        cl = c["cl"]
        jets = []
        for rank, j in enumerate(cl.jets(PT_MIN)):
            k = cl.kin(j)
            const = [rows[i] for i in cl.constituents(j)]
            subs = [cl.kin(s) for s in c["subjets"].get(j, [])] if c["subjets"] else []
            jets.append(dict(rank=rank, index=j, const=const, n=len(const), R=c["radius"](k["pt"]), subjets=subs, **k))
        c["jets"] = jets
        c["jet_of"] = {r: j["rank"] for j in jets for r in j["const"]}
        # the b quarks (last copies) and the jet nearest to each within its radius, as CMSSW matches
        c["roles"], c["quarks"] = {}, {}
        for pid, row in core.had.b_quark_rows(ev).items():
            q = ev.p[row].p
            qe, qp = core.eta(q), core.phi(q)
            best = None
            for j in jets:
                d = math.hypot(j["eta"] - qe, core.dphi(j["phi"], qp))
                if d < j["R"] and (best is None or d < best[0]):
                    best = (d, j)
            c["quarks"][pid] = (dict(pt=core.pt(q), eta=qe, phi=qp, row=row), best)
            if best:
                c["roles"][best[1]["rank"]] = "b" if pid > 0 else "bbar"
    return out


def jet_colour(rank, roles):
    if roles.get(rank) == "b":
        return COL_B
    if roles.get(rank) == "bbar":
        return COL_BBAR
    others = [r for r in range(rank + 1) if r not in roles]
    k = others.index(rank) if rank in others else len(PALETTE)
    return PALETTE[k] if k < len(PALETTE) else COL_OTHER


def circle(ax, eta0, phi0, R, color, lw, ls="-", zorder=3):
    for shift in (0.0, 2 * math.pi, -2 * math.pi):
        f = phi0 + shift
        if -math.pi - R < f < math.pi + R:
            ax.add_patch(Ellipse((eta0, f), 2 * R, 2 * R, fill=False, edgecolor=color, lw=lw, ls=ls, zorder=zorder))


def draw_panel(ax, ev, c, eta_range, label_min):
    ps.draw_axes(ax)
    ax.set_xlim(*eta_range)
    ax.set_xticks([t for t in range(-6, 7) if eta_range[0] <= t <= eta_range[1]])
    groomed = set(c["discarded"])
    for r in sorted(c["rows"], key=lambda r: (c["jet_of"].get(r, -1) >= 0, -core.pt(ev.p[r].p))):
        p = ev.p[r].p
        pt, e, f = core.pt(p), core.eta(p), core.phi(p)
        if not eta_range[0] < e < eta_range[1]:
            continue
        rank = c["jet_of"].get(r, -1)
        if r in groomed:
            ax.scatter(e, f, s=ps.area(pt) * 1.3, marker="x", color=COL_GROOM, linewidth=0.9, zorder=5)
        elif rank < 0:
            ax.scatter(e, f, s=ps.area(pt), marker="o", facecolors="none", edgecolors=COL_UNCL, linewidth=0.8, zorder=2)
        else:
            col = jet_colour(rank, c["roles"])
            ax.scatter(e, f, s=ps.area(pt), marker="o", color=col, alpha=0.9, linewidth=0, zorder=4 if col != COL_OTHER else 3)
    for j in sorted(c["jets"], key=lambda j: j["rank"], reverse=True):
        col = jet_colour(j["rank"], c["roles"])
        lead = col != COL_OTHER
        circle(ax, j["eta"], j["phi"], j["R"], col, 1.6 if lead else 0.8, "-" if lead else "--", zorder=5 if lead else 2)
        ax.scatter(j["eta"], j["phi"], s=ps.area(j["pt"]) * 1.6, marker="*", color=col, zorder=6, linewidth=0)
        for s in j["subjets"]:
            ax.scatter(s["eta"], s["phi"], s=28, marker="D", facecolors="none", edgecolors="black", linewidth=0.9, zorder=7)
        if j["pt"] >= label_min:
            ax.annotate(f"[{j['rank']}] {j['pt']:.1f}", (j["eta"], j["phi"]), xytext=(4, 3), textcoords="offset points",
                        fontsize=7 if lead else 6, color=col, weight="bold" if lead else "normal", zorder=7)
    for pid, (k, best) in c["quarks"].items():
        ax.scatter(k["eta"], k["phi"], s=ps.area(k["pt"]) * 1.3, marker="*", facecolors="none", edgecolors="black", linewidth=1.0, zorder=7)
    ax.set_title(c["title"], fontsize=10, loc="left", color=TEXT, pad=14)


def panel_text(c, n_final):
    n30 = sum(j["pt"] > 30 for j in c["jets"])
    clustered = sum(j["n"] for j in c["jets"])
    lines = [f"{len(c['jets'])} jets above {PT_MIN:g} GeV, {n30} above 30; {clustered} of {n_final} particles in them"
             + (f"; {len(c['discarded'])} groomed away (x)" if c["discarded"] else "")]
    if c.get("params"):
        lines.insert(0, c["params"])
    for pid, name, col in ((5, "$b$", COL_B), (-5, r"$\bar b$", COL_BBAR)):
        k, best = c["quarks"][pid]
        if best is None:
            lines.append((f"{name} quark {k['pt']:.1f} GeV: no jet within its radius", col))
            continue
        d, j = best
        sub = f", {len(j['subjets'])} subjets" if j["subjets"] else ""
        lines.append((f"{name} quark {k['pt']:.1f} GeV $\\rightarrow$ jet [{j['rank']}]: {j['pt']:.2f} GeV, m = {j['m']:.1f}, n = {j['n']}, "
                      f"$\\Delta R$ = {d:.3f}{sub}", col))
    return lines


def print_summary(res, n_final):
    for key, c in res.items():
        print(f"\n{c['title'].replace('$', '').replace(chr(92), '')}" + (f"; {c['params'].replace('$', '').replace(chr(92), '')}" if c.get('params') else ''))
        n30 = sum(j["pt"] > 30 for j in c["jets"])
        print(f"  {len(c['jets'])} jets above {PT_MIN:g} GeV ({n30} above 30), {sum(j['n'] for j in c['jets'])} of {n_final} particles clustered"
              + (f", {len(c['discarded'])} groomed away" if c["discarded"] else ""))
        print("   jet       pT     eta     phi       m    n     R  nsub  role")
        for j in c["jets"][:8]:
            print(f"  {j['rank']:4d} {j['pt']:8.2f} {j['eta']:7.2f} {j['phi']:7.2f} {j['m']:7.2f} {j['n']:4d} {j['R']:5.2f} {len(j['subjets']):5d}  {c['roles'].get(j['rank'], '')}")
        for pid, (k, best) in c["quarks"].items():
            name = "b" if pid > 0 else "bbar"
            if best:
                d, j = best
                print(f"  {name} quark {k['pt']:.2f} GeV (eta {k['eta']:.2f}, phi {k['phi']:.2f}) -> jet [{j['rank']}] {j['pt']:.2f} GeV, n = {j['n']}, dR = {d:.3f}, "
                      f"pT(jet)/pT(q) = {j['pt'] / k['pt']:.3f}")
            else:
                print(f"  {name} quark {k['pt']:.2f} GeV: no jet within its radius")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", type=Path, default=HERE.parent.parent / "Decay" / "event_Decay.lhe")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "algorithms_genjets.pdf")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--radius", type=float, default=core.R_AK4, help="R of anti-kT and Cambridge/Aachen (default 0.4)")
    ap.add_argument("--eta-range", type=float, nargs=2, default=(-4.6, 1.4), metavar=("LO", "HI"))
    ap.add_argument("--label-min", type=float, default=10.0, help="label jets above this pT [GeV]")
    ap.add_argument("--rho", type=float, default=600.0)
    ap.add_argument("--rmin", type=float, default=0.1)
    ap.add_argument("--rmax", type=float, default=1.5)
    ap.add_argument("--mu", type=float, default=30.0)
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--ptsub", type=float, default=30.0)
    args = ap.parse_args()
    config = hotVR.HOTVRConfig(r_min=args.rmin, r_max=args.rmax, rho=args.rho, mu=args.mu, theta=args.theta, pt_sub=args.ptsub)
    n_steps = check_against_toy(config)
    print(f"HOTVR check on the nine particles of hotVR.py: the same {n_steps} decisions and the same jets")
    ev, _, _ = core.dec.read_record(args.input)
    res = cluster_all(ev, config, args.radius)
    n_final = len(res["antikt"]["rows"])
    print_summary(res, n_final)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "mathtext.default": "regular"})
    fig = plt.figure(figsize=(16.5, 7.0))
    fig.text(0.5, 0.965, f"The same {n_final} final particles of our event under three clustering rules", fontsize=13, weight="bold", ha="center", color=TEXT)
    fig.text(0.5, 0.937, "a circle of the jet's radius around every jet above 3 GeV; the $b$ jet red, the $\\bar b$ jet blue, the next six in colour, the rest grey; "
                         "hollow: in no jet above 3 GeV; black stars: the last copies of the $b$ quarks; diamonds: HOTVR subjet axes",
             fontsize=8.5, ha="center", color=TEXT2)
    for n, key in enumerate(("antikt", "ca", "hotvr")):
        ax = fig.add_axes([0.035 + n * 0.325, 0.20, 0.30, 0.66])
        draw_panel(ax, ev, res[key], args.eta_range, args.label_min)
        y = 0.15
        for line in panel_text(res[key], n_final):
            txt, col = (line, TEXT2) if isinstance(line, str) else line
            fig.text(0.035 + n * 0.325, y, txt, fontsize=7.4, color=col, va="center", weight="bold" if col != TEXT2 else "normal")
            y -= 0.03
    fig.savefig(args.output)
    if args.png:
        fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"\nwrote {args.output}" + (f" and {args.output.with_suffix('.png')}" if args.png else ""))


if __name__ == "__main__":
    main()
