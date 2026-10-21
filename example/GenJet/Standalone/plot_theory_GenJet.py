#!/usr/bin/env python3
"""The plots behind the theory sections of chapter 7, drawn from the standalone GenJet script.

One page per theory section of the chapter. Every curve is computed with the classes of
standalone_genjets.py (the Clustering, the ghosts, the IFN replay, flavour-kT), every marker is a number
of the seed-1 record (../event_GenJet.lhe, re-clustered here from ../../Decay/event_Decay.lhe), and the
ensemble panels read the statistics written by --repeat (standalone_genjet_stats.stats.json) and by
validate_genjets_with_fastjet.py --mode full (validate_genjets_with_fastjet_full.json) when they exist.

  page 1  the true jet   the pT around the b quark against the radius: the cone and the anti-kT jet, with
                         and without the neutrinos; the clustering sequence of our AK4 jet (the ladder of
                         d_ij and the growing pT); the jet threshold and the response of the three
                         collections over the ensemble
  page 2  flavour        the ghosts of our event (b hadrons, c hadrons, partons) over the jets; the labels of
                         every jet under the six definitions; over the ensemble, how far the B hadron sits
                         from the jet axis and how much of its momentum the jet keeps
  page 3  infrared safety  a soft g -> b bbar pair between two light jets: the labels of the two jets as a
                         function of the pair's pT under the ghost, the net and the IFN definitions; the
                         neutralisation distance Omega^2 against dR^2; flavour-kT against anti-kT jets
  page 4  our jet        the cone through the stages, now with the jets (AK4, AK4 without neutrinos, AK8);
                         what the AK4 GenJet is made of (the ledger of chapter 26 in advance); the
                         response of the jets to the b quark over the ensemble and in the PYTHIA chain

    python3 plot_theory_GenJet.py                    # -> theory_genjets.pdf
    python3 plot_theory_GenJet.py --page 3 -o o.png  # one page as PNG

Needs numpy and matplotlib (on this Mac: /usr/bin/python3). About a minute (the R scan re-clusters the event
a dozen times, the soft-pair scan a dozen more).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "Had"))
import standalone_genjets as core  # noqa: E402
import plot_journey_Had as jh  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

COL_B, COL_BBAR, COL_HAD = "#ff0000", "#0000ff", "#9b4dff"
COL_STR, COL_GLU, COL_MINI = "#2a78d6", "#2f8f5b", "#eb6834"
COL_A, COL_EVENT, COL_CMSSW, COL_PY, COL_PY2 = "#8e5bb5", "#c0392b", "#1f1f1f", "#7f7f7f", "#b0b0b0"
COL_NU, COL_LEP, COL_CH, COL_GAM = "#1b9e77", "#d95f02", "#7570b3", "#e6ab02"
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d6d3ca"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})
PAGES = {1: "the true jet", 2: "flavour", 3: "infrared safety", 4: "our jet"}

# the CMSSW event (pages 6-8 of ../CMSSW/genJetAnalyzer_ME_PS_MPI_Had_Decay_GenJet_out.pdf): the status-71 b quark and its jets
CMSSW = dict(n_final=634,
             b=dict(pt_quark=91.74, eta=-1.57, phi=-2.08,
                    ak4=dict(pt=117.79, n=24, dr=0.019, response=1.284), ak4nonu=dict(pt=79.82, n=22, dr=0.017, response=0.870),
                    ak8=dict(pt=141.10, n=55, dr=0.088, response=1.538)),
             bbar=dict(pt_quark=116.36, eta=-3.75, phi=1.01,
                       ak4=dict(pt=115.94, n=18, dr=0.004, response=0.996), ak4nonu=dict(pt=115.94, n=18, dr=0.004, response=0.996),
                       ak8=dict(pt=121.01, n=32, dr=0.022, response=1.040)),
             n_jets=dict(ak4=27, ak4nonu=27, ak8=28), n_clustered=dict(ak4=239, ak4nonu=238, ak8=488))
STATS = HERE / "standalone_genjet_stats.stats.json"
PY_FULL = HERE / "validate_genjets_with_fastjet_full.json"
DECAY_STATS = HERE.parent.parent / "Decay" / "Standalone" / "standalone_decay_stats.stats.json"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Setup:
    """The seed-1 record clustered again with the script's classes."""

    def __init__(self, seed=1):
        self.args = SimpleNamespace(input=HERE.parent.parent / "Decay" / "event_Decay.lhe", algorithm="antikt", ptmin=core.JET_PT_MIN, radius=None,
                                    no_flavour_kt=False, soft_pair=None, soft_sep=0.5, repeat=0, seed=seed)
        self.ev, self.st, _ = core.run(self.args, 0)
        self.summary = core.stage_summary(self.st)
        self.b, self.bbar = self.summary["b"]["b"], self.summary["b"]["bbar"]
        self.stats = load_json(STATS)
        self.py_full = load_json(PY_FULL)
        self.decay_stats = load_json(DECAY_STATS)
        self.rng = np.random.default_rng(12345)


def mark(ax, x, label, color=COL_EVENT, y=0.93, ha="left", dx=4, ls="--"):
    ax.axvline(x, color=color, lw=1.1, ls=ls)
    ax.annotate(label, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx if ha == "left" else -dx, 0),
                textcoords="offset points", ha=ha, va="top", fontsize=9, color=color)


def tex(name: str) -> str:
    return jh.hadron_tex(name)


# ---------------------------------------------------------------------------------------------
def page_truejet(st: Setup):
    ev = st.ev
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) the pT around the b quark against R: the cone and the anti-kT jet, with and without neutrinos
    jb = st.b["quark_row"]
    axis = ev.p[jb].p
    radii = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    fin = ev.final_indices()
    nonu = [i for i in fin if not core.is_neutrino(ev.p[i].id)]
    cone, cone_nonu, jet, jet_nonu, prim = [], [], [], [], []
    primaries = [i for i, q in enumerate(ev.p) if abs(q.status) in (81, 82, 83, 84)]
    ghosts = core.ghost_rows(ev)
    for R in radii:
        cone.append(core.pt(sum((ev.p[i].p for i in core.dec.cone_members(ev, fin, axis, R)), np.zeros(4))))
        cone_nonu.append(core.pt(sum((ev.p[i].p for i in core.dec.cone_members(ev, nonu, axis, R)), np.zeros(4))))
        prim.append(core.pt(sum((ev.p[i].p for i in core.dec.cone_members(ev, primaries, axis, R)), np.zeros(4))))
        for lst, store in ((False, jet), (True, jet_nonu)):
            c = core.JetCollection(ev, "scan", R, lst, -1, core.JET_PT_MIN, ghosts)
            m = c.matched(axis)
            store.append(m[1]["pt"] if m else float("nan"))
    a1.plot(radii, cone, "o-", color=COL_B, ms=5, mec="white", label="cone around the quark, all final particles")
    a1.plot(radii, jet, "s-", color=COL_EVENT, ms=5, mec="white", label="anti-$k_T$ jet matched to the quark")
    a1.plot(radii, cone_nonu, "o--", color=COL_NU, ms=4, mec="white", label="cone without neutrinos")
    a1.plot(radii, jet_nonu, "s--", color=COL_LEP, ms=4, mec="white", label="jet without neutrinos (ak4GenJetsNoNu)")
    a1.plot(radii, prim, "^:", color=COL_HAD, ms=4, mec="white", label="cone of primary hadrons (chapter 5)")
    a1.axhline(st.b["pt_quark"], color=TEXT2, lw=0.9, ls=":")
    a1.text(0.12, st.b["pt_quark"] - 1.2, f"$b$ quark {st.b['pt_quark']:.1f} GeV", ha="left", va="top", fontsize=8.5, color=TEXT2)
    for R in (0.4, 0.8):
        a1.axvline(R, color=GRID, lw=1.5)
    a1.set_xlabel("radius $R$")
    a1.set_ylabel("$p_T$ around the $b$ quark  [GeV]")
    a1.set_ylim(70, 175)
    a1.set_title("(a) the same family, a different photograph for every $R$", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8)
    a1.text(0.98, 0.04, "R = 0.4: cone 99.4, jet 99.8 GeV\nR = 0.8: cone 108.7, jet 105.8 GeV\nbeyond R = 1 the neighbouring\ngluon jet [4] is swallowed",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8)
    # (b) the clustering sequence of our AK4 jet
    c4 = st.st.coll["ak4"]
    j4 = c4.info[st.b["jets"]["ak4"]["rank"]]
    steps = c4.cl.merges_into(j4["index"])
    cl = c4.cl
    pts = [math.sqrt(cl.pt2[k]) for (_, i, j, d, k) in steps]
    drs = [math.sqrt(core.dr2(cl.y[i], cl.phi[i], cl.y[j], cl.phi[j])) for (_, i, j, d, k) in steps]
    soft = [min(math.sqrt(cl.pt2[i]), math.sqrt(cl.pt2[j])) for (_, i, j, d, k) in steps]
    n = np.arange(1, len(steps) + 1)
    a2.plot(n, pts, "o-", color=COL_B, ms=4, mec="white", label="$p_T$ of the growing pseudojet")
    a2.set_xlabel("merge number within the jet's history")
    a2.set_ylabel("$p_T$ of the pseudojet  [GeV]", color=COL_B)
    a2b = a2.twinx()
    a2b.plot(n, drs, "s", color=COL_STR, ms=4, label=r"$\Delta R$ of the merge")
    a2b.plot(n, [R for R in np.full(len(n), 0.4)], ":", color=COL_STR, lw=0.8)
    a2b.set_ylabel(r"$\Delta R$ of the pair merged", color=COL_STR)
    a2b.set_ylim(0, 0.5)
    a2b.grid(False)
    ghost_steps = [k for k, (_, i, j, d, kk) in enumerate(steps, start=1) if (i < cl.n0 and c4.tags[i][0] != "particle") or (j < cl.n0 and c4.tags[j][0] != "particle")]
    for k in ghost_steps:
        a2.axvline(k, color=COL_HAD, lw=0.8, alpha=0.5)
    a2.text(0.03, 0.96, f"{len(steps)} merges build jet [{j4['rank']}]: the hard core first,\nthe soft periphery last; purple lines: a ghost joins\n(B hadron, D mesons, the b quark and two gluons)",
            transform=a2.transAxes, va="top", fontsize=8.5)
    a2.set_title(f"(b) anti-$k_T$ builds our jet hard-first: {j4['pt']:.2f} GeV, {j4['n']} particles", fontsize=10.5)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    # (c) threshold: number of jets above pT, and the three collections' response over the ensemble
    thr = np.linspace(0, 60, 121)
    for key, col, ls in (("ak4", COL_B, "-"), ("ak8", COL_BBAR, "-"), ("ak4nonu", COL_NU, "--")):
        pts_ = np.array([j["pt"] for j in st.st.coll[key].info])
        a3.plot(thr, [np.sum(pts_ > t) for t in thr], ls, color=col, lw=1.4, label=f"{core.COLLECTIONS[key]['title']}: {len(pts_)} jets above 3 GeV")
    a3.set_yscale("log")
    a3.set_ylim(0.8, 80)
    a3.axvline(3.0, color=TEXT2, lw=0.9, ls=":")
    a3.text(4.0, 1.0, "GenJet threshold\n3 GeV", fontsize=8.5, color=TEXT2, va="bottom")
    a3.axvline(30.0, color=TEXT2, lw=0.9, ls=":")
    a3.text(31.0, 1.0, "a typical analysis\nthreshold, 30 GeV", fontsize=8.5, color=TEXT2, va="bottom")
    a3.set_xlabel("jet $p_T$ threshold  [GeV]")
    a3.set_ylabel("jets above the threshold")
    a3.set_title("(c) the number of jets depends on the threshold", fontsize=10.5)
    a3.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"What we mean by the true jet: the cone of R = 0.4 holds {st.b['jets']['ak4']['pt_cone']:.2f} GeV, the anti-$k_T$ jet {st.b['jets']['ak4']['pt']:.2f} GeV, "
                 f"the AK8 jet {st.b['jets']['ak8']['pt']:.2f} GeV (page 1)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
FLAV_COLS = {5: "#ff0000", -5: "#0000ff", 4: "#e6ab02", -4: "#b58900", 21: "#2f8f5b", 0: "#dddddd"}


def flav_colour(f):
    if f in FLAV_COLS:
        return FLAV_COLS[f]
    return "#9ecae1" if f > 0 else "#fdae6b"


def page_flavour(st: Setup):
    ev = st.ev
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05])
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    # (a) the ghosts over the jets: a zoom on the two b regions
    c4 = st.st.coll["ak4"]
    g = st.st.ghosts
    win = (-4.4, -0.4, -3.25, 2.6)
    inside = lambda e, f: win[0] - 0.3 < e < win[1] + 0.3 and win[2] - 0.3 < f < win[3] + 0.3  # noqa: E731
    for j in c4.info:
        if j["pt"] < 5 or not inside(j["eta"], j["phi"]):
            continue
        col = COL_B if j["rank"] == st.b["jets"]["ak4"]["rank"] else COL_BBAR if j["rank"] == st.bbar["jets"]["ak4"]["rank"] else "#999999"
        a1.add_patch(Circle((j["eta"], j["phi"]), 0.4, fill=False, edgecolor=col, lw=1.4 if col != "#999999" else 0.8, ls="-" if col != "#999999" else "--"))
        a1.text(j["eta"] + 0.05, j["phi"] + 0.42, f"[{j['rank']}] {j['pt']:.0f}", fontsize=7.5, color=col, clip_on=True)
        for r in j["rows"]:
            k = ev.p[r].p
            a1.scatter(core.eta(k), core.phi(k), s=6 + 10 * math.log10(1 + core.pt(k)), color=col, alpha=0.35, linewidth=0)
    for kind, marker, col, label in (("bhad", "*", COL_B, "b hadron ghost (last in chain)"), ("chad", "D", COL_GAM, "c hadron ghost"),
                                     ("physics", "P", "black", "hard-process parton (status 23)")):
        for r in g[kind]:
            k = ev.p[r].p
            if not inside(core.eta(k), core.phi(k)):
                continue
            a1.scatter(core.eta(k), core.phi(k), s=70 if kind != "chad" else 40, marker=marker, facecolors="none", edgecolors=col, linewidth=1.3, zorder=5)
            if kind != "chad" or core.pt(k) > 5:
                a1.annotate(tex(core.name(ev.p[r].id)) + f" {core.pt(k):.0f}", (core.eta(k), core.phi(k)), xytext=(4, -9), textcoords="offset points", fontsize=7, color=col,
                            zorder=6, annotation_clip=True)
        a1.scatter([], [], marker=marker, facecolors="none", edgecolors=col, label=label)
    for r in g["parton"]:
        k = ev.p[r].p
        if core.pt(k) > 3 and inside(core.eta(k), core.phi(k)):
            a1.scatter(core.eta(k), core.phi(k), s=18, marker="x", color=COL_GLU if ev.p[r].id == 21 else COL_STR, linewidth=0.9, zorder=4)
    a1.scatter([], [], marker="x", color=COL_GLU, label="parton of status 71 (gluon) above 3 GeV")
    a1.scatter([], [], marker="x", color=COL_STR, label="parton of status 71 (quark) above 3 GeV")
    a1.set_xlim(win[0], win[1])
    a1.set_ylim(win[2], win[3])
    a1.set_xlabel(r"$\eta$")
    a1.set_ylabel(r"$\phi$")
    a1.set_title("(a) the ghosts of our event over the AK4 jets", fontsize=10.5)
    a1.legend(loc="upper right", fontsize=7)
    # (b) the label matrix of the jets above 5 GeV
    defs = [("hadron", "hadronFlavour"), ("parton", "partonFlavour"), ("physics", "physics (status 23)"), ("legacy", r"$\Delta R<0.3$ parton"), ("net", "net b (hadrons)"), ("ifn", "IFN b")]
    jets = [j for j in c4.info if j["pt"] > 5]
    stable = {j["rank"]: j for j in st.st.stable_jets}

    def nearest_stable(j):
        best = min(st.st.stable_jets, key=lambda s_: core.dr2(s_["eta"], s_["phi"], j["eta"], j["phi"]))
        return best if core.dr2(best["eta"], best["phi"], j["eta"], j["phi"]) < 0.4 ** 2 else None
    a2.set_xlim(0, len(defs))
    a2.set_ylim(len(jets), 0)
    a2.set_xticks(np.arange(len(defs)) + 0.5)
    a2.set_xticklabels([d[1] for d in defs], fontsize=7.5, rotation=30, ha="right")
    a2.set_yticks(np.arange(len(jets)) + 0.5)
    a2.set_yticklabels([f"[{j['rank']}] {j['pt']:.0f} GeV" for j in jets], fontsize=7.5)
    a2.grid(False)
    for yi, j in enumerate(jets):
        sj = nearest_stable(j)
        vals = [j["hadron_flavour"], j["parton_flavour"], j["physics_flavour"], j["legacy_flavour"],
                (sj["net_flavour"].get(5, 0) if sj else 0), (sj["ifn_flavour"].get(5, 0) if sj else 0)]
        for xi, v in enumerate(vals):
            key = v if xi < 4 else (5 if v > 0 else -5 if v < 0 else 0)
            a2.add_patch(plt.Rectangle((xi, yi), 1, 1, color=flav_colour(key), alpha=0.75 if key else 0.5, lw=0))
            txt = str(v) if xi < 4 else (f"{v:+d}" if v else "0")
            a2.text(xi + 0.5, yi + 0.5, txt, ha="center", va="center", fontsize=7.5, color="white" if key in (5, -5) else TEXT)
    a2.set_title("(b) one jet, six labels: the AK4 jets above 5 GeV", fontsize=10.5, pad=16)
    a2.text(0.5, 1.0, "red b, blue $\\bar b$, yellow c, green gluon, pale blue/orange light quark, grey none", transform=a2.transAxes, fontsize=7.5, va="bottom", ha="center", color=TEXT2)
    # (c) ensemble: the B hadron against the jet axis and its share
    if st.stats:
        d = st.stats["b"]["b"]["jets"]["ak4"]
        drh = np.array(d["dr_hadron"], dtype=float)
        frac = np.array(d["pt_from_hadron"], dtype=float) / np.array(d["pt"], dtype=float)
        a3.hist(drh[~np.isnan(drh)], bins=np.linspace(0, 0.2, 41), color=COL_B, alpha=0.45, label=f"$\\Delta R$(B hadron, jet axis): mean {np.nanmean(drh):.3f}")
        a3.set_xlabel(r"$\Delta R$ between the $\bar B^0$ and the AK4 jet axis")
        a3.set_ylabel(f"decays of the record ({st.stats['repeat']})")
        a3b = a3.twiny()
        a3b.hist(frac[~np.isnan(frac)], bins=np.linspace(0.5, 1.0, 51), histtype="step", color=COL_HAD, lw=1.4, label=f"share of the jet $p_T$ from the B: mean {np.nanmean(frac):.3f}")
        a3b.set_xlabel("share of the jet $p_T$ from the B hadron's descendants", color=COL_HAD, fontsize=9)
        a3b.grid(False)
        hf = np.array(d["hadron_flavour"], dtype=float)
        pf = np.array(d["parton_flavour"], dtype=float)
        a3.text(0.97, 0.55, f"hadronFlavour = 5 in {100 * np.nanmean(hf == 5):.1f}% of the decays\npartonFlavour = 5 in {100 * np.nanmean(np.abs(pf) == 5):.1f}%\n"
                            f"the B's products outside the jet: {np.nanmean(d['n_hadron_outside']):.2f} on average",
                transform=a3.transAxes, ha="right", va="top", fontsize=8.5)
        h1, l1 = a3.get_legend_handles_labels()
        h2, l2 = a3b.get_legend_handles_labels()
        a3.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    else:
        a3.text(0.5, 0.5, "no ensemble file yet:\nstandalone_genjets.py --repeat 400\n--history standalone_genjet_stats.json", ha="center", transform=a3.transAxes, fontsize=9)
    mark(a3, st.b["jets"]["ak4"]["dr_hadron"], f"our event: {st.b['jets']['ak4']['dr_hadron']:.3f}", color=COL_EVENT, y=0.3)
    a3.set_title("(c) the B hadron sits on the jet axis", fontsize=10.5, pad=22)
    fig.suptitle(f"Ghost-associated hadron flavour and parton flavour: our $b$ jet is 5 / {st.b['jets']['ak4']['parton_flavour']:+d}, the gluon-splitting pair shares one jet (page 2)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_irsafety(st: Setup):
    ev = st.ev
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    # (a) the soft pair scan
    pts = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
    labels = dict(ghost=[], net=[], ifn=[], fkt=[])
    between = None
    for p in pts:
        extra, between = core.soft_pair(ev, st.st.coll["ak4"].info, p, 0.5)
        args = SimpleNamespace(**vars(st.args))
        st2 = core.Stage(ev, args, verbose=0, soft_pair=extra)
        # the two jets that hold the soft hadrons
        ghost = net = ifn = fkt = 0
        for j in st2.coll["ak4"].info:
            if j["extras"]:
                ghost += 1 if j["hadron_flavour"] == 5 else 0
        for j in st2.stable_jets:
            if any(lab.startswith("soft") for lab in j["flavoured"]):
                net += 1 if j["net_flavour"].get(5) else 0
                ifn += 1 if j["ifn_flavour"].get(5) else 0
        for j in st2.fkt_jets:
            if any(lab.startswith("soft") for lab in j["flavoured"]):
                fkt += 1 if j["flavour"].get(5) else 0
        labels["ghost"].append(ghost)
        labels["net"].append(net)
        labels["ifn"].append(ifn)
        labels["fkt"].append(fkt)
    for key, col, ls, lab in (("ghost", COL_B, "-", "hadronFlavour (ghost B hadrons)"), ("net", COL_HAD, "--", "net flavour"), ("ifn", COL_GLU, "-", "IFN (anti-$k_T$ kinematics)"),
                              ("fkt", COL_LEP, ":", "flavour-$k_T$")):
        a1.plot(pts, np.array(labels[key]) + {"ghost": 0.0, "net": 0.04, "ifn": -0.04, "fkt": 0.08}[key], "o" + ls, color=col, ms=5, mec="white", lw=1.5, label=lab)
    a1.set_xscale("log")
    a1.set_yticks([0, 1, 2])
    a1.set_yticklabels(["neither jet\nlabelled b", "one jet", "both jets\nlabelled b"])
    a1.set_xlabel("$p_T$ of each hadron of the soft $g \\to b\\bar b$ pair  [GeV]")
    a1.set_title(f"(a) a soft $B^0\\bar B^0$ pair straddling jets [{between[0]}] and [{between[1]}]", fontsize=10.5)
    a1.legend(loc="center left", fontsize=8)
    a1.text(0.03, 0.12, "the pair sits 0.5 apart in $\\Delta R$, one hadron inside each jet;\nan infrared-safe label must not change as $p_T \\to 0$",
            transform=a1.transAxes, fontsize=8, va="bottom")
    a1.set_ylim(-0.35, 2.35)
    # (b) Omega^2 against dR^2
    dy = np.linspace(0, 3, 200)
    for dphi_, col in ((0.0, COL_STR), (1.0, COL_GLU), (2.0, COL_LEP)):
        om = core.omega2(dy, dphi_, 0.0, 0.0, 1.0)
        a2.plot(dy, om, color=col, lw=1.4, label=rf"$\Omega^2$, $\Delta\phi$ = {dphi_:g}")
        a2.plot(dy, dy ** 2 + dphi_ ** 2, ":", color=col, lw=1.0)
    a2.plot([], [], ":", color=TEXT2, label=r"$\Delta R^2$ (dotted)")
    a2.set_yscale("log")
    a2.set_ylim(1e-2, 2e2)
    a2.set_xlabel(r"$\Delta y$")
    a2.set_ylabel(r"$\Omega^2_{ik}$ = 2[(cosh$\,\omega\Delta y$ - 1)/$\omega^2$ - (cos$\,\Delta\phi$ - 1)],  $\omega$ = 1")
    a2.set_title("(b) the neutralisation distance grows like $e^{|\\Delta y|}$ at large rapidity", fontsize=10.5)
    a2.legend(loc="upper left", fontsize=8)
    for q, rec, col in (("b", st.b, COL_B), ("bbar", st.bbar, COL_BBAR)):
        sj = rec.get("stable_jet")
        if sj:
            a2.text(0.97, 0.05 + (0.08 if q == "bbar" else 0.0), f"{q} jet: net {core.flavour_text(sj['net_flavour'])}, IFN {core.flavour_text(sj['ifn_flavour'])}", transform=a2.transAxes,
                    ha="right", fontsize=8.5, color=col)
    # (c) flavour-kT against anti-kT (the heavy hadrons kept)
    akt = st.st.stable_jets
    fkt = st.st.fkt_jets
    xs, ys, cols = [], [], []
    for j in akt:
        if j["pt"] < 3:
            continue
        m = min(fkt, key=lambda f: core.dr2(f["eta"], f["phi"], j["eta"], j["phi"]), default=None)
        if m is None or core.dr2(m["eta"], m["phi"], j["eta"], j["phi"]) > 0.4 ** 2:
            continue
        xs.append(j["pt"])
        ys.append(m["pt"])
        cols.append(COL_B if j["ifn_flavour"].get(5, 0) > 0 else COL_BBAR if j["ifn_flavour"].get(5, 0) < 0 else "#999999")
    a3.scatter(xs, ys, c=cols, s=28, alpha=0.8, linewidth=0)
    lim = max(xs + ys + [10]) * 1.1
    a3.plot([0, lim], [0, lim], ":", color=TEXT2, lw=0.9)
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_xlim(2.5, lim)
    a3.set_ylim(2.5, lim)
    a3.set_xlabel("anti-$k_T$ jet $p_T$  [GeV]  (heavy hadrons kept)")
    a3.set_ylabel("nearest flavour-$k_T$ jet $p_T$  [GeV]")
    a3.scatter([], [], color=COL_B, label="IFN flavour b")
    a3.scatter([], [], color=COL_BBAR, label=r"IFN flavour $\bar b$")
    a3.scatter([], [], color="#999999", label="no flavour")
    a3.legend(loc="upper left", fontsize=8)
    nb = sum(1 for f in fkt if f["flavour"].get(5))
    a3.text(0.97, 0.05, f"{len(akt)} anti-$k_T$ jets, {len(fkt)} flavour-$k_T$ jets above 3 GeV;\n{nb} flavour-$k_T$ jets carry b flavour", transform=a3.transAxes, ha="right", fontsize=8.5)
    a3.set_title("(c) flavour-$k_T$ jets against anti-$k_T$ jets", fontsize=10.5)
    fig.suptitle("Infrared-safe flavour: a soft pair must not relabel the photograph (page 3)", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
def page_jet(st: Setup):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2))
    r = st.b
    # (a) the cone through the stages, now with the jets
    dcone = st.decay_stats["b"]["b"]["cones"]["0.4"] if st.decay_stats else None
    j4, j4n, j8 = r["jets"]["ak4"], r["jets"]["ak4nonu"], r["jets"]["ak8"]
    labels = ["partons\n(stage 3)", "primary\nhadrons (4)", "final\nparticles (5)", "AK4 GenJet\n(6)", "AK4 no $\\nu$", "AK8 GenJet"]
    vals = [None, None, j4["pt_cone"], j4["pt"], j4n["pt"], j8["pt"]]
    # the first two from the decay summary of chapter 6 (the cone around the same axis)
    if st.decay_stats:
        vals[0] = None
    dec_args = SimpleNamespace(input=HERE.parent.parent / "Had" / "event_Had.lhe", tau0max=core.dec.TAU0_MAX, limit_tau0=True, no_mix=False, stable_b=False, radius=0.4)
    e6, d6 = core.dec.run(dec_args, 1)
    s6 = core.dec.summarise(e6, d6)["b"]["b"]["cones"][0.4]
    vals[0], vals[1] = s6["partons"]["pt"], s6["primaries"]["pt"]
    x = np.arange(len(labels))
    a1.plot(x, vals, "o-", color=COL_B, ms=7, mec="white", lw=1.4, label="our event (seed 1)")
    for xi, v in zip(x, vals):
        a1.annotate(f"{v:.1f}", (xi, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=COL_B)
    if st.stats:
        d = st.stats["b"]["b"]["jets"]
        means = [vals[0], np.mean(dcone["primaries"]["pt"]) if dcone else vals[1], np.nanmean(d["ak4"]["pt_cone"]), np.nanmean(d["ak4"]["pt"]), np.nanmean(d["ak4nonu"]["pt"]), np.nanmean(d["ak8"]["pt"])]
        stds = [0, np.std(dcone["primaries"]["pt"]) if dcone else 0, np.nanstd(d["ak4"]["pt_cone"]), np.nanstd(d["ak4"]["pt"]), np.nanstd(d["ak4nonu"]["pt"]), np.nanstd(d["ak8"]["pt"])]
        a1.errorbar(x + 0.08, means, yerr=stds, fmt="s", color=COL_B, ms=4, lw=0.8, alpha=0.6, label=f"{st.stats['repeat']} decays (mean, std)")
    a1.axhline(r["pt_quark"], color=TEXT2, lw=0.9, ls=":")
    a1.text(0.0, r["pt_quark"] + 0.6, f"$b$ quark {r['pt_quark']:.1f}", ha="left", fontsize=8.5, color=TEXT2)
    a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=8.5)
    a1.set_ylabel("$p_T$  [GeV]")
    a1.set_title("(a) the family through the stages, and its photographs", fontsize=10.5)
    a1.legend(loc="lower right", fontsize=8)
    a1.text(0.03, 0.96, f"CMSSW's $b$ jet: AK4 {CMSSW['b']['ak4']['pt']:.1f}, without $\\nu$ {CMSSW['b']['ak4nonu']['pt']:.1f},\nAK8 {CMSSW['b']['ak8']['pt']:.1f} GeV for a quark of {CMSSW['b']['pt_quark']:.1f}",
            transform=a1.transAxes, va="top", fontsize=8.5, color=COL_CMSSW)
    # (b) what the AK4 GenJet is made of
    comp = j4["composition"]
    keys = ["charged_hadron", "photon", "neutral_hadron", "electron", "muon", "neutrino"]
    names = ["charged\nhadrons", "photons", "neutral\nhadrons", "electrons", "muons", "neutrinos"]
    cols = [COL_CH, COL_GAM, COL_STR, COL_LEP, COL_LEP, COL_NU]
    vals_c = [comp[k]["pt"] for k in keys]
    bars = a2.bar(np.arange(len(keys)) - 0.18, vals_c, 0.36, color=cols, alpha=0.85, label="our event")
    for b_, k in zip(bars, keys):
        a2.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.8, f"{comp[k]['pt']:.1f}\n({comp[k]['n']})", ha="center", fontsize=7.5)
    if st.stats:
        cl = [c for c in st.stats["b"]["b"]["jets"]["ak4"]["composition"] if c]
        means = [np.mean([c[k] for c in cl]) for k in keys]
        stds = [np.std([c[k] for c in cl]) for k in keys]
        a2.bar(np.arange(len(keys)) + 0.18, means, 0.36, yerr=stds, color=cols, alpha=0.4, label=f"{len(cl)} decays", error_kw=dict(lw=0.8))
    if st.py_full:
        cl = [c for c in st.py_full["b"]["b"]["jets"]["ak4"]["composition"] if c]
        means = [np.mean([c[k] for c in cl]) for k in keys]
        a2.plot(np.arange(len(keys)) + 0.18, means, "_", color=COL_PY, ms=14, mew=2, label=f"PYTHIA + FastJet, full chain ({len(cl)})")
    a2.set_xticks(np.arange(len(keys)))
    a2.set_xticklabels(names, fontsize=8.5)
    a2.set_ylabel("$p_T$ in the AK4 GenJet  [GeV]")
    a2.set_title(f"(b) what the {j4['pt']:.1f} GeV jet is made of (the ledger of chapter 26)", fontsize=10.5)
    a2.legend(loc="upper right", fontsize=8)
    # (c) the response over the ensemble
    bins = np.linspace(0.6, 1.8, 49)
    if st.stats:
        d = st.stats["b"]["b"]["jets"]
        for key, col, ls in (("ak4", COL_B, "-"), ("ak4nonu", COL_NU, "--"), ("ak8", COL_BBAR, "-")):
            v = np.array(d[key]["response"], dtype=float)
            v = v[~np.isnan(v)]
            a3.hist(v, bins=bins, histtype="step", color=col, lw=1.5, ls=ls, label=f"{core.COLLECTIONS[key]['title']}: {v.mean():.3f} $\\pm$ {v.std():.3f}")
    if st.py_full:
        d = st.py_full["b"]["b"]["jets"]
        for key, col in (("ak4", COL_PY), ("ak4nonu", COL_PY2)):
            v = np.array(d[key]["response"], dtype=float)
            v = v[~np.isnan(v)]
            a3.hist(v, bins=np.linspace(0.3, 2.4, 43), histtype="step", color=col, lw=1.2, ls=":" if key == "ak4nonu" else "-", density=False,
                    weights=np.full(len(v), (st.stats["repeat"] if st.stats else 400) / max(len(v), 1)),
                    label=f"PYTHIA chain, {core.COLLECTIONS[key]['title']}: {v.mean():.3f} $\\pm$ {v.std():.3f}")
    a3.set_yscale("log")
    a3.set_ylim(0.7, 3000)
    for key, col, y in (("ak4", COL_B, 0.50), ("ak8", COL_BBAR, 0.42)):
        mark(a3, r["jets"][key]["response"], f"our event {r['jets'][key]['response']:.3f}", color=col, y=y)
    mark(a3, CMSSW["b"]["ak4nonu"]["response"], f"CMSSW without $\\nu$: {CMSSW['b']['ak4nonu']['response']:.3f}", color=COL_CMSSW, y=0.34, ls=":", ha="right")
    mark(a3, CMSSW["b"]["ak4"]["response"], f"CMSSW AK4: {CMSSW['b']['ak4']['response']:.3f}", color=COL_CMSSW, y=0.26, ls=":")
    a3.set_xlabel("$p_T$(jet) / $p_T$($b$ quark, last copy)")
    a3.set_ylabel("decays of the record (PYTHIA chain scaled to 400)")
    a3.set_title("(c) the response: what the photograph shows of the quark", fontsize=10.5)
    a3.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Our jet at this stage: AK4 GenJet {j4['pt']:.2f} GeV ({j4['n']} particles, hadronFlavour {j4['hadron_flavour']}), without neutrinos {j4n['pt']:.2f}, AK8 {j8['pt']:.2f} GeV (page 4)",
                 fontsize=12)
    fig.tight_layout()
    return fig


PAGE_FUNCS = {1: page_truejet, 2: page_flavour, 3: page_irsafety, 4: page_jet}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=HERE / "theory_genjets.pdf")
    ap.add_argument("--page", type=int, default=None, help="one page only (PNG or PDF)")
    args = ap.parse_args()
    st = Setup()
    print(f"{st.summary['n_final']} final particles; the b quark's AK4 jet: {st.b['jets']['ak4']['pt']:.2f} GeV, hadronFlavour {st.b['jets']['ak4']['hadron_flavour']}, "
          f"partonFlavour {st.b['jets']['ak4']['parton_flavour']}")
    if args.page:
        fig = PAGE_FUNCS[args.page](st)
        fig.savefig(args.output, dpi=130)
        print(f"wrote {args.output}")
        return
    with PdfPages(str(args.output)) as pdf:
        for n in sorted(PAGE_FUNCS):
            fig = PAGE_FUNCS[n](st)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  page {n}: {PAGES[n]}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
