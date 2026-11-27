#!/usr/bin/env python3
"""The family inside the Geant4 CMS detector, station by station, in 3D.

Five pages, from the centre outwards, all drawn from what Geant4 did with the 672 primaries of the signal
(../g4tracks_signal.npz, ../simhits_signal.npz) inside the geometry it tracked them through
(../cms_shapes_SIM.npz) and the field it bent them in (field_map_SIM.npz):

  magnet   the solenoid, the return yoke and the field lines, with the tracker, the calorimeters and part of
           the muon chambers inside; next to it the field in the (z, r) plane
  tracker  the silicon modules and every trajectory Geant4 followed in the tracker volume, with the hits
  ecal     the crystals, coloured by the energy they received, and one shower from close by: the energy
           deposits of its thousands of electrons and photons inside the crystals
  hcal     the scintillator tiles and the deposits of the hadronic showers in them
  muon     the yoke and the chambers: what is left of the family behind the calorimeters

  python3 display_geant4_SIM.py                 # the five pages -> pages/geant4_<name>.png
  python3 display_geant4_SIM.py --page tracker  # one of them
  python3 display_geant4_SIM.py --dpi 200       # larger

Only the signal is drawn. The 960 collisions of the crowd go through exactly the same chain (geant4_SIM.py
--sample intime, --sample oot); their hits appear in the displays of plot_simhits_SIM.py.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import g4view_SIM as gv
import standalone_SIM as sim
from field_map_SIM import FieldMap

HERE = Path(__file__).resolve().parent
PAGES = ["magnet", "tracker", "ecal", "hcal", "muon"]
TXT = "#e8eaf0"
E_CMAP = plt.get_cmap("inferno")
B_CMAP = plt.get_cmap("turbo")


def e_colour(e, lo=1e-3, hi=30.0):
    x = (np.log10(np.maximum(e, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return E_CMAP(0.25 + 0.75 * np.clip(x, 0, 1))


def header(fig, title, lines):
    fig.text(0.015, 0.955, title, color=TXT, fontsize=17, fontweight="bold")
    for i, l in enumerate(lines):
        fig.text(0.015, 0.92 - 0.03 * i, l, color="#aab2c0", fontsize=10.5)


def kind_legend(fig, ev, mask, x=0.015, y=0.2, extra=()):
    hs = []
    for k in range(5):
        n = len(np.unique(ev.seg_trk[mask & (ev.seg_kind == k)]))
        if n:
            hs.append(Line2D([], [], color=gv.KIND_COL[k], lw=2, label=f"{gv.KIND_NAME[k]} ({n})"))
    n = len(np.unique(ev.seg_trk[mask & (ev.seg_kind == 5)]))
    if n:
        hs.append(Line2D([], [], color=gv.NU_COL, lw=2, label=f"neutrino ({n}), unseen"))
    hs += list(extra)
    leg = fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(x, y), frameon=False, fontsize=10, labelcolor=TXT,
                     title="trajectories by particle", title_fontsize=10)
    leg.get_title().set_color(TXT)
    leg._legend_box.align = "left"


def crystal_energy(det, ev):
    """Energy per crystal: every shower voxel is given to the crystal whose axis it is nearest to."""
    from scipy.spatial import cKDTree
    out = {}
    for name, code in (("eb", sim.EB), ("ee", sim.EE)):
        c = det.raw[name]
        ctr = c.mean(axis=1)
        v = ev.vox[ev.vox[:, 0] == code]
        e = np.zeros(len(c))
        if len(v):
            _, i = cKDTree(ctr).query(v[:, 1:4])
            np.add.at(e, i, v[:, 4])
        out[name] = e
    return out


def tile_energy(det, ev):
    from scipy.spatial import cKDTree
    out = {}
    for name, code in (("hb", sim.HB), ("he", sim.HE)):
        q = det.L[name]
        v = ev.vox[ev.vox[:, 0] == code]
        e = np.zeros(len(q))
        if len(v):
            # a tile is large: the nearest of its four corners and centre decides
            pts = np.concatenate([q.mean(axis=1)] + [0.5 * (q.mean(axis=1) + q[:, j]) for j in range(4)])
            _, i = cKDTree(pts).query(v[:, 1:4])
            np.add.at(e, i % len(q), v[:, 4])
        out[name] = e
    return out


# ---------------------------------------------------------------------------
def page_magnet(det, ev, fmap, out, dpi):
    fig = plt.figure(figsize=(14.4, 7.2), dpi=dpi, facecolor=gv.BG)
    ax = fig.add_axes([0.0, 0.0, 0.62, 0.9], facecolor=gv.BG)
    cam = gv.Camera.orbit((0, -60, 0), 4700.0, 62.0, 21.0, fov=34.0, aspect=0.62 * 14.4 / (0.9 * 7.2))
    pt = gv.Painter(cam)
    cut = gv.quadrant_cut(-35.0, 95.0)
    half = lambda x, y, z: cut(x, y, z) | (z > 700.0)
    pt.surface(det.quads("yoke", cut), gv.DET_COL["yoke"], 0.5, edge="#5a1f1b", lw=0.15)
    pt.surface(det.quads("dt", cut), gv.DET_COL["dt"], 0.22, lw=0.1)
    pt.surface(det.quads("csc", cut), gv.DET_COL["csc"], 0.3, lw=0.1)
    pt.surface(det.quads("rpc", cut), gv.DET_COL["rpc"], 0.12, lw=0.05)
    ph0, ph1 = math.radians(95.0), math.radians(325.0)
    pt.surface(np.concatenate([gv.cylinder(320.0, -615.0, 615.0, ph0, ph1), gv.cylinder(340.0, -615.0, 615.0, ph0, ph1),
                               gv.annulus(320.0, 340.0, 615.0, ph0, ph1), gv.annulus(320.0, 340.0, -615.0, ph0, ph1)]), gv.DET_COL["coil"], 0.55, edge="#6d7480", lw=0.1)
    pt.surface(det.quads("ho", cut), gv.DET_COL["ho"], 0.3)
    pt.surface(det.quads("hb", cut), gv.DET_COL["hb"], 0.35, lw=0.1)
    pt.surface(det.quads("he", cut), gv.DET_COL["he"], 0.35, lw=0.1)
    pt.surface(det.quads("hf", None), gv.DET_COL["hf"], 0.5, edge="#4a443c", lw=0.2)
    pt.surface(det.quads("eb", cut)[::3], gv.DET_COL["eb"], 0.5, lw=0.0)
    pt.surface(det.quads("ee", cut)[::2], gv.DET_COL["ee"], 0.5, lw=0.0)
    pt.surface(det.quads("trk", None)[::2], gv.DET_COL["trk"], 0.45, lw=0.0)
    pt.surface(det.quads("pix", None), gv.DET_COL["pix"], 0.8, lw=0.0)
    # field lines in the two faces of the cut and in the vertical plane behind
    lines = gv.field_lines(fmap, levels=11)
    for phi, a in ((-35.0, 1.0), (95.0, 1.0), (215.0, 0.45)):
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        for L in lines:
            P = np.column_stack([L[:, 1] * c, L[:, 1] * s, L[:, 0]])
            col = B_CMAP(np.clip(0.5 * (L[:-1, 2] + L[1:, 2]) / 4.0, 0, 1))
            col[:, 3] = a
            pt.line(P[:-1], P[1:], col, lw=1.3)
    pt.paint(ax)
    header(fig, "The magnet of CMS inside Geant4: coil, return yoke and field lines",
           ["geometry: cmsSimGeo2021.gdml, one third of the azimuth removed", "field: Ampere's law solved for this coil and this iron (field_map_SIM.py); lines coloured by |B|"])
    hs = [Line2D([], [], color=gv.DET_COL[k], lw=6, label=l) for k, l in (("pix", "pixel detector"), ("trk", "strip tracker"), ("eb", "ECAL crystals"), ("hb", "HCAL scintillator"),
                                                                        ("coil", "superconducting coil"), ("yoke", "steel of the return yoke"), ("dt", "muon chambers (DT, CSC)"), ("hf", "forward calorimeter"))]
    leg = fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.012, 0.02), frameon=False, fontsize=9.5, labelcolor=TXT, ncol=2)
    # the field in the (z, r) plane
    ax2 = fig.add_axes([0.655, 0.16, 0.31, 0.6], facecolor=gv.BG)
    br, bz = fmap.grid(average=True)
    b = np.hypot(br, bz)
    z = np.concatenate([-fmap.z[:0:-1], fmap.z])
    bb = np.concatenate([b[:, :0:-1], b], axis=1)
    im = ax2.pcolormesh(z / 100, fmap.r / 100, bb, cmap=B_CMAP, vmin=0, vmax=4.0, shading="auto", rasterized=True)
    for L in gv.field_lines(fmap, levels=16):
        ax2.plot(L[:, 0] / 100, L[:, 1] / 100, color="white", lw=0.5, alpha=0.8)
    ax2.set_xlim(-14, 14); ax2.set_ylim(0, 9.0)
    ax2.set_xlabel("z [m]", color=TXT, fontsize=11); ax2.set_ylabel("r [m]", color=TXT, fontsize=11)
    ax2.tick_params(colors=TXT, labelsize=9)
    for sp in ax2.spines.values():
        sp.set_color("#59606c")
    cax = fig.add_axes([0.655, 0.83, 0.31, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=TXT, labelsize=9)
    cb.set_label("|B| [T], averaged over azimuth; white: field lines", color=TXT, fontsize=9.5, labelpad=-38)
    c = fmap.b(0, 0, 0)
    y1 = fmap.b(475.0, 0, 0, True); y2 = fmap.b(585.0, 0, 0, True); d1 = fmap.b(400.0, 0, 760.0, True)
    fig.text(0.655, 0.075, f"centre {c[2]:.2f} T    barrel yoke {y1[2]:+.2f}, {y2[2]:+.2f} T    endcap disk {math.hypot(d1[0], d1[2]):.2f} T (radial)\n"
                           f"coil current {fmap.current / 1e6:.1f} million ampere-turns", color="#aab2c0", fontsize=9.5, va="top")
    fig.savefig(out, dpi=dpi, facecolor=gv.BG)
    plt.close(fig)


def page_tracker(det, ev, fmap, out, dpi, cam=None, t_now=None, fig_ax=None, decorate=True):
    fig, ax = fig_ax or gv.new_figure(14.4, 7.2, dpi)
    cam = cam or gv.Camera.orbit((0, -8, 0), 660.0, 68.0, 20.0, fov=36.0, aspect=2.0)
    pt = gv.Painter(cam)
    cut = lambda x, y, z: (np.degrees(np.arctan2(y, x)) > -10.0) & (np.degrees(np.arctan2(y, x)) < 110.0)
    pt.surface(det.quads("trk", cut), gv.DET_COL["trk"], 0.2, edge="#6aa5e8", lw=0.18)
    pt.surface(det.quads("pix", cut), gv.DET_COL["pix"], 0.3, edge="#a9e0ff", lw=0.12)
    pt.surface(det.quads("pipe"), gv.DET_COL["pipe"], 0.10, lw=0.0)
    mask = tracker_mask(ev)
    idx, col, lw = gv.track_style(ev, mask, t_now)
    a, b = ev.segments(idx)
    pt.line(a, b, col, lw=lw)
    h = ev.hits
    hm = np.ones(len(h["trk_x"]), bool) if t_now is None else h["trk_t"] <= t_now
    pt.dot(np.column_stack([h["trk_x"], h["trk_y"], h["trk_z"]])[hm], "#ffe14d", size=2.2, alpha=0.9)
    pt.paint(ax)
    if decorate:
        n = len(np.unique(ev.seg_trk[mask]))
        header(fig, "The tracker inside Geant4: every trajectory of the family, and its hits",
               [f"{len(np.unique(ev.seg_trk[~ev.outer[ev.seg_trk]]))} trajectories followed in the tracker volume, {n} drawn (charged above 2 MeV, photons above 100 MeV);"
                f" {len(h['trk_x'])} hits (yellow) in {len(det.trk_quads)} silicon modules",
                "thick = energetic; the spirals are electrons of a few MeV curling in the 3.8 T field; one third of the modules removed"])
        kind_legend(fig, ev, mask, y=0.03, extra=[Line2D([], [], ls="none", marker="o", color="#ffe14d", ms=5, label="tracker hit")])
        fig.savefig(out, dpi=dpi, facecolor=gv.BG)
        plt.close(fig)
    return pt


def tracker_mask(ev, e_charged=0.002, e_photon=0.1, e_neutral=0.5):
    """The trajectories worth drawing in the tracker: charged particles above 2 MeV, photons above 100 MeV, neutral
    hadrons above 500 MeV. The thousands of softer photons would only be a green fog."""
    k, ek = ev.seg_kind, ev.seg_ek
    return ev.seg_in_trk & ~ev.outer[ev.seg_trk] & ((np.isin(k, (0, 3, 4, 6)) & (ek > e_charged)) | ((k == 2) & (ek > e_photon)) | ((k == 1) & (ek > e_neutral)))


def _best_shower(ev, code):
    """The primary that left the most energy in a subdetector, and the centre of its deposits."""
    v = ev.vox[ev.vox[:, 0] == code]
    prim = v[:, 6].astype(int)
    e = np.bincount(prim, weights=v[:, 4])
    j = int(np.argmax(e))
    w = v[prim == j]
    ctr = np.average(w[:, 1:4], axis=0, weights=w[:, 4])
    return j, ctr, float(e[j])


def page_ecal(det, ev, fmap, out, dpi):
    fig = plt.figure(figsize=(14.4, 7.2), dpi=dpi, facecolor=gv.BG)
    ce = crystal_energy(det, ev)
    # left: all crystals from outside, coloured by their energy
    ax = fig.add_axes([0.0, 0.0, 0.56, 0.9], facecolor=gv.BG)
    cam = gv.Camera.orbit((0, -10, 0), 1180.0, 66.0, 21.0, fov=36.0, aspect=0.56 * 14.4 / (0.9 * 7.2))
    pt = gv.Painter(cam)
    cut = lambda x, y, z: (np.degrees(np.arctan2(y, x)) > -5.0) & (np.degrees(np.arctan2(y, x)) < 115.0) & (z > -340.0)
    for name in ("eb", "ee"):
        q, c = det.L[name], det.centre[name]
        keep = ~cut(c[:, 0], c[:, 1], c[:, 2])
        e = ce[name]
        cold = keep & (e < 1e-3)
        pt.surface(q[cold], gv.DET_COL[name], 0.4, lw=0.0)
        hot = np.flatnonzero(keep & (e >= 1e-3))
        for lo, hi in ((1e-3, 0.03), (0.03, 0.3), (0.3, 3.0), (3.0, 1e9)):
            m = hot[(e[hot] >= lo) & (e[hot] < hi)]
            if len(m):
                pt.surface(q[m], mc.to_hex(e_colour(np.array([math.sqrt(lo * min(hi, 60.0))]))[0]), 0.95, lw=0.0, shade=False)
    mask = tracker_mask(ev, 0.3, 0.3, 0.5)
    idx, col, lw = gv.track_style(ev, mask)
    col[:, 3] *= 0.6
    a, b = ev.segments(idx)
    pt.line(a, b, col, lw=lw * 0.5)
    pt.paint(ax)
    # right: one electromagnetic shower from close by
    j, ctr, etot = _best_shower(ev, sim.EB)
    ax2 = fig.add_axes([0.575, 0.03, 0.415, 0.8], facecolor="#0a0d16")
    u = ctr / np.linalg.norm(ctr)
    side = np.cross(u, [0, 0, 1.0]); side /= np.linalg.norm(side)
    eye = ctr - 30.0 * u + 62.0 * side + 26.0 * np.cross(side, u)
    cam2 = gv.Camera(eye, ctr + 2.0 * u, up=np.cross(side, u), fov=34.0, aspect=0.415 * 14.4 / (0.8 * 7.2), near=1.0)
    p2 = gv.Painter(cam2)
    c = det.raw["eb"]
    d = np.linalg.norm(c.mean(axis=1) - ctr, axis=1)
    near = d < 16.0
    p2.surface(gv.box_faces(c[near]), gv.DET_COL["eb"], 0.045, edge="#7fe6cc", lw=0.35, shade=False)
    v = ev.vox[(ev.vox[:, 0] == sim.EB) & (np.linalg.norm(ev.vox[:, 1:4] - ctr, axis=1) < 22.0)]
    o = np.argsort(v[:, 4])
    p2.dot(v[o, 1:4], e_colour(v[o, 4], 1e-4, 3.0), size=2.0 + 26.0 * np.clip(np.log10(np.maximum(v[o, 4], 1e-4) / 1e-4), 0, 4.5) / 4.5, scale=False)
    m2 = ev.seg_prim == j
    idx, col, lw = gv.track_style(ev, m2 & ~ev.outer[ev.seg_trk])
    a, b = ev.segments(idx)
    p2.line(a, b, col, lw=lw * 1.3)
    p2.paint(ax2)
    for sp in ax2.spines.values():
        sp.set_color("#59606c")
    h = ev.hits
    pid = int(h["prim_id"][j]); p4 = h["prim_p4"][j]
    n_hot = int((ce["eb"] > 1e-3).sum() + (ce["ee"] > 1e-3).sum())
    header(fig, "The crystals inside Geant4: where the electromagnetic energy lands",
           [f"61 200 + 14 648 lead-tungstate crystals (front faces, one third removed); {n_hot} received more than 1 MeV:", f"EB {ce['eb'].sum():.1f} GeV, EE {ce['ee'].sum():.1f} GeV"])
    fig.text(0.58, 0.875, f"one shower from close by: {sim.dec.name(pid)} of {p4[0]:.1f} GeV, {etot:.1f} GeV in the crystals", color=TXT, fontsize=11, fontweight="bold")
    fig.text(0.58, 0.845, f"{len(v)} deposits on a {gv_vox():.1f} cm grid; wire frames: the real crystals, 2.2 x 2.2 x 23 cm", color="#aab2c0", fontsize=9.5)
    cax = fig.add_axes([0.03, 0.06, 0.2, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mc.LogNorm(1e-3, 30.0), cmap=mc.ListedColormap(E_CMAP(np.linspace(0.25, 1, 128)))), cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=TXT, labelsize=9); cb.set_label("energy in a crystal [GeV]", color=TXT, fontsize=9.5)
    fig.savefig(out, dpi=dpi, facecolor=gv.BG)
    plt.close(fig)


def gv_vox():
    return 0.5


def page_hcal(det, ev, fmap, out, dpi):
    fig = plt.figure(figsize=(14.4, 7.2), dpi=dpi, facecolor=gv.BG)
    te = tile_energy(det, ev)
    ax = fig.add_axes([0.0, 0.0, 0.56, 0.9], facecolor=gv.BG)
    cam = gv.Camera.orbit((0, -20, 0), 1950.0, 66.0, 21.0, fov=36.0, aspect=0.56 * 14.4 / (0.9 * 7.2))
    pt = gv.Painter(cam)
    cutf = lambda x, y, z: (np.degrees(np.arctan2(y, x)) > -5.0) & (np.degrees(np.arctan2(y, x)) < 115.0)
    cut = lambda x, y, z: cutf(x, y, z) & (z > -560.0)
    for name in ("hb", "he"):
        pt.surface(det.quads(name, cut), gv.DET_COL[name], 0.06, edge="#c79a45", lw=0.12, shade=False)
    pt.surface(det.quads("eb", cut)[::2], gv.DET_COL["eb"], 0.12, lw=0.0, shade=False)
    pt.surface(det.quads("ee", cut)[::2], gv.DET_COL["ee"], 0.12, lw=0.0, shade=False)
    hv = ev.vox[np.isin(ev.vox[:, 0], (sim.HB, sim.HE, sim.HO))]
    hv = hv[np.argsort(hv[:, 4])]
    pt.dot(hv[:, 1:4], e_colour(hv[:, 4], 1e-5, 0.03), size=1.5 + 14.0 * np.clip(np.log10(np.maximum(hv[:, 4], 1e-5) / 1e-5), 0, 3.5) / 3.5, scale=False)
    mask = tracker_mask(ev, 0.5, 1.0, 1.0)
    idx, col, lw = gv.track_style(ev, mask)
    col[:, 3] *= 0.6
    a, b = ev.segments(idx)
    pt.line(a, b, col, lw=lw * 0.5)
    pt.paint(ax)
    # right: one hadronic shower: crystals in front, tiles behind
    code = sim.HB if (ev.vox[:, 0] == sim.HB).any() else sim.HE
    j, ctr, etot = _best_shower(ev, code)
    ax2 = fig.add_axes([0.575, 0.03, 0.415, 0.8], facecolor="#0a0d16")
    w = ev.vox[(ev.vox[:, 6] == j) & np.isin(ev.vox[:, 0], (sim.EB, sim.EE, sim.HB, sim.HE))]
    mid = 0.5 * (w[:, 1:4].min(axis=0) + w[:, 1:4].max(axis=0))
    u = mid / np.linalg.norm(mid)
    side = np.cross(u, [0, 0, 1.0]); side /= np.linalg.norm(side)
    span = float(np.linalg.norm(w[:, 1:4].max(axis=0) - w[:, 1:4].min(axis=0)))
    eye = mid + (1.25 * span) * side + 0.35 * span * np.cross(side, u)
    cam2 = gv.Camera(eye, mid, up=np.cross(side, u), fov=36.0, aspect=0.415 * 14.4 / (0.8 * 7.2), near=2.0)
    p2 = gv.Painter(cam2)
    for name in ("hb", "he"):
        q = det.L[name]
        near = np.linalg.norm(det.centre[name] - mid, axis=1) < 0.75 * span
        p2.surface(q[near], gv.DET_COL[name], 0.006, edge="#c79a45", lw=0.2, shade=False)
    c = det.raw["eb"]
    near = np.linalg.norm(c.mean(axis=1) - mid, axis=1) < 0.6 * span
    p2.surface(gv.near_face(c[near]), gv.DET_COL["eb"], 0.06, edge="#4fbfa5", lw=0.1, shade=False)
    o = np.argsort(w[:, 4])
    is_h = np.isin(w[o, 0], (sim.HB, sim.HE))
    p2.dot(w[o, 1:4][~is_h], e_colour(w[o, 4][~is_h], 1e-4, 1.0), size=3.0, scale=False, alpha=0.9)
    p2.dot(w[o, 1:4][is_h], e_colour(w[o, 4][is_h], 1e-5, 0.03), size=3.0 + 20.0 * np.clip(np.log10(np.maximum(w[o, 4][is_h], 1e-5) / 1e-5), 0, 3.5) / 3.5, scale=False)
    idx, col, lw = gv.track_style(ev, (ev.seg_prim == j) & ~ev.outer[ev.seg_trk])
    a, b = ev.segments(idx)
    p2.line(a, b, col, lw=lw * 1.2)
    p2.paint(ax2)
    for sp in ax2.spines.values():
        sp.set_color("#59606c")
    h = ev.hits
    pid = int(h["prim_id"][j]); p4 = h["prim_p4"][j]
    ecal_part = float(w[np.isin(w[:, 0], (sim.EB, sim.EE)), 4].sum())
    n_hot = int((te["hb"] > 1e-4).sum() + (te["he"] > 1e-4).sum())
    header(fig, "The hadron calorimeter inside Geant4: brass you do not see, scintillator you do",
           [f"{len(det.L['hb'])} + {len(det.L['he'])} scintillator volumes (HB, HE; one third removed), each dot a deposit of the showers in the plastic:",
            f"HB {te['hb'].sum():.2f} GeV, HE {te['he'].sum():.2f} GeV, a few per cent of what the showers lost in the brass between the layers"])
    fig.text(0.58, 0.875, f"one hadronic shower: primary {sim.dec.name(pid)}, E = {p4[0]:.1f} GeV", color=TXT, fontsize=11, fontweight="bold")
    fig.text(0.58, 0.845, f"{ecal_part:.1f} GeV in the crystals (small dots), {etot * 1e3:.0f} MeV in the scintillator layers (large dots)", color="#aab2c0", fontsize=9.5)
    cax = fig.add_axes([0.03, 0.06, 0.2, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=mc.LogNorm(1e-5, 0.03), cmap=mc.ListedColormap(E_CMAP(np.linspace(0.25, 1, 128)))), cax=cax, orientation="horizontal")
    cb.ax.tick_params(colors=TXT, labelsize=9); cb.set_label("energy of a deposit in the scintillator (1 cm grid) [GeV]", color=TXT, fontsize=9.5)
    fig.savefig(out, dpi=dpi, facecolor=gv.BG)
    plt.close(fig)


def page_muon(det, ev, fmap, out, dpi):
    fig, ax = gv.new_figure(14.4, 7.2, dpi)
    cam = gv.Camera.orbit((0, -60, 0), 3700.0, 64.0, 20.0, fov=35.0, aspect=2.0)
    pt = gv.Painter(cam)
    cut = gv.quadrant_cut(-20.0, 100.0)
    pt.surface(det.quads("yoke", cut), gv.DET_COL["yoke"], 0.22, edge="#5a1f1b", lw=0.1)
    pt.surface(det.quads("dt", cut), gv.DET_COL["dt"], 0.35, edge="#8a8430", lw=0.1)
    pt.surface(det.quads("csc", cut), gv.DET_COL["csc"], 0.35, edge="#8a8430", lw=0.1)
    pt.surface(det.quads("rpc", cut), gv.DET_COL["rpc"], 0.2, lw=0.05)
    pt.surface(det.quads("gem", cut), gv.DET_COL["gem"], 0.4, lw=0.05)
    ph0, ph1 = math.radians(100.0), math.radians(340.0)
    pt.surface(gv.cylinder(340.0, -615.0, 615.0, ph0, ph1), gv.DET_COL["coil"], 0.25, lw=0.0)
    pt.surface(det.quads("hb", cut), gv.DET_COL["hb"], 0.12, lw=0.0)
    pt.surface(det.quads("he", cut), gv.DET_COL["he"], 0.12, lw=0.0)
    # the showers in the calorimeters as a glow, the trajectories in the tracker and in the muon system
    v = ev.vox[ev.vox[:, 4] > 3e-3]
    pt.dot(v[:, 1:4], e_colour(v[:, 4], 1e-3, 10.0), size=1.5, alpha=0.7, scale=False)
    mask_in = ~ev.outer[ev.seg_trk] & (ev.seg_ek > 0.3)
    idx, col, lw = gv.track_style(ev, mask_in)
    a, b = ev.segments(idx)
    pt.line(a, b, col, lw=lw * 0.5)
    mask = ev.outer[ev.seg_trk]
    idx, col, lw = gv.track_style(ev, mask)
    col[:, 3] = np.maximum(col[:, 3], 0.9)
    a, b = ev.segments(idx)
    pt.line(a, b, col, lw=np.maximum(lw, 1.4))
    h = ev.hits
    if len(h["mu_x"]):
        pt.dot(np.column_stack([h["mu_x"], h["mu_y"], h["mu_z"]]), "#ffffff", size=40.0, marker="*", scale=False)
    pt.paint(ax)
    nmu = len(np.unique(ev.seg_trk[mask & (ev.seg_kind == 4)]))
    ntr = len(np.unique(ev.seg_trk[mask]))
    header(fig, "The muon system inside Geant4: what is left behind the calorimeters",
           [f"steel of the yoke (red) and chambers (yellow), one third removed; {ntr} charged trajectories above 20 MeV were followed out here,",
            f"{nmu} of them muons from pion and kaon decays; {len(h['mu_x'])} hits in the gas of the chambers: the calorimeters and the coil absorbed the rest"])
    kind_legend(fig, ev, mask, y=0.03, extra=[Line2D([], [], ls="none", marker="*", color="white", ms=9, label="muon-chamber hit")])
    fig.savefig(out, dpi=dpi, facecolor=gv.BG)
    plt.close(fig)


FUNC = {"magnet": page_magnet, "tracker": page_tracker, "ecal": page_ecal, "hcal": page_hcal, "muon": page_muon}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", choices=PAGES, nargs="+", default=PAGES)
    ap.add_argument("-o", "--outdir", type=Path, default=HERE / "pages")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    args.outdir.mkdir(exist_ok=True)
    t0 = time.time()
    det, ev, fmap = gv.Detector(), gv.Event(), FieldMap()
    print(f"loaded the detector, {len(ev.T)} trajectories and the field: {time.time() - t0:.1f} s")
    for p in args.page:
        t1 = time.time()
        out = args.outdir / f"geant4_{p}.png"
        FUNC[p](det, ev, fmap, out, args.dpi)
        print(f"{out.name}: {time.time() - t1:.1f} s")


if __name__ == "__main__":
    main()
