#!/usr/bin/env python3
"""The film of chapter 10: the family inside the Geant4 CMS detector, then the crowd, then the sum.

Seven scenes, narrated, no music, all drawn from what Geant4 did (geant4_SIM.py) inside the geometry it tracked
the particles through, with the viewer of g4view_SIM.py:

  1 the detector   CMS as Geant4 has it, built from the inside out; then the field lines of the magnet
  2 the tracker    the clock runs: helices grow from the collision point and leave hits; then, from close by,
                   real examples out of this event: a looper, a decay in flight, a photon conversion, an
                   electron radiating, a nuclear interaction
  3 the crystals   the electromagnetic showers light up the ECAL; one shower from close by, deposit by deposit
  4 the brass      the hadronic showers in the scintillator layers of the HCAL
  5 the iron       what is left behind the calorimeters: the muon system
  6 the crowd      the same detector for the pileup: the 63 collisions of our bunch crossing, then all sixteen
                   crossings (the steps are those of scenes 2 to 5 and are not shown again)
  7 the sum        the r-z page of plot_simhits_SIM.py, first the family alone, then with the crowd, with the
                   directions of the b jets marked

Every number of the narration is read from geant4_SIM_signal.json and add_hits_SIM.json; large numbers are
written and spoken short (13K, 1.4M).

Narration: as in the other chapter films (example/soundtrack.py): one line per phase (--print-script), edge-tts
or the macOS `say` voice, clips cached in build/narration/SIM/, an .srt next to the .mp4.

  python3 animate_standalone_SIM.py --print-script
  python3 animate_standalone_SIM.py --stills animate_standalone_SIM_stills.pdf --no-video     # the key frames
  python3 animate_standalone_SIM.py --fast -o /tmp/preview.mp4                               # a quick look
  python3 animate_standalone_SIM.py --jobs 8                                                 # the film, 1920x1080
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
import display_geant4_SIM as disp  # noqa: E402
import g4view_SIM as gv  # noqa: E402
import standalone_SIM as sim  # noqa: E402
from field_map_SIM import FieldMap  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent))
import soundtrack as snd  # noqa: E402

FIGSIZE = (12.8, 7.2)
ASPECT = 16 / 9
TXT, TXT2 = "#e8eaf0", "#aab2c0"
SCENE_NAMES = {1: "detector", 2: "tracker", 3: "ecal", 4: "hcal", 5: "muon", 6: "pileup", 7: "sum"}
PU_COL = {"early": "#d6d04a", "intime": "#6aa8ff", "late": "#ff7a7a"}
SHORT = snd.short_number
PAGES = HERE / "pages"
BG_DPI = 150


# ---------------------------------------------------------------------------
# what the scenes share
# ---------------------------------------------------------------------------
class World:
    """The detector, the event, the field and the examples picked out of the event."""

    def __init__(self):
        self.det, self.ev, self.fmap = gv.Detector(), gv.Event(), FieldMap()
        self.G = json.loads((HERE / "geant4_SIM_signal.json").read_text())
        self.N = json.loads((HERE / "add_hits_SIM.json").read_text())
        ev = self.ev
        T = ev.T
        self.start = np.searchsorted(ev.Ptrk, np.arange(len(T)))
        self.stop = np.searchsorted(ev.Ptrk, np.arange(len(T)), side="right")
        order = np.argsort(T[:, 2], kind="stable")
        par = T[order, 2].astype(int)
        self.kids_order, self.kids_lo, self.kids_hi = order, np.searchsorted(par, np.arange(len(T))), np.searchsorted(par, np.arange(len(T)), side="right")
        self.lines = gv.field_lines(self.fmap, levels=11)
        self.ce = disp.crystal_energy(self.det, ev)
        self.examples = self.pick_examples()
        self._bg: Dict[tuple, np.ndarray] = {}

    def kids(self, i):
        return self.kids_order[self.kids_lo[i]:self.kids_hi[i]]

    def family(self, i, depth=3):
        out, front = [i], [i]
        for _ in range(depth):
            front = [int(c) for p in front for c in self.kids(p)]
            out += front
        return np.array(out, dtype=int)

    def points(self, i):
        return self.ev.P[self.start[i]:self.stop[i]]

    def track_at(self, prim, pid, pos, tol=0.05):
        """The trajectory of a particle of this kind of this primary that passes the point pos."""
        T, ev = self.ev.T, self.ev
        best, bd = -1, 1e9
        for i in np.flatnonzero((T[:, 0] == prim) & (T[:, 1] == pid)):
            p = self.points(i)
            if len(p):
                d = float(np.min(np.linalg.norm(p[:, :3] - pos, axis=1)))
                if d < bd:
                    best, bd = int(i), d
        return best if bd < max(tol, 0.5) else -1

    def pick_examples(self) -> dict:
        ev = self.ev
        V, names = ev.V, ev.proc_names
        r = np.hypot(V[:, 2], V[:, 3])
        ex = {}

        def best(proc, m, key):
            idx = np.flatnonzero((V[:, 0] == names.index(proc)) & m)
            if not len(idx):
                return None
            for i in idx[np.argsort(-key[idx])]:
                par = self.track_at(int(V[i, 1]), int(V[i, 5]), V[i, 2:5])
                if par >= 0 and len(self.kids(par)):
                    return dict(v=V[i], parent=par, pos=V[i, 2:5].astype(float), t=float(V[i, 8]))
            return None

        central = (r > 4.0) & (r < 90.0) & (np.abs(V[:, 4]) < 110.0)
        ex["decay"] = best("decay", central & (V[:, 5] == 310), V[:, 6]) or best("decay", central & (np.abs(V[:, 5]) > 100), V[:, 6])
        ex["conversion"] = best("conversion", central & (V[:, 6] > 1.0), V[:, 6])
        ex["nuclear"] = best("nuclear interaction", central & (V[:, 6] > 2.0) & (np.abs(V[:, 5]) < 4000), V[:, 7])
        # the electron of a conversion that radiates most
        brem = names.index("bremsstrahlung")
        T = ev.T
        e_trk = np.flatnonzero((np.abs(T[:, 1]) == 11) & (T[:, 8] > 1.0) & (T[:, 9] == 0) & (np.hypot(T[:, 4], T[:, 5]) < 60.0) & (np.abs(T[:, 6]) < 110.0))
        nb = np.array([int(((T[self.kids(i), 3] == brem) & (T[self.kids(i), 8] > 0.05)).sum()) if len(self.kids(i)) else 0 for i in e_trk])
        if len(e_trk) and nb.max() > 0:
            i = int(e_trk[np.argmax(nb + 1e-3 * T[e_trk, 8])])
            p = self.points(i)
            ex["brems"] = dict(parent=i, pos=p[len(p) // 2, :3].astype(float), t=float(T[i, 7]), n=int(nb.max()))
        # a looper: a charged primary of low pT with the longest path
        h = ev.hits
        pt = np.hypot(h["prim_p4"][:, 1], h["prim_p4"][:, 2])
        eta = np.arcsinh(h["prim_p4"][:, 3] / np.maximum(pt, 1e-9))
        cand = [i for i in np.flatnonzero(T[:, 2] < 0) if abs(int(T[i, 1])) in (211, 321, 2212) and 0.15 < pt[int(T[i, 0])] < 0.6 and abs(eta[int(T[i, 0])]) < 0.8]
        if cand:
            i = int(max(cand, key=lambda j: self.stop[j] - self.start[j]))
            p = self.points(i)
            ex["looper"] = dict(parent=i, pos=p[:, :3].mean(axis=0).astype(float), t=float(T[i, 7]), pt=float(pt[int(T[i, 0])]))
        return ex

    def background(self, key, cam, fill):
        """The static part of a view, drawn once per process and kept as an image."""
        if key not in self._bg:
            fig = plt.figure(figsize=FIGSIZE, dpi=BG_DPI, facecolor=gv.BG)
            ax = fig.add_axes([0, 0, 1, 1], facecolor=gv.BG)
            pt = gv.Painter(cam)
            fill(pt)
            pt.paint(ax)
            fig.canvas.draw()
            self._bg[key] = np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
        return self._bg[key]


def pileup_display(path: Path = None, n_it: int = 30000, n_oot: int = 110000) -> dict:
    """What scene 6 draws of the crowd, cut down once and kept in a small file (../g4pileup_display.npz): a
    random sample of the tracker hits, the calorimeter cells above a threshold per bunch crossing, the muon
    hits, and the totals per bunch crossing (index 0 is bx -12, index 12 is our own crossing)."""
    path = path or HERE.parent / "g4pileup_display.npz"
    src = [HERE.parent / "simhits_intime.npz", HERE.parent / "simhits_oot.npz"]
    if path.exists() and all(path.stat().st_mtime >= s.stat().st_mtime for s in src):
        return dict(np.load(path))
    rng = np.random.default_rng(1)
    out = {}
    n_trk, n_col, e_ecal, e_hcal, n_mu = np.zeros(16), np.zeros(16), np.zeros(16), np.zeros(16), np.zeros(16)
    trk, cal, mu = [], [], []
    for f, nmax in zip(src, (n_it, n_oot)):
        d = np.load(f)
        bx = d["trk_bx"].astype(int)
        n_trk += np.bincount(bx + 12, minlength=16)
        n_col += np.bincount(d["col_bx"].astype(int) + 12, minlength=16)
        n_mu += np.bincount(d["mu_bx"].astype(int) + 12, minlength=16)
        idx = rng.choice(len(bx), min(len(bx), nmax), replace=False)
        trk.append(np.column_stack([d["trk_x"][idx], d["trk_y"][idx], d["trk_z"][idx], d["trk_t"][idx], bx[idx]]))
        mu.append(np.column_stack([d["mu_x"], d["mu_y"], d["mu_z"], d["mu_t"], d["mu_bx"]]))
        det = d["cal_code"] // sim.CODE
        cbx = d["cal_bx"].astype(int)
        e = d["cal_e"].astype(float)
        e_ecal += np.bincount(cbx[np.isin(det, (sim.EB, sim.EE))] + 12, weights=e[np.isin(det, (sim.EB, sim.EE))], minlength=16)
        e_hcal += np.bincount(cbx[np.isin(det, (sim.HB, sim.HE, sim.HO))] + 12, weights=e[np.isin(det, (sim.HB, sim.HE, sim.HO))], minlength=16)
        m = np.isin(det, (sim.EB, sim.EE, sim.HB, sim.HE))
        key = d["cal_code"][m] * 16 + (cbx[m] + 12)                       # one entry per (cell, bunch crossing)
        uk, inv = np.unique(key, return_inverse=True)
        es = np.bincount(inv, weights=e[m])
        code, b = uk // 16, uk % 16 - 12
        dd = code // sim.CODE
        keep = np.where(np.isin(dd, (sim.EB, sim.EE)), es > 0.5, es > 0.004)
        _, x, y, z = sim.cell_position(code[keep])
        cal.append(np.column_stack([x, y, z, es[keep], b[keep], dd[keep]]))
    out = dict(trk=np.concatenate(trk).astype(np.float32), cal=np.concatenate(cal).astype(np.float32), mu=np.concatenate(mu).astype(np.float32),
               n_trk=n_trk, n_col=n_col, e_ecal=e_ecal, e_hcal=e_hcal, n_mu=n_mu)
    np.savez_compressed(path, **out)
    return out


def chrome(fig, number, title, caption, clock=None, right=None):
    head = f"Scene {number}: {title}"
    fig.text(0.015, 0.955, head, fontsize=15, weight="bold", color=TXT, va="center")
    if len(head) < 64:                                     # a long title needs the whole line
        fig.text(0.985, 0.955, right or "Geant4 in the CMS geometry (geant4_SIM.py)", fontsize=9.5, color=TXT2, ha="right", va="center")
    if caption:
        fig.text(0.015, 0.03, caption, fontsize=10.5, color=TXT, va="bottom")
    if clock:
        fig.text(0.985, 0.9, clock, fontsize=20, color=TXT, ha="right", va="center", weight="bold")


def stage(fig, bg=None):
    fig.clf()
    fig.patch.set_facecolor(gv.BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=gv.BG)
    ax.set_xlim(-ASPECT, ASPECT); ax.set_ylim(-1, 1); ax.axis("off")
    # the dark sheet behind everything (the video writer paints the figure white, and an axes switched off has no patch)
    ax.add_patch(plt.Rectangle((-ASPECT, -1), 2 * ASPECT, 2, facecolor=gv.BG, edgecolor="none", zorder=-5))
    if bg is not None:
        ax.imshow(bg, extent=(-ASPECT, ASPECT, -1, 1), aspect="auto", zorder=0, interpolation="bilinear")
    return ax


def counter(fig, x, y, value, label, color=TXT):
    fig.text(x, y, value, fontsize=17, color=color, va="center", weight="bold")
    fig.text(x, y - 0.035, label, fontsize=9.5, color=TXT2, va="center")


class Scene:
    """A scene is a list of segments (phase key, from, to, seconds at the nominal speed): `at` gives the phase and
    the value between from and to at a frame index."""
    number, title = 0, ""

    def __init__(self, W: World, segments):
        self.W = W
        self.seg, k = [], 0
        for key, a, b, seconds in segments:
            n = int(round(seconds * snd.NOMINAL_FPS))
            self.seg.append((key, k, k + n, a, b))
            k += n
        self.nframes = k

    def phases(self) -> List[Tuple[str, int, int]]:
        return [(key, k0, k1) for key, k0, k1, _, _ in self.seg]

    def at(self, k: float) -> Tuple[str, float, float]:
        for key, k0, k1, a, b in self.seg:
            if k < k1 or (key, k0, k1, a, b) == self.seg[-1]:
                u = min(max((k - k0) / max(k1 - 1 - k0, 1), 0.0), 1.0)
                return key, u, a + (b - a) * u
        raise RuntimeError("no segments")

    def draw(self, fig, k: float):
        key, u, val = self.at(k)
        self.render(fig, key, u, val)


# ---------------------------------------------------------------------------
# scene 1: the detector and the field
# ---------------------------------------------------------------------------
class DetectorScene(Scene):
    number, title = 1, "the detector, as Geant4 has it"
    ORDER = [("pix", 0.85, 0.0), ("trk", 0.5, 0.0), ("eb", 0.5, 0.0), ("ee", 0.5, 0.0), ("hb", 0.35, 0.1), ("he", 0.35, 0.1), ("coil", 0.5, 0.1),
             ("yoke", 0.5, 0.15), ("dt", 0.22, 0.1), ("csc", 0.3, 0.1), ("rpc", 0.12, 0.05), ("hf", 0.5, 0.2)]
    WORDS = [("pix", "pixel detector: 1856 modules"), ("trk", "strip tracker: 15 148 modules"), ("eb", "ECAL: 75 848 crystals"), ("hb", "HCAL: brass and scintillator"),
             ("coil", "the coil: 3.8 T"), ("yoke", "the return yoke: 10 000 t of steel"), ("dt", "muon chambers"), ("hf", "forward calorimeter")]

    def __init__(self, W):
        super().__init__(W, [("build", 0.0, 1.0, 9.0), ("field", 0.0, 1.0, 8.0)])
        self.cut = gv.quadrant_cut(-35.0, 95.0)

    def camera(self, u=1.0):
        return gv.Camera.orbit((0, -60, 0), 4700.0 - 500.0 * u, 48.0 + 14.0 * u, 21.0, fov=34.0, aspect=ASPECT)

    def fill(self, pt, upto=1.0):
        det, cut = self.W.det, self.cut
        n = len(self.ORDER)
        for i, (name, alpha, lw) in enumerate(self.ORDER):
            a = np.clip(upto * n - i, 0.0, 1.0)
            if a <= 0:
                continue
            if name == "coil":
                ph0, ph1 = math.radians(95.0), math.radians(325.0)
                q = np.concatenate([gv.cylinder(320.0, -615.0, 615.0, ph0, ph1), gv.cylinder(340.0, -615.0, 615.0, ph0, ph1),
                                    gv.annulus(320.0, 340.0, 615.0, ph0, ph1), gv.annulus(320.0, 340.0, -615.0, ph0, ph1)])
            else:
                q = det.quads(name, None if name in ("pix", "trk", "hf") else cut)
                if name in ("eb", "ee", "trk"):
                    q = q[::3 if name == "eb" else 2]
            pt.surface(q, gv.DET_COL[name], alpha * a, lw=lw)

    def render(self, fig, key, u, val):
        W = self.W
        if key == "build":
            cam = self.camera(u)
            ax = stage(fig)
            pt = gv.Painter(cam)
            self.fill(pt, upto=u * 1.05)
            pt.paint(ax)
            n = len(self.ORDER)
            shown = [w for nm, w in self.WORDS if u * 1.05 * n - [o[0] for o in self.ORDER].index(nm) > 0.3]
            for i, w in enumerate(shown):
                fig.text(0.015, 0.86 - 0.04 * i, w, fontsize=11, color=TXT if i == len(shown) - 1 else TXT2, va="center")
            chrome(fig, 1, self.title, "The Run-3 simulation geometry of CMS (cmsSimGeo2021.gdml): 3398 kinds of volume, 2.1 million placed volumes, 490 materials. One third of the azimuth is removed.")
            return
        cam = self.camera(1.0)
        ax = stage(fig, W.background(("det", 1), cam, lambda p: self.fill(p, 1.0)))
        pt = gv.Painter(cam)
        cmap = disp.B_CMAP
        for phi, a in ((-35.0, 1.0), (95.0, 1.0), (215.0, 0.45)):
            c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
            for j, L in enumerate(W.lines):
                n = max(2, int(len(L) * min(1.0, u * 1.15)))
                P = np.column_stack([L[:n, 1] * c, L[:n, 1] * s, L[:n, 0]])
                col = cmap(np.clip(0.5 * (L[:n - 1, 2] + L[1:n, 2]) / 4.0, 0, 1))
                col[:, 3] = a
                pt.line(P[:-1], P[1:], col, lw=1.4)
        pt.paint(ax)
        b = W.fmap.b
        fig.text(0.015, 0.86, "field lines, coloured by |B|", fontsize=11, color=TXT, va="center")
        for i, (lab, val_, colv) in enumerate((("centre", b(0, 0, 0)[2], 3.8), ("barrel yoke", b(475.0, 0, 0, True)[2], 2.0), ("endcap disk", math.hypot(*b(400.0, 0, 760.0, True)[::2]), 2.3))):
            fig.text(0.015, 0.81 - 0.045 * i, f"{val_:+.1f} T", fontsize=15, color=cmap(colv / 4.0), va="center", weight="bold")
            fig.text(0.09, 0.81 - 0.045 * i, lab, fontsize=10, color=TXT2, va="center")
        chrome(fig, 1, "the magnet: coil, return yoke and field lines",
               f"The field of field_map_SIM.py: Ampere's law for this coil and this iron, {W.fmap.current / 1e6:.1f} million ampere-turns. Inside the coil the lines run along the beam; the steel brings them back.")


# ---------------------------------------------------------------------------
# scene 2: the tracker
# ---------------------------------------------------------------------------
class TrackerScene(Scene):
    number, title = 2, "the tracker: helices and hits"

    def __init__(self, W):
        seg = [("helix", 0.0, 6.0, 9.0), ("hits", 6.0, 30.0, 8.0)]
        self.close = {}
        for key, name in (("looper", "looper"), ("decay", "decay"), ("conversion", "conversion"), ("brems", "brems"), ("nuclear", "nuclear")):
            if W.examples.get(name):
                seg.append((key, 0.0, 1.0, 7.0))
        super().__init__(W, seg)
        ev = W.ev
        self.mask = disp.tracker_mask(ev)
        self.cut = lambda x, y, z: (np.degrees(np.arctan2(y, x)) > -10.0) & (np.degrees(np.arctan2(y, x)) < 110.0)
        self.cam = gv.Camera.orbit((0, -8, 0), 660.0, 68.0, 20.0, fov=36.0, aspect=ASPECT)
        h = ev.hits
        self.hit_xyz = np.column_stack([h["trk_x"], h["trk_y"], h["trk_z"]])
        self.hit_t = h["trk_t"] - ev.t0
        self.hit_sorted = np.sort(self.hit_t)
        self.de_cum = np.cumsum(h["trk_de"][np.argsort(self.hit_t)])

    def fill(self, pt, near=None, radius=None):
        det = self.W.det
        for name, alpha, edge, lw in (("trk", 0.2, "#6aa5e8", 0.18), ("pix", 0.3, "#a9e0ff", 0.12)):
            q, c = det.L[name], det.centre[name]
            if near is None:
                q = det.quads(name, self.cut)
            else:
                q = q[np.linalg.norm(c - near, axis=1) < radius]
            if near is not None:                          # from close by: a few faint frames, so that the example stands alone
                q = q[::2]
            pt.surface(q, gv.DET_COL[name], alpha if near is None else 0.012, edge=edge if near is None else "#35597f", lw=lw if near is None else 0.18, shade=near is None)
        if near is None:
            pt.surface(det.quads("pipe"), gv.DET_COL["pipe"], 0.10, lw=0.0)

    def wide(self, fig, key, tnow):
        W, ev = self.W, self.W.ev
        ax = stage(fig, W.background(("trk", 0), self.cam, self.fill))
        pt = gv.Painter(self.cam)
        idx, col, lw = gv.track_style(ev, self.mask, ev.t0 + tnow, fade=6.0)
        a, b = ev.segments(idx)
        pt.line(a, b, col, lw=lw)
        m = self.hit_t <= tnow
        pt.dot(self.hit_xyz[m], "#ffe14d", size=2.2, alpha=0.9)
        pt.paint(ax)
        n = int(np.searchsorted(self.hit_sorted, tnow, side="right"))
        counter(fig, 0.015, 0.86, SHORT(len(np.unique(ev.seg_trk[idx]))), "trajectories on their way")
        counter(fig, 0.015, 0.76, SHORT(n), "hits in the silicon", "#ffe14d")
        counter(fig, 0.015, 0.66, f"{(self.de_cum[n - 1] if n else 0.0):.2f} GeV", "left in the sensors")
        cap = ("White: charged hadrons, orange: electrons, green: photons, violet: neutral hadrons. Thick = energetic. Light crosses the tracker in four nanoseconds."
               if key == "helix" else "Yellow: one hit per sensor crossed, with its place, its time and the energy lost there. The spirals are electrons of a few MeV.")
        chrome(fig, 2, self.title, cap, clock=f"t = {tnow:4.1f} ns")

    def closeup(self, fig, key, u):
        W, ev = self.W, self.W.ev
        ex = W.examples[key]
        fam = W.family(ex["parent"], depth=3 if key != "looper" else 0)
        pts = np.concatenate([W.points(i)[:, :3] for i in fam if self.W.stop[i] > self.W.start[i]])
        ctr = ex["pos"]
        d = np.linalg.norm(pts - ctr, axis=1)
        span = float(np.clip(np.percentile(d, 85), 12.0, 70.0)) if key != "looper" else float(np.percentile(d, 98)) * 1.15
        par = W.points(ex["parent"])
        axis = np.array([0.0, 0.0, 1.0])
        if key == "looper":
            eye = ctr + np.array([0.35, 0.3, 1.0]) * span * 2.6
            up = (0, 1, 0)
        else:
            direction = par[-1, :3] - par[0, :3]
            direction /= max(np.linalg.norm(direction), 1e-9)
            side = np.cross(direction, axis)
            side = side / np.linalg.norm(side) if np.linalg.norm(side) > 0.2 else np.array([0.0, 1.0, 0.0])
            eye = ctr + side * span * 2.4 + np.cross(side, direction) * span * 0.9
            up = np.cross(side, direction)
        cam = gv.Camera(eye, ctr, up=up, fov=36.0, aspect=ASPECT, near=1.0)
        ax = stage(fig, W.background(("close", key), cam, lambda p: self.fill(p, near=ctr, radius=span * 1.25)))
        pt = gv.Painter(cam)
        # everything else: only the energetic neighbours, and almost invisible
        near_seg = self.mask & (np.linalg.norm(ev.P[ev.seg_a, :3] - ctr, axis=1) < span * 2.0) & ~np.isin(ev.seg_trk, fam) & (ev.T[ev.seg_trk, 8] > 0.3)
        idx, col, lw = gv.track_style(ev, near_seg)
        col[:, 3] *= 0.07
        a, b = ev.segments(idx)
        pt.line(a, b, col, lw=np.minimum(lw * 0.5, 0.8))
        # the example: the clock runs from the birth of the parent to the end of its family
        m = np.isin(ev.seg_trk, fam)
        t_all = ev.seg_t[m]
        t0, t1 = float(t_all.min()), float(np.percentile(t_all, 97 if key != "looper" else 100))
        if key in ("decay", "conversion", "nuclear"):
            t1 = min(t1, ex["t"] + 5.0)                       # the first nanoseconds after the vertex, not the last slow neutron
            t0 = max(t0, ex["t"] - 3.0)
        tnow = t0 + (t1 - t0) * min(1.0, u * 1.1)
        idx, col, lw = gv.track_style(ev, m, tnow)
        col[:, 3] = np.maximum(col[:, 3], 0.9)
        a, b = ev.segments(idx)
        pt.line(a, b, col, lw=np.maximum(lw * 1.8, 1.2))
        hm = np.isin(ev.hits["trk_prim"], [int(ev.T[ex["parent"], 0])]) & (ev.hits["trk_t"] <= tnow) & (np.linalg.norm(self.hit_xyz - ctr, axis=1) < span * 2.5)
        pt.dot(self.hit_xyz[hm], "#ffe14d", size=14.0, alpha=0.95, scale=False)
        if key != "looper" and tnow >= ex["t"] - 1e-3 and key != "brems":
            pt.dot(ctr[None, :], "#ffffff", size=70.0, marker="*", scale=False)
        pt.paint(ax)
        T = ev.T
        name = sim.dec.name(int(T[ex["parent"], 1]))
        kids = W.kids(ex["parent"])
        r = math.hypot(ctr[0], ctr[1])
        if key == "looper":
            R = 100.0 * ex["pt"] / (0.299792458 * 3.8)
            head, lines = "a looper", [f"{name}, pT = {ex['pt']:.2f} GeV", f"radius of the helix: {R:.0f} cm", "it cannot reach the calorimeter at r = 129 cm", f"{int(hm.sum())} hits so far"]
        elif key == "decay":
            head, lines = "a decay in flight", [f"{name}, {T[ex['parent'], 8]:.1f} GeV", f"decays {r:.0f} cm from the beam", "into " + " ".join(sim.dec.name(int(T[i, 1])) for i in kids[:4])]
        elif key == "conversion":
            head, lines = "a photon conversion", [f"photon, {ex['v'][6]:.1f} GeV", f"meets a nucleus at r = {r:.0f} cm", "and becomes an electron and a positron"]
        elif key == "brems":
            head, lines = "bremsstrahlung", [f"{name}, {T[ex['parent'], 8]:.1f} GeV", f"radiates {ex['n']} photons above 50 MeV", "on its way through the silicon (green)"]
        else:
            head, lines = "a nuclear interaction", [f"{name}, {ex['v'][6]:.1f} GeV", f"hits a nucleus at r = {r:.0f} cm", f"{int(ex['v'][7])} particles come out"]
        fig.text(0.015, 0.87, head, fontsize=16, color="#ffd23f", va="center", weight="bold")
        for i, l in enumerate(lines):
            fig.text(0.015, 0.82 - 0.04 * i, l, fontsize=11.5, color=TXT, va="center")
        chrome(fig, 2, "the tracker, from close by: a real example out of this event", "The trajectory in colour is Geant4's; the rest of the event is almost switched off. Faint frames: silicon modules nearby.",
               clock=f"t = {tnow - ev.t0:5.2f} ns")

    def render(self, fig, key, u, val):
        if key in ("helix", "hits"):
            self.wide(fig, key, val)
        else:
            self.closeup(fig, {"looper": "looper", "decay": "decay", "conversion": "conversion", "brems": "brems", "nuclear": "nuclear"}[key], u)


# ---------------------------------------------------------------------------
# scenes 3 and 4: the calorimeters
# ---------------------------------------------------------------------------
class CaloScene(Scene):
    def __init__(self, W, number, title, name, codes, wide_dist, vox_lo, vox_hi, unit_title):
        self.number, self.title = number, title
        super().__init__(W, [("wall", 3.0, 16.0, 8.0), ("shower", 0.0, 1.0, 9.0)])
        self.name, self.codes, self.vlo, self.vhi = name, codes, vox_lo, vox_hi
        ev = W.ev
        self.cam = gv.Camera.orbit((0, -15, 0), wide_dist, 66.0, 21.0, fov=36.0, aspect=ASPECT)
        self.cut = lambda x, y, z: (np.degrees(np.arctan2(y, x)) > -5.0) & (np.degrees(np.arctan2(y, x)) < 115.0) & (z > -560.0)
        v = ev.vox[np.isin(ev.vox[:, 0], codes)]
        self.v = v[np.argsort(v[:, 5])]
        self.vt = self.v[:, 5] - ev.t0
        self.ecum = np.cumsum(self.v[:, 4])
        self.mask = disp.tracker_mask(ev, 0.5, 0.5, 1.0)
        code = codes[0] if (ev.vox[:, 0] == codes[0]).any() else codes[-1]
        self.j, self.ctr, self.etot = disp._best_shower(ev, code)
        w = ev.vox[(ev.vox[:, 6] == self.j) & np.isin(ev.vox[:, 0], (sim.EB, sim.EE, sim.HB, sim.HE))]
        if name == "ecal":
            w = w[np.linalg.norm(w[:, 1:4] - self.ctr, axis=1) < 22.0]
        self.w = w
        self.axis = self.ctr / np.linalg.norm(self.ctr)
        self.depth = (w[:, 1:4] - self.ctr) @ self.axis

    def fill_wide(self, pt):
        det = self.W.det
        if self.name == "ecal":
            for nm in ("eb", "ee"):
                pt.surface(det.quads(nm, self.cut), gv.DET_COL[nm], 0.4, lw=0.0)
        else:
            for nm in ("hb", "he"):
                pt.surface(det.quads(nm, self.cut), gv.DET_COL[nm], 0.06, edge="#c79a45", lw=0.12, shade=False)
            for nm in ("eb", "ee"):
                pt.surface(det.quads(nm, self.cut)[::2], gv.DET_COL[nm], 0.12, lw=0.0, shade=False)

    def size(self, e):
        return 1.5 + 14.0 * np.clip(np.log10(np.maximum(e, self.vlo) / self.vlo), 0, 3.5) / 3.5

    def render(self, fig, key, u, val):
        W, ev = self.W, self.W.ev
        if key == "wall":
            ax = stage(fig, W.background((self.name, 0), self.cam, self.fill_wide))
            pt = gv.Painter(self.cam)
            idx, col, lw = gv.track_style(ev, self.mask, ev.t0 + val, fade=8.0)
            col[:, 3] *= 0.6
            a, b = ev.segments(idx)
            pt.line(a, b, col, lw=lw * 0.5)
            n = int(np.searchsorted(self.vt, val, side="right"))
            v = self.v[:n]
            keep = v[:, 4] > self.vlo * 3
            pt.dot(v[keep, 1:4], disp.e_colour(v[keep, 4], self.vlo, self.vhi), size=self.size(v[keep, 4]) * (0.5 if self.name == "ecal" else 1.0), scale=False)
            pt.paint(ax)
            unit = "GeV in the crystals" if self.name == "ecal" else "GeV in the scintillator"
            counter(fig, 0.015, 0.86, f"{(self.ecum[n - 1] if n else 0.0):,.1f}" if self.name == "ecal" else f"{(self.ecum[n - 1] if n else 0.0):,.2f}", unit, "#ffb347")
            counter(fig, 0.015, 0.76, SHORT(n), "deposits so far (3D grid)")
            cap = ("Every dot is energy left by a shower in the lead tungstate, drawn where Geant4 put it. The crystals are the absorber and the detector at once."
                   if self.name == "ecal" else
                   "The brass between the layers is not drawn and not read out. Every dot is energy left in a scintillator layer: a few per cent of the shower.")
            chrome(fig, self.number, self.title, cap, clock=f"t = {val:4.1f} ns")
            return
        # one shower from close by, revealed along its depth
        det = W.det
        ctr = self.ctr
        w = self.w
        span = float(np.linalg.norm(w[:, 1:4].max(axis=0) - w[:, 1:4].min(axis=0))) if self.name == "hcal" else 30.0
        mid = 0.5 * (w[:, 1:4].min(axis=0) + w[:, 1:4].max(axis=0)) if self.name == "hcal" else ctr + 2.0 * self.axis
        side = np.cross(self.axis, [0, 0, 1.0]); side /= np.linalg.norm(side)
        upv = np.cross(side, self.axis)
        eye = mid + (1.25 * span) * side + 0.35 * span * upv if self.name == "hcal" else ctr - 30.0 * self.axis + 62.0 * side + 26.0 * upv
        cam = gv.Camera(eye, mid, up=upv, fov=36.0, aspect=ASPECT, near=1.0)

        def fill(pt):
            if self.name == "ecal":
                c = det.raw["eb"]
                pt.surface(gv.box_faces(c[np.linalg.norm(c.mean(axis=1) - ctr, axis=1) < 16.0]), gv.DET_COL["eb"], 0.045, edge="#7fe6cc", lw=0.35, shade=False)
            else:
                for nm in ("hb", "he"):
                    q = det.L[nm]
                    pt.surface(q[np.linalg.norm(det.centre[nm] - mid, axis=1) < 0.75 * span], gv.DET_COL[nm], 0.006, edge="#c79a45", lw=0.2, shade=False)
                c = det.raw["eb"]
                pt.surface(gv.near_face(c[np.linalg.norm(c.mean(axis=1) - mid, axis=1) < 0.6 * span]), gv.DET_COL["eb"], 0.06, edge="#4fbfa5", lw=0.1, shade=False)

        ax = stage(fig, W.background((self.name, 1), cam, fill))
        pt = gv.Painter(cam)
        dmin, dmax = float(self.depth.min()), float(self.depth.max())
        dnow = dmin + (dmax - dmin) * min(1.0, u * 1.15)
        m = self.depth <= dnow
        ww = w[m]
        o = np.argsort(ww[:, 4])
        ww = ww[o]
        if self.name == "ecal":
            pt.dot(ww[:, 1:4], disp.e_colour(ww[:, 4], 1e-4, 3.0), size=2.0 + 26.0 * np.clip(np.log10(np.maximum(ww[:, 4], 1e-4) / 1e-4), 0, 4.5) / 4.5, scale=False)
        else:
            is_h = np.isin(ww[:, 0], (sim.HB, sim.HE))
            pt.dot(ww[~is_h, 1:4], disp.e_colour(ww[~is_h, 4], 1e-4, 1.0), size=3.0, scale=False, alpha=0.9)
            pt.dot(ww[is_h, 1:4], disp.e_colour(ww[is_h, 4], 1e-5, 0.03), size=3.0 + 20.0 * np.clip(np.log10(np.maximum(ww[is_h, 4], 1e-5) / 1e-5), 0, 3.5) / 3.5, scale=False)
        idx, col, lw = gv.track_style(ev, (ev.seg_prim == self.j) & ~ev.outer[ev.seg_trk])
        a, b = ev.segments(idx)
        pt.line(a, b, col, lw=lw * 1.3)
        pt.paint(ax)
        h = ev.hits
        pid, p4 = int(h["prim_id"][self.j]), h["prim_p4"][self.j]
        e_c = float(ww[np.isin(ww[:, 0], (sim.EB, sim.EE)), 4].sum())
        e_s = float(ww[np.isin(ww[:, 0], (sim.HB, sim.HE)), 4].sum())
        fig.text(0.015, 0.87, "one shower, deposit by deposit", fontsize=16, color="#ffd23f", va="center", weight="bold")
        fig.text(0.015, 0.82, f"primary: {sim.dec.name(pid)}, {p4[0]:.1f} GeV", fontsize=11.5, color=TXT, va="center")
        if self.name == "ecal":
            counter(fig, 0.015, 0.74, f"{e_c:.1f} GeV", "in the crystals so far", "#ffb347")
            counter(fig, 0.015, 0.64, SHORT(len(ww)), "deposits on a 5 mm grid")
            cap = "Frames: the real crystals, 2.2 cm wide and 23 cm long (26 radiation lengths). An electromagnetic shower fills a handful of them."
        else:
            counter(fig, 0.015, 0.74, f"{e_c:.1f} GeV", "in the crystals in front", "#ffb347")
            counter(fig, 0.015, 0.64, f"{e_s * 1e3:.0f} MeV", "in the scintillator layers", "#ffb347")
            cap = "Small dots: the crystals in front. Large dots: the scintillator layers, 9 mm of plastic every 6 cm of brass. A hadronic shower is longer, wider and lumpier."
        chrome(fig, self.number, self.title + ", from close by", cap)


# ---------------------------------------------------------------------------
# scene 5: the muon system
# ---------------------------------------------------------------------------
class MuonScene(Scene):
    number, title = 5, "behind the calorimeters: the iron and the muon chambers"

    def __init__(self, W):
        super().__init__(W, [("iron", 4.0, 80.0, 9.0)])
        self.cam = gv.Camera.orbit((0, -60, 0), 3700.0, 64.0, 20.0, fov=35.0, aspect=ASPECT)
        ev = W.ev
        self.v = ev.vox[ev.vox[:, 4] > 3e-3]

    def fill(self, pt):
        det = self.W.det
        cut = gv.quadrant_cut(-20.0, 100.0)
        pt.surface(det.quads("yoke", cut), gv.DET_COL["yoke"], 0.22, edge="#5a1f1b", lw=0.1)
        for nm, a in (("dt", 0.35), ("csc", 0.35), ("rpc", 0.2), ("gem", 0.4)):
            pt.surface(det.quads(nm, cut), gv.DET_COL[nm], a, edge="#8a8430" if nm in ("dt", "csc") else None, lw=0.1)
        pt.surface(gv.cylinder(340.0, -615.0, 615.0, math.radians(100.0), math.radians(340.0)), gv.DET_COL["coil"], 0.25, lw=0.0)
        pt.surface(det.quads("hb", cut), gv.DET_COL["hb"], 0.12, lw=0.0)
        pt.surface(det.quads("he", cut), gv.DET_COL["he"], 0.12, lw=0.0)

    def render(self, fig, key, u, val):
        W, ev = self.W, self.W.ev
        ax = stage(fig, W.background(("mu", 0), self.cam, self.fill))
        pt = gv.Painter(self.cam)
        v = self.v[self.v[:, 5] - ev.t0 <= val]
        pt.dot(v[:, 1:4], disp.e_colour(v[:, 4], 1e-3, 10.0), size=1.5, alpha=0.7, scale=False)
        idx, col, lw = gv.track_style(ev, disp.tracker_mask(ev, 0.3, 1.0, 1.0), ev.t0 + val)
        a, b = ev.segments(idx)
        pt.line(a, b, col, lw=lw * 0.5)
        mask = ev.outer[ev.seg_trk]
        idx, col, lw = gv.track_style(ev, mask, ev.t0 + val)
        col[:, 3] = np.maximum(col[:, 3], 0.9)
        a, b = ev.segments(idx)
        pt.line(a, b, col, lw=np.maximum(lw, 1.4))
        h = ev.hits
        if len(h["mu_x"]):
            m = h["mu_t"] - ev.t0 <= val
            pt.dot(np.column_stack([h["mu_x"], h["mu_y"], h["mu_z"]])[m], "#ffffff", size=40.0, marker="*", scale=False)
        pt.paint(ax)
        ntr = len(np.unique(ev.seg_trk[idx]))
        nmu = len(np.unique(ev.seg_trk[idx][ev.seg_kind[idx] == 4]))
        counter(fig, 0.015, 0.86, f"{ntr}", "charged trajectories above 20 MeV out here")
        counter(fig, 0.015, 0.76, f"{nmu}", "of them muons", gv.KIND_COL[4])
        counter(fig, 0.015, 0.66, f"{len(h['mu_x'])}", "hits in the gas of the chambers")
        chrome(fig, 5, self.title, "Red: the steel of the return yoke. Yellow: the muon chambers between its layers. The glow in the middle is the energy the calorimeters kept.",
               clock=f"t = {val:4.0f} ns")


# ---------------------------------------------------------------------------
# scene 6: the crowd through the same detector
# ---------------------------------------------------------------------------
class PileupScene(Scene):
    number, title = 6, "the crowd crosses the same detector"

    def __init__(self, W):
        super().__init__(W, [("intime", 0.0, 25.0, 8.0), ("all", -12.0, 3.999, 9.0)])
        self.cam = gv.Camera.orbit((0, -60, 0), 3700.0, 64.0, 20.0, fov=35.0, aspect=ASPECT)
        self.D = pileup_display()
        D = self.D
        self.trk_bx, self.cal_bx, self.mu_bx = D["trk"][:, 4].astype(int), D["cal"][:, 4].astype(int), D["mu"][:, 4].astype(int)
        self.cal_r = np.linalg.norm(D["cal"][:, :3], axis=1)
        self.is_ecal = np.isin(D["cal"][:, 5], (sim.EB, sim.EE))
        self.it_sorted = np.sort(D["trk"][self.trk_bx == 0, 3])
        self.frac0 = (self.trk_bx == 0).sum() / max(D["n_trk"][12], 1)       # the drawn share of the in-time hits

    def fill(self, pt):
        det = self.W.det
        cut = gv.quadrant_cut(-20.0, 100.0)
        pt.surface(det.quads("yoke", cut), gv.DET_COL["yoke"], 0.13, edge="#5a1f1b", lw=0.08)
        for nm, a in (("dt", 0.22), ("csc", 0.22), ("rpc", 0.12), ("gem", 0.25)):
            pt.surface(det.quads(nm, cut), gv.DET_COL[nm], a, lw=0.0)
        pt.surface(gv.cylinder(340.0, -615.0, 615.0, math.radians(100.0), math.radians(340.0)), gv.DET_COL["coil"], 0.15, lw=0.0)
        pt.surface(det.quads("hb", cut), gv.DET_COL["hb"], 0.07, lw=0.0)
        pt.surface(det.quads("he", cut), gv.DET_COL["he"], 0.07, lw=0.0)

    def cells(self, pt, m):
        c = self.D["cal"][m]
        ec = self.is_ecal[m]
        pt.dot(c[ec, :3], disp.e_colour(c[ec, 3], 0.5, 60.0), size=2.0 + 5.0 * np.clip(np.log10(c[ec, 3] / 0.5), 0, 2.2), alpha=0.85, scale=False)
        pt.dot(c[~ec, :3], disp.e_colour(c[~ec, 3], 0.004, 0.5), size=2.5 + 5.0 * np.clip(np.log10(c[~ec, 3] / 0.004), 0, 2.2), alpha=0.85, scale=False)

    def render(self, fig, key, u, val):
        W, D = self.W, self.D
        ax = stage(fig, W.background(("pu", 0), self.cam, self.fill))
        pt = gv.Painter(self.cam)
        trk, mu = D["trk"], D["mu"]
        if key == "intime":
            m = (self.trk_bx == 0) & (trk[:, 3] <= val)
            pt.dot(trk[m, :3], PU_COL["intime"], size=1.6, alpha=0.75, scale=False)
            self.cells(pt, (self.cal_bx == 0) & (self.cal_r / 29.98 <= val))
            mm = (self.mu_bx == 0) & (mu[:, 3] <= val)
            pt.dot(mu[mm, :3], "#ffffff", size=30.0, marker="*", scale=False)
            pt.paint(ax)
            n = int(np.searchsorted(self.it_sorted, val, side="right") / max(self.frac0, 1e-9))
            counter(fig, 0.015, 0.86, f"{int(D['n_col'][12])}", "collisions of our own bunch crossing", PU_COL["intime"])
            counter(fig, 0.015, 0.76, SHORT(min(n, D["n_trk"][12])), "hits in the silicon so far")
            counter(fig, 0.015, 0.66, f"{D['e_ecal'][12] / 1e3 * min(1.0, val / 12.0):.1f} TeV", "in the crystals", "#ffb347")
            chrome(fig, 6, self.title, "Blue: tracker hits of the 63 collisions of our bunch crossing, each simulated alone, through the same Geant4. Dots in the calorimeters: cells, coloured by energy.",
                   clock=f"t = {val:4.1f} ns")
            return
        b = int(math.floor(val))
        for grp, mg in (("early", (self.trk_bx < 0) & (self.trk_bx <= b)), ("late", (self.trk_bx > 0) & (self.trk_bx <= b)), ("intime", (self.trk_bx == 0) & (b >= 0))):
            if mg.any():
                pt.dot(trk[mg, :3], PU_COL[grp], size=1.3, alpha=0.6, scale=False)
        self.cells(pt, self.cal_bx <= b)
        mm = self.mu_bx <= b
        pt.dot(mu[mm, :3], "#ffffff", size=22.0, marker="*", scale=False)
        pt.paint(ax)
        sl = slice(0, b + 13)
        counter(fig, 0.015, 0.86, f"{int(D['n_col'][sl].sum())}", "collisions so far")
        counter(fig, 0.015, 0.76, SHORT(D["n_trk"][sl].sum()), "hits in the silicon")
        counter(fig, 0.015, 0.66, f"{D['e_ecal'][sl].sum() / 1e3:.0f} TeV", "in the crystals", "#ffb347")
        counter(fig, 0.015, 0.56, f"{int(D['n_mu'][sl].sum())}", "hits in the muon chambers")
        chrome(fig, 6, "sixteen bunch crossings, one after the other", "Yellow: the twelve crossings before ours. Blue: our own. Red: the three after. One file per sample: a freedom only a simulation has.",
               clock=f"bunch crossing {b:+d}" if b else "bunch crossing 0")


# ---------------------------------------------------------------------------
# scene 7: the sum, on one page
# ---------------------------------------------------------------------------
class SumScene(Scene):
    number, title = 7, "the sum, in the r-z view"
    IMG = {"alone": "simHits_signal_10_SIM_ALL_rz.png", "rz": "simHits_mixed_10_SIM_ALL_rz.png"}

    def __init__(self, W):
        super().__init__(W, [("alone", 0.0, 1.0, 6.0), ("rz", 0.0, 1.0, 8.0)])
        self._img = {}

    def render(self, fig, key, u, val):
        if key not in self._img:
            self._img[key] = plt.imread(PAGES / self.IMG[key])
        fig.clf()
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.02, 0.06, 0.96, 0.875], facecolor="white")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=-5, clip_on=False))
        ax.imshow(self._img[key], interpolation="bilinear")
        ax.axis("off")
        cap = {"alone": "The hits of the family alone, r against z. Red dashed lines: where its b jets point. Only the first muon station is inside the page.",
               "rz": "The family with the 960 collisions of the crowd, all through Geant4, added by hand: every hit shifted by 25 ns times its bunch crossing."}[key]
        fig.text(0.02, 0.965, f"Scene 7: {self.title}", fontsize=15, weight="bold", color="#0b0b0b", va="center")
        fig.text(0.98, 0.965, "geant4_SIM.py + add_hits_SIM.py + plot_simhits_SIM.py", fontsize=9.5, color="#52514e", ha="right", va="center")
        fig.text(0.02, 0.025, cap, fontsize=10.5, color="#0b0b0b", va="bottom")


def build_scenes(W) -> List[Scene]:
    return [DetectorScene(W), TrackerScene(W),
            CaloScene(W, 3, "the crystals: electromagnetic showers", "ecal", (sim.EB, sim.EE), 1180.0, 1e-3, 3.0, "ECAL"),
            CaloScene(W, 4, "the HCAL: showers in brass, light in plastic", "hcal", (sim.HB, sim.HE, sim.HO), 1950.0, 1e-5, 0.03, "HCAL"),
            MuonScene(W), PileupScene(W), SumScene(W)]


# ---------------------------------------------------------------------------
# narration
# ---------------------------------------------------------------------------
def narration(W) -> Dict[str, str]:
    G, N, ex, ev = W.G, W.N, W.examples, W.ev
    row, n = G["stages"][0], G["n"]
    c, p, tr = N["collisions"], N["primaries"], N["tracker"]["total"]
    ec = N["ECAL"]
    total = sum(tr.values())
    T = ev.T
    J = sim.load_stage().genjets["ak4"]
    lines = {
        "detector_build": ("The gate has opened. ... This is the one stop of the road where we do not write the simulation ourselves. This is Geant4, the toolkit that every experiment at the collider uses, "
                           "and this is the CMS detector of Run 3 as Geant4 has it: two million volumes, each with its shape and its material. From the inside out: the silicon of the tracker, the crystals, "
                           "the brass and plastic of the hadron calorimeter, the coil, the steel of the yoke, and the muon chambers between its layers."),
        "detector_field": ("And around it all, the field. A superconducting coil, thirteen metres long, carries forty million ampere-turns. Inside it the field lines run along the beam, three point eight tesla strong. "
                           "At the ends they turn outwards, and the steel of the yoke brings them back, two tesla in the other direction. Every charged particle of our family will feel this field at every step."),
        "tracker_helix": (f"Time zero. {row['primaries']} members of our family step into the detector, and Geant4 takes them one step at a time. "
                          "The field bends every charged particle into a helix: the more momentum, the straighter. Photons and neutral hadrons fly on straight lines. In four nanoseconds, the fastest have crossed the tracker."),
        "tracker_hits": (f"Every time a charged particle crosses a silicon sensor, it knocks out electrons along its path and loses a little energy: about a hundred kilo electron volts. That is ionisation, and that is a hit. "
                         f"Geant4 writes down {SHORT(row['trk'])} of them, and {row['de']:.1f} GeV left in the silicon. In the material between the sensors the particles are also scattered, a tiny random angle at a time, "
                         f"and now and then an electron is kicked out hard enough to become a track of its own: {SHORT(n['delta_rays'])} of these delta rays."),
    }
    if ex.get("looper"):
        R = 100.0 * ex["looper"]["pt"] / (0.299792458 * 3.8)
        lines["tracker_looper"] = (f"Now look closer. This pion has a transverse momentum of only {ex['looper']['pt']:.2f} GeV. Its helix has a radius of {R:.0f} centimetres, too small to reach the calorimeter. "
                                   "So it turns, and turns again, crossing the same layers over and over and leaving a hit each time, until it has lost its energy or runs out of the tracker along the beam. This is a looper.")
    if ex.get("decay"):
        i = ex["decay"]["parent"]
        r = math.hypot(*ex["decay"]["pos"][:2])
        lines["tracker_decay"] = (f"Some members do not live long enough to cross. This is a {spoken_name(int(T[i, 1]))} of {T[i, 8]:.0f} GeV. It is neutral, so it leaves no hits. "
                                  f"{r:.0f} centimetres from the beam it decays, and two charged pions appear out of nothing. In the tracker volume, Geant4 counts {n['decays']} decays in flight.")
    if ex.get("conversion"):
        r = math.hypot(*ex["conversion"]["pos"][:2])
        lines["tracker_conversion"] = (f"A photon of {ex['conversion']['v'][6]:.0f} GeV. It flies straight and unseen, until, {r:.0f} centimetres from the beam, it passes close to a nucleus of the material "
                                       f"and turns into an electron and a positron, which the field pulls apart. {n['conversions']} photons convert before they reach the crystals.")
    if ex.get("brems"):
        i = ex["brems"]["parent"]
        lines["tracker_brems"] = (f"And electrons pay a toll of their own. This one, {T[i, 8]:.0f} GeV, is slowed by the nuclei it passes and radiates photons, here in green, along its direction of flight. "
                                  "After each one it has less energy and curls more tightly. That is bremsstrahlung.")
    if ex.get("nuclear"):
        r = math.hypot(*ex["nuclear"]["pos"][:2])
        i = ex["nuclear"]["parent"]
        lines["tracker_nuclear"] = (f"The hadrons feel the strong force as well. This {spoken_name(int(T[i, 1]))} of {ex['nuclear']['v'][6]:.0f} GeV hits a nucleus head on, {r:.0f} centimetres from the beam, "
                                    f"and {int(ex['nuclear']['v'][7])} particles come out: pions, protons, neutrons, fragments of the nucleus. {n['nuclear']} such nuclear interactions happen in the tracker. "
                                    "The tracker is thin, but it is not empty.")
    lines.update({
        "ecal_wall": (f"Those who make it out of the tracker meet a wall of crystal. Lead tungstate, as dense as steel and clear as glass. An electron or a photon does not get through: it showers. "
                      f"One becomes two, two become four, until thousands of particles share the energy and it is absorbed. The crystals receive {row['EB'] + row['EE']:.0f} GeV from our family."),
        "ecal_shower": ("From close by, one shower. Geant4 follows every electron and every photon of it, down to a thousandth of the energy it started with. "
                        "Each dot is energy left in the crystal, and the frames are the real crystals: two centimetres wide, twenty three long. The shower is over in a handful of them."),
        "hcal_wall": (f"Behind the crystals, the hadron calorimeter: a metre of brass with thin layers of plastic scintillator in between. A hadron showers in the brass, and only what happens to cross the plastic is seen. "
                      f"Of all the energy the family's hadrons lose here, the scintillator receives {row['HB'] + row['HE']:.1f} GeV."),
        "hcal_shower": ("One hadronic shower, from the side. It starts deep, it is wide and uneven, and part of its energy never becomes light: it breaks nuclei apart, or leaves with neutrons and neutrinos. "
                        "This is why a jet is harder to measure than an electron."),
        "muon_iron": (f"And behind the calorimeters and the coil? Almost nothing. Ten interaction lengths of material have absorbed the family. A few particles leak into the steel of the yoke, among them {int((np.unique(ev.seg_trk[ev.outer[ev.seg_trk] & (ev.seg_kind == 4)])).size)} muons born when pions and kaons decayed, "
                      f"and the chambers record {row['mu']} hits, nearly all of them in one chamber of the endcap, where a single slow proton stopped in the gas. For an event without an energetic muon, this near silence is the right answer."),
        "pileup_intime": (f"That was the family: one collision. But it did not cross alone. In the same bunch crossing, {c['intime']} other pairs of protons collided, and their {SHORT(p['intime'])} particles go through the same detector, "
                          f"the same field, the same Geant4. We do not repeat the steps. Each collision is simulated on its own, as if the others were not there. Together they leave {SHORT(tr['intime'])} hits in the silicon."),
        "pileup_all": (f"And the detector has a memory. {c['early'] + c['late']} more collisions, from the twelve bunch crossings before ours and the three after it, twenty five nanoseconds apart. "
                       "The signal, the pileup in time, the pileup out of time: three samples, kept in three files. If the signal changes, the crowd need not be simulated again. "
                       "Only a simulation has this freedom. In data, everything happens at once."),
        "sum_alone": ("Now everything on one page: the radius against the distance along the beam. Only the first muon station fits on this page. This is the family alone. The red lines are the directions of its b jets. "
                      f"One points to the corner where the barrel meets the endcap, on the left: that is our jet, {J[0]['pt']:.0f} GeV. "
                      "The other b jets went forward: they end in the forward calorimeter, eleven metres down the beam pipe, at the far left."),
        "sum_rz": (f"And now with the crowd: {SHORT(1 + c['intime'] + c['early'] + c['late'])} collisions, added by hand, every hit shifted by twenty five nanoseconds times the number of its crossing. "
                   f"The sum: {SHORT(total)} hits in the tracker, and our family owns {100 * tr['signal'] / total:.1f} percent of them. In the crystals, {SHORT(ec['hit_all'])} cells hold energy, "
                   f"and in only {SHORT(ec['dom']['signal'])} of them is the family the largest contributor. Our jet is still there, along the same red line. It is no longer alone."),
        "outro": ("Nothing has been measured yet. A hit is deposited energy, not a signal. ... Turning sixteen crossings of hits into the pulses of the electronics is the next film."),
    })
    return lines


def spoken_name(pid: int) -> str:
    return {310: "K short", 130: "K long", 211: "pion", -211: "pion", 321: "kaon", -321: "kaon", 2212: "proton", -2212: "antiproton", 2112: "neutron", -2112: "antineutron",
            3122: "lambda", -3122: "antilambda", 22: "photon", 11: "electron", -11: "positron"}.get(pid, "hadron")


_LAST: Dict[int, Tuple[int, float]] = {}


def draw(fig, sc, k, t):
    """One frame; a frame that repeats the one before (a hold under the narration) is not drawn again."""
    key = (sc.number, round(float(k), 3))
    if _LAST.get(id(fig)) != key:
        sc.draw(fig, k)
        _LAST[id(fig)] = key


STILLS = [(1, "field", 1.0), (2, "hits", 0.35), (2, "conversion", 0.9), (3, "shower", 0.9), (4, "wall", 1.0), (5, "iron", 1.0), (6, "all", 1.0), (7, "rz", 1.0)]


def write_stills(scenes, out: Path, dpi: int = 85, quality: int = 88):
    """Eight key frames of the film on one page. The page is a photograph, not a drawing, so it is stored as a
    JPEG inside the PDF: 0.7 MB instead of 6."""
    from PIL import Image
    by = {sc.number: sc for sc in scenes}
    imgs = []
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi)
    for number, key, u in STILLS:
        sc = by[number]
        seg = [s for s in sc.seg if s[0] == key]
        if not seg:
            continue
        _, k0, k1, _, _ = seg[0]
        sc.draw(fig, k0 + (k1 - 1 - k0) * u)
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    h, w = imgs[0].shape[:2]
    gap = 8
    rows = (len(imgs) + 1) // 2
    page = np.full((rows * h + (rows + 1) * gap, 2 * w + 3 * gap, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        y, x = gap + (i // 2) * (h + gap), gap + (i % 2) * (w + gap)
        page[y:y + h, x:x + w] = im
    Image.fromarray(page).save(out, "PDF", resolution=170.0, quality=quality)
    print(f"{out} written ({len(imgs)} key frames, {out.stat().st_size / 1e6:.2f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_SIM.mp4")
    p.add_argument("--scene", default="all", help="all, or numbers like 1,4")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--fast", action="store_true", help="a preview: every fourth frame at 60 dpi")
    p.add_argument("--stills", type=Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, default=REPO / "build" / "narration" / "SIM")
    snd.add_narration_args(p)
    args = p.parse_args()
    global BG_DPI
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
    BG_DPI = args.dpi
    snd.narration_defaults(args)
    W = World()
    lines = narration(W)
    if args.print_script:
        snd.print_script(lines)
        return
    scenes = build_scenes(W)
    if args.stills is not None:
        write_stills(scenes, args.stills)
    if args.no_video:
        return
    if args.scene != "all":
        want = {int(s) for s in args.scene.split(",")}
        scenes = [sc for sc in scenes if sc.number in want]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and not args.no_narration:
        sys.exit("ffmpeg not found on PATH (needed for the narration clips)")
    clips = snd.load_narration(lines, args, args.cache_dir, ffmpeg) if ffmpeg else {}
    film = snd.Film(scenes, snd.clip_lengths(clips), SCENE_NAMES, slow_max=4.0)
    print(f"film: {film.describe()}")
    audio = None
    tmp = tempfile.TemporaryDirectory(prefix="jitjet_sim_")
    if ffmpeg and clips:
        audio = Path(tmp.name) / "soundtrack.wav"
        snd.mix_audio(film.duration, film.voice_at, clips, None, args.music_db, audio)       # the voice alone: this film has no music
        snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
    snd.render_video(film, args.output, args.fps / args.step, args.dpi, audio, draw, FIGSIZE, "b bbar event inside Geant4: the simulated hits", jobs=args.jobs)
    print(f"wrote {args.output} ({film.duration:.1f} s)")


if __name__ == "__main__":
    main()
