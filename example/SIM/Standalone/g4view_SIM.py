#!/usr/bin/env python3
"""A small 3D viewer for the Geant4 run of chapter 10: the CMS detector as Geant4 has it, the trajectories, the
deposits and the field lines, seen by a perspective camera.

Used by display_geant4_SIM.py (the pages of the chapter) and animate_standalone_SIM.py (the film). It reads

  ../cms_shapes_SIM.npz     the corners of every sensor, crystal, tile, chamber and yoke plate (dump_shapes_SIM.py)
  ../g4tracks_signal.npz    the trajectories, vertices and shower deposits of the signal (geant4_SIM.py)
  field_map_SIM.npz         the field (field_map_SIM.py)

and draws with matplotlib only: every point is projected by the camera, surfaces are painted from the farthest to
the nearest (the painter's algorithm), lines and dots go on top. CMS coordinates: x to the centre of the LHC ring,
y up, z along the beam; lengths in cm.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import standalone_SIM as sim

HERE = Path(__file__).resolve().parent
BG = "#05070d"
# colours on the dark background: the categories of the book, with white for the black of the paper pages
KIND_COL = ["#f2f2f2", "#cc66ff", "#22cc55", "#ff7a1a", "#33a0ff"]          # charged hadron, neutral hadron, photon, e, mu
KIND_NAME = ["charged hadron", "neutral hadron", "photon", r"e$^\pm$", r"$\mu^\pm$"]
NU_COL = "#5a6270"
DET_COL = {"trk": "#4f8fd9", "pix": "#7fd0ff", "eb": "#39c5a0", "ee": "#39c5a0", "es": "#8fd98a", "hb": "#d9a13f", "he": "#d9a13f", "ho": "#b98a4a",
           "hf": "#8a7f70", "yoke": "#b3423a", "coil": "#c9ced6", "dt": "#d9d06a", "csc": "#d9d06a", "rpc": "#a0c060", "gem": "#60c0a0", "pipe": "#9aa3b0"}
PROC_COL = {"decay": "#ffd23f", "conversion": "#22cc55", "bremsstrahlung": "#ff7a1a", "nuclear interaction": "#ff3b6b",
            "delta ray": "#8fb8ff", "Compton scattering": "#7de0c0", "photoelectric effect": "#b0b0b0", "annihilation": "#ff9ad5",
            "elastic scattering": "#c0a0ff", "other": "#888888"}


def kind_of_pid(pid):
    """0..4 as in the book; 5 for neutrinos, 6 for nuclei and fragments."""
    pid = np.asarray(pid).astype(np.int64)
    a = np.abs(pid)
    k = np.zeros(len(pid), np.int8)
    k[np.isin(a, (2112, 130, 310, 3122, 3322, 111, 311))] = 1
    k[a == 22] = 2
    k[a == 11] = 3
    k[a == 13] = 4
    k[np.isin(a, (12, 14, 16))] = 5
    k[a > 1000000000] = 6
    return k


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
class Camera:
    def __init__(self, eye, target=(0, 0, 0), up=(0, 1, 0), fov=38.0, aspect=16 / 9, near=5.0):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        f = self.target - self.eye
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, np.asarray(up, float))
        if np.linalg.norm(r) < 1e-6:
            r = np.cross(self.f, np.array([1.0, 0, 0]))
        self.r = r / np.linalg.norm(r)
        self.u = np.cross(self.r, self.f)
        self.k = 1.0 / math.tan(math.radians(fov) / 2)
        self.aspect, self.near = aspect, near

    def project(self, P):
        """(X, Y, depth): screen coordinates (Y in [-1, 1]) and the distance along the line of sight."""
        d = np.asarray(P, float) - self.eye
        z = d @ self.f
        zz = np.where(z > self.near, z, np.nan)
        return self.k * (d @ self.r) / zz, self.k * (d @ self.u) / zz, z

    @staticmethod
    def orbit(target, dist, az, el, **kw):
        """A camera at distance dist from target: az is the angle around the vertical (0 = looking along -z from +z ...),
        el the elevation above the horizontal plane, both in degrees."""
        a, e = math.radians(az), math.radians(el)
        eye = np.asarray(target, float) + dist * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        return Camera(eye, target, **kw)


def lerp_camera(c0, c1, s, **kw):
    s = 0.5 - 0.5 * math.cos(math.pi * min(max(s, 0.0), 1.0))
    return Camera(c0.eye + s * (c1.eye - c0.eye), c0.target + s * (c1.target - c0.target), **kw)


# ---------------------------------------------------------------------------
# the detector as polygons
# ---------------------------------------------------------------------------
def midplane(c):
    """The quad through the middle of the thinnest direction of boxes c (n, 8, 3) in the corner order of dump_shapes."""
    ex = np.linalg.norm(c[:, 1] - c[:, 0], axis=1)
    ey = np.linalg.norm(c[:, 2] - c[:, 0], axis=1)
    ez = np.linalg.norm(c[:, 4] - c[:, 0], axis=1)
    thin = np.argmin(np.stack([ex, ey, ez]), axis=0)
    q = np.zeros((len(c), 4, 3), np.float32)
    for ax, pairs in ((0, ((0, 1), (2, 3), (6, 7), (4, 5))), (1, ((0, 2), (1, 3), (5, 7), (4, 6))), (2, ((0, 4), (1, 5), (3, 7), (2, 6)))):
        m = thin == ax
        for j, (a, b) in enumerate(pairs):
            q[m, j] = 0.5 * (c[m, a] + c[m, b])
    return q


def near_face(c):
    """The face of a crystal that looks at the collision point."""
    lo, hi = c[:, [0, 1, 3, 2]], c[:, [4, 5, 7, 6]]
    m = np.linalg.norm(lo.mean(axis=1), axis=1) < np.linalg.norm(hi.mean(axis=1), axis=1)
    return np.where(m[:, None, None], lo, hi)


def box_faces(c):
    """The six faces of boxes c (n, 8, 3) -> (6 n, 4, 3)."""
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return np.concatenate([c[:, list(i)] for i in idx])


def cylinder(r, z0, z1, phi0=0.0, phi1=2 * math.pi, n=72):
    ph = np.linspace(phi0, phi1, n + 1)
    x, y = r * np.cos(ph), r * np.sin(ph)
    q = np.zeros((n, 4, 3), np.float32)
    q[:, 0] = np.column_stack([x[:-1], y[:-1], np.full(n, z0)]); q[:, 1] = np.column_stack([x[1:], y[1:], np.full(n, z0)])
    q[:, 2] = np.column_stack([x[1:], y[1:], np.full(n, z1)]); q[:, 3] = np.column_stack([x[:-1], y[:-1], np.full(n, z1)])
    return q


def annulus(r0, r1, z, phi0=0.0, phi1=2 * math.pi, n=72):
    ph = np.linspace(phi0, phi1, n + 1)
    c, s = np.cos(ph), np.sin(ph)
    q = np.zeros((n, 4, 3), np.float32)
    q[:, 0] = np.column_stack([r0 * c[:-1], r0 * s[:-1], np.full(n, z)]); q[:, 1] = np.column_stack([r1 * c[:-1], r1 * s[:-1], np.full(n, z)])
    q[:, 2] = np.column_stack([r1 * c[1:], r1 * s[1:], np.full(n, z)]); q[:, 3] = np.column_stack([r0 * c[1:], r0 * s[1:], np.full(n, z)])
    return q


class Detector:
    """The polygons of CMS by layer. Each layer: quads (n, 4, 3), face colour, alphas."""

    def __init__(self, path: Path = HERE.parent / "cms_shapes_SIM.npz"):
        s = np.load(path)
        self.raw = {k: s[k] for k in s.files}
        L = {}
        t, sub = s["trk"], s["trk_sub"]
        pix = np.isin(sub, (sim.PXB, sim.PXF))
        L["pix"] = midplane(t[pix])
        L["trk"] = midplane(t[~pix])
        self.trk_quads, self.trk_sub = midplane(t), sub
        L["eb"] = near_face(s["eb"]); L["ee"] = near_face(s["ee"])
        L["hb"] = midplane(s["hb"]); L["he"] = midplane(s["he"]); L["ho"] = midplane(s["ho"])
        L["hf"] = box_faces(s["hf"])
        L["yoke"] = box_faces(s["yoke"])
        mu, ms = s["mu"], s["mu_sub"]
        L["csc"] = midplane(mu[ms == sim.CSC]); L["rpc"] = midplane(mu[ms == sim.RPC]); L["gem"] = midplane(mu[ms == sim.GEM])
        L["dt"] = self._dt_layers(mu[ms == sim.DT])
        L["coil"] = np.concatenate([cylinder(320.0, -615.0, 615.0), cylinder(340.0, -615.0, 615.0)])
        L["pipe"] = cylinder(2.25, -300.0, 300.0, n=24)
        self.L = L
        self.centre = {k: v.mean(axis=1) for k, v in L.items()}

    @staticmethod
    def _dt_layers(c):
        """The 172 000 drift cells of the barrel, merged into their layers (one quad per layer, sector and wheel)."""
        ctr = c.mean(axis=1)
        phi = np.arctan2(ctr[:, 1], ctr[:, 0])
        sec = np.round(phi / (math.pi / 6)).astype(int) % 12
        ps = sec * math.pi / 6
        u = ctr[:, 0] * np.cos(ps) + ctr[:, 1] * np.sin(ps)             # along the normal of the sector
        v = -ctr[:, 0] * np.sin(ps) + ctr[:, 1] * np.cos(ps)
        wheel = np.clip(np.round(ctr[:, 2] / 268.0), -2, 2).astype(int)
        key = (sec * 5 + wheel + 2) * 1000 + np.round(u / 2.0).astype(int)
        order = np.argsort(key, kind="stable")
        key, u, v, z, ps = key[order], u[order], v[order], ctr[order, 2], ps[order]
        st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        en = np.r_[st[1:], len(key)]
        q = np.zeros((len(st), 4, 3), np.float32)
        for i, (a, b) in enumerate(zip(st, en)):
            uu, p = u[a:b].mean(), ps[a]
            v0, v1 = v[a:b].min() - 2.1, v[a:b].max() + 2.1
            z0, z1 = z[a:b].min(), z[a:b].max()
            if z1 - z0 < 50.0:                                          # a z superlayer: the wires run along phi
                z0, z1 = z0 - 2.1, z1 + 2.1
            else:
                z0, z1 = z0 - 117.0, z1 + 117.0
            cp, sp = math.cos(p), math.sin(p)
            for j, (vv, zz) in enumerate(((v0, z0), (v1, z0), (v1, z1), (v0, z1))):
                q[i, j] = (uu * cp - vv * sp, uu * sp + vv * cp, zz)
        return q

    def quads(self, name, cut=None):
        """The quads of a layer, without those whose centre the function cut(x, y, z) -> bool removes."""
        q = self.L[name]
        if cut is None:
            return q
        c = self.centre[name]
        return q[~cut(c[:, 0], c[:, 1], c[:, 2])]


def quadrant_cut(phi0=-20.0, phi1=100.0):
    """Remove everything between two azimuths (degrees): the classic cut-away."""
    def cut(x, y, z):
        ph = np.degrees(np.arctan2(y, x))
        return (ph > phi0) & (ph < phi1)
    return cut


# ---------------------------------------------------------------------------
# the signal: trajectories, vertices, deposits
# ---------------------------------------------------------------------------
class Event:
    def __init__(self, path: Path = HERE.parent / "g4tracks_signal.npz", hits: Path = HERE.parent / "simhits_signal.npz"):
        g = np.load(path)
        self.T, self.P, self.Ptrk, self.V, self.vox = g["T"], g["P"], g["P_trk"], g["V"], g["vox"]
        self.proc_names = [str(x) for x in g["proc_names"]]
        self.kind = kind_of_pid(self.T[:, 1])
        self.outer = self.T[:, 9] == 1
        same = self.Ptrk[1:] == self.Ptrk[:-1]
        self.seg_a = np.flatnonzero(same)                 # segment i: point seg_a[i] -> seg_a[i] + 1
        self.seg_trk = self.Ptrk[self.seg_a]
        self.seg_t = self.P[self.seg_a + 1, 3]
        self.seg_kind = self.kind[self.seg_trk]
        self.seg_ek = self.T[self.seg_trk, 8]
        self.seg_prim = self.T[self.seg_trk, 0].astype(int)
        self.seg_is_prim = self.T[self.seg_trk, 2] < 0
        pa, pb = self.P[self.seg_a, :3], self.P[self.seg_a + 1, :3]
        self.seg_in_trk = (np.abs(pa[:, 2]) < 292.0) & (np.abs(pb[:, 2]) < 292.0) & (np.hypot(pa[:, 0], pa[:, 1]) < 124.0) & (np.hypot(pb[:, 0], pb[:, 1]) < 124.0)
        h = np.load(hits)
        self.hits = {k: h[k] for k in h.files}
        self.t0 = float(self.P[:, 3].min())

    def segments(self, mask=None):
        a = self.seg_a if mask is None else self.seg_a[mask]
        return self.P[a, :3], self.P[a + 1, :3]


# ---------------------------------------------------------------------------
# field lines
# ---------------------------------------------------------------------------
def field_lines(fmap, levels=14, rmax=1250.0, zmax=1900.0):
    """Field lines in the (z, r) half plane: lines of constant flux, so that their density is the field strength.
    Returns a list of (z, r, |B|) arrays. The flux values are chosen equally spaced in the flux through the bore."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    psi = np.concatenate([fmap.psi[:, :0:-1], fmap.psi], axis=1)
    z = np.concatenate([-fmap.z[:0:-1], fmap.z])
    r = fmap.r
    br, bz = fmap.grid(average=True)
    bmag = np.hypot(br, bz)
    bmag = np.concatenate([bmag[:, :0:-1], bmag], axis=1)
    top = float(fmap.psi[np.searchsorted(r, 300.0), 0])          # the flux through the bore at z = 0
    lv = top * (np.arange(levels) + 0.5) / levels
    fig = plt.figure()
    cs = plt.contour(z, r, psi, levels=np.sort(lv))
    out = []
    for path in cs.get_paths():
        for seg in path.to_polygons(closed_only=False):
            seg = np.asarray(seg)
            if len(seg) < 8:
                continue
            m = (np.abs(seg[:, 0]) <= zmax) & (seg[:, 1] <= rmax)
            if m.sum() < 8:
                continue
            ii = np.clip(np.searchsorted(r, seg[:, 1]), 0, len(r) - 1)
            jj = np.clip(np.searchsorted(z, seg[:, 0]), 0, len(z) - 1)
            out.append(np.column_stack([seg[:, 0], seg[:, 1], bmag[ii, jj]])[m])
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
class Painter:
    """Collects surfaces, lines and dots in 3D and paints them on a matplotlib axes through a camera."""

    def __init__(self, cam: Camera, light=(0.35, 0.8, 0.5)):
        self.cam = cam
        self.polys, self.lines, self.dots, self.texts = [], [], [], []
        l = np.asarray(light, float)
        self.light = l / np.linalg.norm(l)

    def surface(self, quads, color, alpha=0.25, edge=None, lw=0.2, shade=True):
        if len(quads):
            self.polys.append((np.asarray(quads, float), color, alpha, edge, lw, shade))

    def line(self, a, b, color, lw=0.8, alpha=1.0, front=True):
        """Segments a[i] -> b[i]. color: one colour or an (n, 4) array."""
        if len(a):
            self.lines.append((np.asarray(a, float), np.asarray(b, float), color, lw, alpha, front))

    def polyline(self, p, color, lw=0.8, alpha=1.0):
        p = np.asarray(p, float)
        self.line(p[:-1], p[1:], color, lw, alpha)

    def dot(self, p, color, size=4.0, alpha=1.0, marker="o", scale=True):
        if len(p):
            self.dots.append((np.asarray(p, float), color, size, alpha, marker, scale))

    def text(self, p, s, **kw):
        self.texts.append((np.asarray(p, float), s, kw))

    def paint(self, ax, ref_depth=None):
        import matplotlib.colors as mc
        from matplotlib.collections import LineCollection, PolyCollection
        cam = self.cam
        ax.set_xlim(-cam.aspect, cam.aspect); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.axis("off")
        ref = ref_depth or float(np.linalg.norm(cam.target - cam.eye))
        # surfaces, far to near
        V, D, FC, EC, LW = [], [], [], [], []
        for q, color, alpha, edge, lw, shade in self.polys:
            n = len(q)
            X, Y, Z = cam.project(q.reshape(-1, 3))
            X, Y, Z = X.reshape(n, 4), Y.reshape(n, 4), Z.reshape(n, 4)
            ok = np.isfinite(X).all(axis=1) & (np.abs(X).min(axis=1) < cam.aspect * 1.2) & (np.abs(Y).min(axis=1) < 1.2)
            if not ok.any():
                continue
            rgb = np.array(mc.to_rgb(color))
            if shade:
                nrm = np.cross(q[ok, 1] - q[ok, 0], q[ok, 3] - q[ok, 0])
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
                lam = 0.55 + 0.45 * np.abs(nrm @ self.light)
            else:
                lam = np.ones(int(ok.sum()))
            fc = np.column_stack([np.outer(lam, rgb), np.full(int(ok.sum()), alpha)])
            V.append(np.stack([X[ok], Y[ok]], axis=2)); D.append(Z[ok].mean(axis=1)); FC.append(fc)
            ec = np.array(mc.to_rgba(edge if edge else color, min(1.0, alpha * (2.5 if edge else 1.2))))
            EC.append(np.tile(ec, (int(ok.sum()), 1))); LW.append(np.full(int(ok.sum()), lw))
        if V:
            V, D, FC, EC, LW = np.concatenate(V), np.concatenate(D), np.concatenate(FC), np.concatenate(EC), np.concatenate(LW)
            o = np.argsort(-D)
            ax.add_collection(PolyCollection(V[o], facecolors=FC[o], edgecolors=EC[o], linewidths=LW[o], zorder=1, rasterized=True))
        for a, b, color, lw, alpha, front in self.lines:
            X0, Y0, Z0 = cam.project(a)
            X1, Y1, Z1 = cam.project(b)
            ok = np.isfinite(X0) & np.isfinite(X1) & ((np.abs(X0) < cam.aspect * 1.5) | (np.abs(X1) < cam.aspect * 1.5))
            if not ok.any():
                continue
            segs = np.stack([np.column_stack([X0[ok], Y0[ok]]), np.column_stack([X1[ok], Y1[ok]])], axis=1)
            col = color[ok] if isinstance(color, np.ndarray) and color.ndim == 2 else mc.to_rgba(color, alpha)
            w = lw[ok] if isinstance(lw, np.ndarray) else lw
            ax.add_collection(LineCollection(segs, colors=col, linewidths=w, zorder=5 if front else 2, capstyle="round", rasterized=True))
        for p, color, size, alpha, marker, scale in self.dots:
            X, Y, Z = cam.project(p)
            ok = np.isfinite(X) & (np.abs(X) < cam.aspect) & (np.abs(Y) < 1)
            if not ok.any():
                continue
            s = (size[ok] if isinstance(size, np.ndarray) else size) * ((ref / Z[ok]) ** 2 if scale else 1.0)
            c = color[ok] if isinstance(color, np.ndarray) and color.ndim == 2 else color
            ax.scatter(X[ok], Y[ok], s=np.clip(s, 0.05, 400.0), c=c, alpha=None if isinstance(c, np.ndarray) else alpha, marker=marker, linewidths=0, zorder=6, rasterized=True)
        for p, s, kw in self.texts:
            X, Y, Z = cam.project(p[None, :])
            if np.isfinite(X[0]) and abs(X[0]) < cam.aspect and abs(Y[0]) < 1:
                ax.text(X[0], Y[0], s, zorder=9, **kw)


def new_figure(width=12.8, height=7.2, dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    return fig, ax


def track_style(ev: Event, mask, t_now=None, fade=None, lw_scale=1.0):
    """Colours and widths of the segments in mask: by the kind of the particle, thicker for the energetic ones, and,
    in a film, only up to the time t_now with the head brighter than the tail."""
    import matplotlib.colors as mc
    idx = np.flatnonzero(mask)
    kind = ev.seg_kind[idx]
    pal = np.array([mc.to_rgba(c) for c in KIND_COL] + [mc.to_rgba(NU_COL), mc.to_rgba("#a08060")])
    col = pal[kind].copy()
    ek = ev.seg_ek[idx]
    lw = lw_scale * np.clip(0.35 + 0.28 * np.log10(np.maximum(ek, 1e-4) / 1e-3), 0.3, 2.0)
    alpha = np.clip(0.35 + 0.2 * np.log10(np.maximum(ek, 1e-4) / 1e-3), 0.3, 1.0)
    alpha[kind == 2] *= 0.55                       # photons: fainter
    alpha[kind == 1] *= 0.7
    alpha[kind == 5] = 0.35
    if t_now is not None:
        t = ev.seg_t[idx]
        vis = t <= t_now
        if fade:
            alpha = alpha * np.clip(1.0 - (t_now - t) / fade, 0.25, 1.0)
        idx, col, lw, alpha = idx[vis], col[vis], lw[vis], alpha[vis]
    col[:, 3] = alpha
    return idx, col, lw
