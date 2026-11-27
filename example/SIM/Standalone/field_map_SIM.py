#!/usr/bin/env python3
"""The magnetic field of CMS, computed from the coil and the iron of the Geant4 geometry.

CMSSW gives Geant4 a field map measured and computed by the magnet group. That map is not public as a
file one can read on a laptop, so this script computes one: the coil and the iron are read out of the same
geometry Geant4 tracks through (../../G4/geometry/cmsSimGeo2021.gdml), and Maxwell's equations are solved on
a grid.

The problem is taken as symmetric around the beam (the twelve-sided yoke is averaged over azimuth) and
mirror-symmetric in z. With the flux function psi(r, z) = r A_phi the field is

    B_r = -(1/r) d psi/dz,     B_z = (1/r) d psi/dr,

a field line is a line of constant psi, and Ampere's law curl(B / mu) = mu0 J becomes

    d/dr( (1/(mu_r r)) d psi/dr ) + d/dz( (1/(mu_r r)) d psi/dz ) = -mu0 J_phi.

* J_phi is uniform in the volumes made of the material "Coil_average" (the superconducting winding) and is
  scaled until B_z = 3.8 T at the centre.
* mu_r = 1 outside the iron. In the steel of the return yoke mu_r follows a saturating B(H) curve,
  B = mu0 H + (2 Bs / pi) atan(pi (mu_i - 1) mu0 H / (2 Bs)) with Bs = 2.05 T and mu_i = 2000, and the
  equation is solved again until mu_r stops changing (Picard iteration). Where only a fraction f of the
  azimuth is iron (the gaps between the sectors, the chimneys) the cell gets mu_r = 1 + f (mu_iron - 1): the
  flux runs along the plates, so H is the same in the iron and in the gap, and the field inside the iron
  is mu_iron mu0 H while the gap next to it has mu0 H.

Geant4's navigator answers "which material is at this point" for the 1.4 million points of the scan; the
linear system (90 000 unknowns) is solved by scipy. About a minute. Run with the Geant4 interpreter:

  .venv/bin/python field_map_SIM.py            # writes field_map_SIM.npz and prints the field at a few places

The map is used by geant4_SIM.py (the field Geant4 bends the particles in) and by display_geant4_SIM.py (the
field lines). Lengths in cm, fields in tesla.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GDML = Path(__import__("os").environ.get("JITJET_SIM_GDML", HERE.parent.parent / "G4" / "geometry" / "cmsSimGeo2021.gdml"))
MAP = HERE / "field_map_SIM.npz"
B_CENTRE = 3.8                    # tesla
H_GRID = 5.0                      # cm
R_MAX, Z_MAX = 1300.0, 2000.0     # cm: the box of the solution (psi = 0 on its walls)
N_PHI = 16
BS, MU_I = 2.05, 2000.0           # the steel: saturation magnetisation [T], initial permeability
FERRO = ("materialsM_Steel-008", "materialsSteel-008", "materialsIron", "materialsYokeSteel", "materialsAISI-1018-Steel",
         "hcalforwardmaterialHFAbsorber")
COIL = ("materialsCoil_average",)
MU0 = 4e-7 * math.pi


def mu_iron(b0):
    """Relative permeability of the steel as a function of mu0 H [T]."""
    b0 = np.maximum(b0, 1e-9)
    return 1.0 + (2.0 * BS / math.pi) * np.arctan(math.pi * (MU_I - 1.0) * b0 / (2.0 * BS)) / b0


def scan_geometry(say=print):
    """Fraction of the azimuth that is iron, and that is coil, at every node of the (r, z) grid."""
    from geant4_pybind import G4GDMLParser, G4GeometryManager, G4Navigator, G4ThreeVector, cm
    parser = G4GDMLParser()
    parser.Read(str(GDML), False)
    world = parser.GetWorldVolume()
    G4GeometryManager.GetInstance().CloseGeometry(True, False)
    nav = G4Navigator()
    nav.SetWorldVolume(world)
    r = np.arange(0.0, R_MAX + 1e-6, H_GRID)
    z = np.arange(0.0, Z_MAX + 1e-6, H_GRID)
    fe = np.zeros((len(r), len(z)))
    co = np.zeros((len(r), len(z)))
    kind = {}
    cs = [(math.cos((k + 0.37) * 2 * math.pi / N_PHI), math.sin((k + 0.37) * 2 * math.pi / N_PHI)) for k in range(N_PHI)]
    t0 = time.time()
    for i, ri in enumerate(r):
        for j, zj in enumerate(z):
            nf = nc = 0
            for c, s in cs:
                pv = nav.LocateGlobalPointAndSetup(G4ThreeVector(ri * c * cm, ri * s * cm, zj * cm))
                if pv is None:
                    continue
                lv = pv.GetLogicalVolume()
                nm = lv.GetName()
                k = kind.get(nm)
                if k is None:
                    m = lv.GetMaterial().GetName()
                    k = kind[nm] = 1 if m in FERRO else (2 if m in COIL else 0)
                nf += k == 1
                nc += k == 2
            fe[i, j], co[i, j] = nf / N_PHI, nc / N_PHI
        if i % 40 == 0:
            say(f"   scan r = {ri:6.0f} cm, {time.time() - t0:.0f} s")
    return r, z, fe, co


def solve(r, z, fe, co, say=print, iterations=150):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    nr, nz = len(r), len(z)
    h = H_GRID * 1e-2                                    # metres
    rm = r * 1e-2
    idx = -np.ones((nr, nz), dtype=np.int64)
    inner = np.zeros((nr, nz), bool)
    inner[1:-1, :-1] = True                              # psi = 0 at r = 0, r = R_MAX, z = Z_MAX; mirror at z = 0
    idx[inner] = np.arange(inner.sum())
    n = int(inner.sum())
    ii, jj = np.nonzero(inner)
    mu = 1.0 + fe * (MU_I - 1.0)
    mufe = np.full((nr, nz), MU_I)
    scale = 1.0
    for it in range(iterations):
        nu = 1.0 / mu
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        P = idx[ii, jj]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            i2, j2 = ii + di, jj + dj
            j2m = np.where(j2 < 0, 1, j2)                # the mirror at z = 0
            nuf = 0.5 * (nu[ii, jj] + nu[i2, j2m])
            rf = rm[ii] + 0.5 * di * h
            a = nuf / rf / (h * h)
            diag -= a
            q = idx[i2, j2m]
            ok = q >= 0
            rows.append(P[ok]); cols.append(q[ok]); vals.append(a[ok])
        rows.append(P); cols.append(P); vals.append(diag)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        rhs = -MU0 * co[ii, jj]                          # unit current density in the coil
        sol = spl.spsolve(A.tocsc(), rhs)
        psi = np.zeros((nr, nz))
        psi[ii, jj] = sol
        scale = B_CENTRE / (2.0 * psi[1, 0] / (h * h))   # B_z on the axis = 2 psi(h) / h^2
        psi *= scale
        br, bz = field_of_psi(psi, rm, h)
        b0 = np.hypot(br, bz) / mu                       # mu0 H
        new = mu_iron(b0)
        change = float(np.max(np.abs(np.log(new / mufe))[fe > 0.5])) if (fe > 0.5).any() else 0.0
        mufe = np.exp(0.5 * np.log(mufe) + 0.5 * np.log(new))
        mu = 1.0 + fe * (mufe - 1.0)
        if it % 10 == 0 or change < 0.02:
            say(f"   iteration {it:2d}: largest change of ln(mu) in the iron {change:.3f}")
        if change < 0.02:
            break
    current = scale * co.sum() * h * h * 2.0             # ampere-turns, both halves
    return psi, br, bz, mu, mufe, current


def field_of_psi(psi, rm, h):
    nr, nz = psi.shape
    dz = np.zeros_like(psi)
    dz[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * h)
    dz[:, -1] = (psi[:, -1] - psi[:, -2]) / h
    dr = np.zeros_like(psi)
    dr[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * h)
    dr[-1, :] = (psi[-1, :] - psi[-2, :]) / h
    rr = np.maximum(rm, 1e-9)[:, None]
    br = -dz / rr
    bz = dr / rr
    br[0, :] = 0.0
    bz[0, :] = 2.0 * psi[1, :] / (h * h)
    return br, bz


class FieldMap:
    """The solved map. b0 = mu0 H is the field outside the iron; inside the iron it is mufe times that."""

    def __init__(self, path: Path = MAP):
        d = np.load(path)
        self.r, self.z = d["r"], d["z"]
        self.psi, self.fe, self.co, self.mufe = d["psi"], d["fe"], d["co"], d["mufe"]
        self.b0r, self.b0z = d["b0r"], d["b0z"]
        self.h = float(self.r[1] - self.r[0])
        self.nr, self.nz = len(self.r), len(self.z)
        self.current = float(d["current"])
        # plain lists: a lookup in a Python list is several times faster than in a numpy array, and Geant4
        # asks for the field ten million times
        self._br, self._bz, self._mu = self.b0r.tolist(), self.b0z.tolist(), self.mufe.tolist()

    def b(self, x, y, z, iron=False):
        """(Bx, By, Bz) in tesla at one point [cm]."""
        r = math.hypot(x, y)
        az = abs(z)
        fi, fj = r / self.h, az / self.h
        i, j = int(fi), int(fj)
        if i >= self.nr - 1 or j >= self.nz - 1:
            return 0.0, 0.0, 0.0
        u, v = fi - i, fj - j
        w00, w10, w01, w11 = (1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v
        a, b_ = self._br, self._bz
        br = w00 * a[i][j] + w10 * a[i + 1][j] + w01 * a[i][j + 1] + w11 * a[i + 1][j + 1]
        bz = w00 * b_[i][j] + w10 * b_[i + 1][j] + w01 * b_[i][j + 1] + w11 * b_[i + 1][j + 1]
        if iron:
            m = self._mu
            f = w00 * m[i][j] + w10 * m[i + 1][j] + w01 * m[i][j + 1] + w11 * m[i + 1][j + 1]
            br *= f
            bz *= f
        if z < 0:
            br = -br
        if r > 1e-9:
            return br * x / r, br * y / r, bz
        return 0.0, 0.0, bz

    def grid(self, average=True):
        """(B_r, B_z) on the grid [T]: the azimuthal average (what the field lines show) or the field outside the iron."""
        if average:
            mu = 1.0 + self.fe * (self.mufe - 1.0)
            return self.b0r * mu, self.b0z * mu
        return self.b0r, self.b0z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=MAP)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    say = (lambda *_: None) if args.quiet else print
    t0 = time.time()
    say(f"1. the coil and the iron, from {GDML.name} ({N_PHI} azimuths per grid point, {H_GRID:.0f} cm grid)")
    r, z, fe, co = scan_geometry(say)
    say(f"   iron in {int((fe > 0).sum())} cells, coil in {int((co > 0).sum())} cells; {time.time() - t0:.0f} s")
    say("2. Ampere's law on the grid, with the saturation of the steel")
    psi, br, bz, mu, mufe, current = solve(r, z, fe, co, say)
    np.savez_compressed(args.output, r=r, z=z, psi=psi.astype(np.float32), fe=fe.astype(np.float32), co=co.astype(np.float32),
                        mufe=mufe.astype(np.float32), b0r=(br / mu).astype(np.float32), b0z=(bz / mu).astype(np.float32), current=current)
    m = FieldMap(args.output)
    say(f"3. the map: {current / 1e6:.1f} million ampere-turns give {B_CENTRE} T at the centre (CMS: 41.7 million for 4 T, 39.6 for 3.8 T)")
    for name, (x, zz, iron) in {"centre": (0, 0, False), "tracker edge (r 110, z 280)": (110, 280, False), "ECAL barrel (r 140)": (140, 0, False),
                                "HCAL barrel (r 230)": (230, 0, False), "HCAL endcap (r 150, z 480)": (150, 480, False),
                                "inside the coil cryostat (r 290)": (290, 0, False), "first muon station (r 420)": (420, 0, False),
                                "barrel yoke, layer 1 (r 475)": (475, 0, True), "barrel yoke, layer 2 (r 585)": (585, 0, True),
                                "barrel yoke, layer 3 (r 680)": (680, 0, True), "endcap disk 1 (r 400, z 760)": (400, 760, True),
                                "endcap disk 2 (r 400, z 860)": (400, 860, True), "HF (r 60, z 1200)": (60, 1200, False)}.items():
        b = m.b(x, 0.0, zz, iron)
        say(f"   {name:34s} B_r {b[0]:+6.2f}  B_z {b[2]:+6.2f}  |B| {math.hypot(b[0], b[2]):5.2f} T")
    say(f"wrote {args.output} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
