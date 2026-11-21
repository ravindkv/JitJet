#!/usr/bin/env python3
"""The CMS detector in three dimensions from the CMS geometry itself, with ROOT's TGeo: the cut-away picture of
chapter 9 (the picture to carry and the film), ray-traced through the simulation geometry of Run 3.

The geometry is the public file of the CMS event display, cmsSimGeo2021.root of
https://github.com/cms-data/Fireworks-Geometry (the `tag` 2021 simulation geometry, written with CMSSW_11_1_0_pre2
by Fireworks/Geometry/python/dumpSimGeometry_cfg.py: a TGeoManager with 3398 volumes and 2.1 million nodes; the
tracker, the calorimeters, the solenoid, the yoke and the muon chambers of Run 3). A file dumped from the release
of the chapter (dumpSimGeometry_cfg.py tag=2025 in CMSSW_15_0_5) can be given with -g instead.

The picture is an orthographic view with the quadrant x > 0, y > 0, the one facing the viewer, removed. For every
pixel a straight ray is sent from the viewer into the geometry with TGeo's navigator (InitTrack,
FindNextBoundaryAndStep); it starts where it leaves the removed quadrant, crosses air, gas and vacuum (density below
0.02 g/cm3) and stops in the first solid volume, which is coloured by the subdetector its path belongs to (tracker,
ECAL, HCAL, solenoid, iron of the yoke, muon chambers, HF, beam pipe) and shaded by the normal of the surface it hit;
the two faces of the cut are flat. The cavern, the forward shielding around HF and the ZDC are left out. The loop
over the pixels is compiled C++ (ROOT's interpreter); 1.4 million rays take about a minute.

  PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python draw_cms3d_tgeo.py     # -> cms3d_tgeo.png
  ... draw_cms3d_tgeo.py --pixel 4 -o quick.png                                                   # a quick look

Needs PyROOT and numpy (on this Mac: the ME virtual environment with ROOT 6.32 from Homebrew). The view (azimuth 28,
elevation 16 degrees) and the frame (30 m x 19 m, centred on the collision point) are those of draw_cms3d.py, so
that the picture can replace the schematic one wherever it is placed.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "geometry" / "cmsSimGeo2021.root"
HALF_W, HALF_H = 1500.0, 950.0          # cm: the frame, symmetric about the collision point
#            none        tracker          ECAL            HCAL             solenoid         yoke (iron)     muon chambers    HF               beam pipe
COLOURS = [(0, 0, 0), (150, 190, 235), (90, 175, 95), (235, 160, 70), (175, 175, 180), (200, 45, 45), (238, 232, 222), (150, 120, 190), (120, 120, 125)]
NAMES = ["", "tracker", "ECAL", "HCAL", "solenoid", "iron yoke", "muon chambers", "HF", "beam pipe"]

CPP = r"""
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoMaterial.h"
#include "TString.h"
#include <cmath>
namespace jitjet {
// the subdetector of the current position, 0 if it is to be seen through
int category(TGeoManager* geo) {
  TGeoVolume* v = geo->GetCurrentVolume();
  if (!v || !v->GetMaterial() || v->GetMaterial()->GetDensity() < 0.02) return 0;
  TString p(geo->GetPath());
  if (p.Contains("/tracker:Tracker") || p.Contains("/trackerpixelnose:")) return 1;
  if (p.Contains("/eregalgo:ECAL")) return 2;
  if (p.Contains("/hcalalgo:HCal")) return 3;
  if (p.Contains("/hcalforwardalgo:VCAL")) return 7;
  if (p.Contains("/beampipe:")) return 8;
  if (p.Contains("/muonBase:MUON")) {
    if (p.Contains("/mgnt:MGNT")) return 4;
    TString m(v->GetMaterial()->GetName());
    if (p.Contains("muonYoke:") || m.Contains("Steel") || m.Contains("Iron")) return 5;
    return 6;
  }
  return 0;                                  // the cavern, the forward shielding, the ZDC, the beam line
}
// one ray per pixel: cat[i] and the shade[i] of the surface; e1, e2 span the screen, v points to the viewer
void trace(TGeoManager* geo, int nx, int ny, double halfw, double halfh, const double* e1, const double* e2, const double* v,
           const double* light, int* cat, float* shade, float* depth) {
  const double far = 4000.0;
  double dir[3] = {-v[0], -v[1], -v[2]};
  for (int iy = 0; iy < ny; ++iy) {
    double w = halfh - (iy + 0.5) * 2 * halfh / ny;
    for (int ix = 0; ix < nx; ++ix) {
      double u = -halfw + (ix + 0.5) * 2 * halfw / nx;
      int i = iy * nx + ix;
      cat[i] = 0; shade[i] = 0; depth[i] = 0;
      double p[3];
      for (int k = 0; k < 3; ++k) p[k] = u * e1[k] + w * e2[k] + far * v[k];
      // the removed quadrant x > 0, y > 0: the ray is inside it until t = min(x/vx, y/vy)
      double t0 = 0.0; double n0[3] = {0, 0, 0}; bool cutface = false;
      if (p[0] > 0 && p[1] > 0) {
        double tx = p[0] / v[0], ty = p[1] / v[1];
        t0 = (tx < ty ? tx : ty) + 1e-3;
        n0[tx < ty ? 0 : 1] = 1.0; cutface = true;
      }
      double q[3] = {p[0] + t0 * dir[0], p[1] + t0 * dir[1], p[2] + t0 * dir[2]};
      geo->InitTrack(q, dir);
      double t = t0; double n[3] = {n0[0], n0[1], n0[2]};
      for (int step = 0; step < 3000; ++step) {
        if (geo->IsOutside()) break;
        int c = category(geo);
        if (c) {
          if (!(step == 0 && cutface)) { const double* nn = geo->FindNormalFast(); if (nn) { n[0] = nn[0]; n[1] = nn[1]; n[2] = nn[2]; } }
          double nv = n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
          if (nv < 0) { n[0] = -n[0]; n[1] = -n[1]; n[2] = -n[2]; }
          double nl = n[0] * light[0] + n[1] * light[1] + n[2] * light[2];
          cat[i] = c; shade[i] = (step == 0 && cutface) ? (n0[1] > 0 ? 1.0 : 0.86) : 0.50 + 0.50 * (nl > 0 ? nl : 0.0);
          depth[i] = t;
          break;
        }
        geo->FindNextBoundaryAndStep(1.e30, kFALSE);
        t += geo->GetStep();
        if (t > 2 * far) break;
      }
    }
  }
}
}
"""


def write_png(path: Path, rgba: np.ndarray):
    """A PNG from an (h, w, 4) uint8 array, with the standard library only."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def view_vectors(azimuth=28.0, elevation=16.0):
    """The view of draw_cms3d.View: towards the viewer v, screen right e1 (about +z), screen up e2 (about +y), the light."""
    a, b = math.radians(azimuth), math.radians(elevation)
    v = np.array([math.cos(b) * math.cos(a), math.sin(b), -math.cos(b) * math.sin(a)])
    e1 = np.array([math.sin(a), 0.0, math.cos(a)])
    e2 = np.cross(e1, v)
    light = np.array([0.55, 0.75, -0.35])
    return v, e1, e2, light / np.linalg.norm(light)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-g", "--geometry", type=Path, default=GEOMETRY, help="a ROOT file with the TGeoManager of the CMS simulation geometry")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "cms3d_tgeo.png")
    ap.add_argument("--pixel", type=float, default=2.0, help="the size of a pixel in cm (2 cm: 1500 x 950 pixels)")
    args = ap.parse_args()
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    f = ROOT.TFile.Open(str(args.geometry))
    geo = next((k.ReadObj() for k in f.GetListOfKeys() if k.GetClassName() == "TGeoManager"), None)
    if geo is None:
        sys.exit(f"no TGeoManager in {args.geometry}")
    tag = f.Get("tag").GetTitle() if f.Get("tag") else "?"
    rel = f.Get("CMSSW_VERSION").GetTitle() if f.Get("CMSSW_VERSION") else "?"
    print(f"geometry {args.geometry.name}: tag {tag}, {rel}, {geo.GetListOfVolumes().GetEntries()} volumes, top volume {geo.GetTopVolume().GetName()}")
    ROOT.gInterpreter.Declare(CPP)
    nx, ny = int(round(2 * HALF_W / args.pixel)), int(round(2 * HALF_H / args.pixel))
    v, e1, e2, light = view_vectors()
    cat = np.zeros(nx * ny, dtype=np.int32)
    shade = np.zeros(nx * ny, dtype=np.float32)
    depth = np.zeros(nx * ny, dtype=np.float32)
    t0 = time.time()
    ROOT.jitjet.trace(geo, nx, ny, HALF_W, HALF_H, e1, e2, v, light, cat, shade, depth)
    print(f"{nx} x {ny} rays in {time.time() - t0:.0f} s")
    cat, shade, depth = cat.reshape(ny, nx), shade.reshape(ny, nx), depth.reshape(ny, nx)
    for c in range(1, len(NAMES)):
        print(f"   {NAMES[c]:<14s} {100 * (cat == c).mean():5.1f} % of the pixels")
    col = np.array(COLOURS, dtype=float)[cat] * shade[..., None]
    # a thin dark outline where the depth jumps or the subdetector changes: the edges of the volumes
    edge = np.zeros_like(cat, dtype=bool)
    for ax in (0, 1):
        d = np.abs(np.diff(depth, axis=ax)) > 12.0
        k = np.diff(cat, axis=ax) != 0
        e = d | k
        if ax == 0:
            edge[1:, :] |= e
        else:
            edge[:, 1:] |= e
    col[edge & (cat > 0)] *= 0.72
    rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(cat > 0, 255, 0)
    write_png(args.output, rgba)
    print(f"wrote {args.output} ({nx} x {ny} pixels, {2 * HALF_W / 100:.0f} m x {2 * HALF_H / 100:.0f} m, centred on the collision point)")


if __name__ == "__main__":
    main()
