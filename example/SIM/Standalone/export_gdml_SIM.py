#!/usr/bin/env python3
"""The CMS simulation geometry as a GDML file, the format Geant4 reads.

../../G4/geometry/cmsSimGeo2021.root is the TGeo dump of the geometry CMSSW gives to Geant4 (chapter 9): 3398
volume types, 490 materials, 2.1 million placed volumes. ROOT writes the same tree as GDML, with every solid,
every material (elements and mass fractions) and every placement, and Geant4's G4GDMLParser reads it back.
Nothing is simplified on the way.

Run with an interpreter that has ROOT (here the ME venv with PYTHONPATH=$(root-config --libdir)):

  PYTHONPATH=$(root-config --libdir) ../../ME/Standalone/.venv/bin/python export_gdml_SIM.py [another.root]

Writes ../../G4/geometry/cmsSimGeo2021.gdml (10 MB, not in git: `make gdml` remakes it in two seconds).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEO = HERE.parent.parent / "G4" / "geometry" / "cmsSimGeo2021.root"
OUT = GEO.with_suffix(".gdml")


def main():
    global GEO, OUT
    if len(sys.argv) > 1:                                  # another dump, e.g. one of the 2025 geometry written from CMSSW
        GEO = Path(sys.argv[1]).resolve()
        OUT = GEO.with_suffix(".gdml")
    import ROOT
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    f = ROOT.TFile.Open(str(GEO))
    geo = f.Get("cmsGeo")
    shapes = Counter(v.GetShape().ClassName() for v in geo.GetListOfVolumes())
    geo.Export(str(OUT))
    print(f"{GEO.name}: top volume {geo.GetTopVolume().GetName()}, {geo.GetListOfVolumes().GetEntries()} volume types, "
          f"{geo.GetListOfMaterials().GetEntries()} materials, {geo.CountNodes()} placed volumes")
    print("shapes: " + ", ".join(f"{k[4:]} {v}" for k, v in shapes.most_common()))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
