#!/usr/bin/env python3
"""Independent LO integration of PYTHIA process 124, pp -> (q qbar -> b bbar).

Dependencies: numpy, scipy, LHAPDF6 Python bindings + requested PDF member.
The default alpha_s backend additionally needs pythia8 or pythia8mc. An alpha_s
table exported by export_pythia_xsec.h removes that dependency.

Examples:
  python qqbar_bbbar_xsec.py
  python qqbar_bbbar_xsec.py --config pythia_snapshot.json
  python qqbar_bbbar_xsec.py --ptmin 50 --ptmax 100
  python qqbar_bbbar_xsec.py --scan-ptmin 20,30,50,100 --output scan.json
  python qqbar_bbbar_xsec.py --mur-factor 2 --muf-factor 2

All energies/masses are GeV; squared scales are GeV^2; reported xsecs are pb.
This is the Born hard-process rate, not a showered/fiducial b-jet prediction.
No adjustment or fit to the reference PYTHIA cross section is performed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc

GEV2_TO_PB = 3.89379372e8
FLAVOUR_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b"}


@dataclass
class Config:
    # Defaults inferred from the user's log; export an initialized generator
    # to resolve settings/version/PDF-member information absent from that log.
    ecm: float = 13600.0
    mb: float = 4.8
    pdf: str = "NNPDF31_nnlo_as_0118"
    member: int = 0
    ptmin: float = 30.0
    ptmax: float | None = None
    mhatmin: float = 0.0
    mhatmax: float | None = None
    # Optional extra cuts on BOTH Born quarks; rapidity is not pseudorapidity.
    ymax: float | None = None
    etamax: float | None = None
    nquark: int = 5
    # PYTHIA codes 1,2,3 all give mT^2 for equal outgoing masses; 4=sHat, 5=fixed.
    renorm_scale: int = 2
    factor_scale: int = 1
    renorm_mult_fac: float = 1.0  # multiplies Q^2, not Q
    factor_mult_fac: float = 1.0
    renorm_fix_scale: float = 10000.0  # fixed Q^2
    factor_fix_scale: float = 10000.0
    kfactor: float = 1.0
    alpha_backend: str = "pythia"  # pythia | table | lhapdf | fixed
    alphas_mz: float = 0.118
    alpha_order: int = 2
    alpha_nfmax: int = 6
    alpha_fixed: float = 0.118
    alpha_table: list | None = None  # ordered [[Q2, alphaS], ...]
    pdf_extrapolate: bool = False
    clip_negative_pdf: bool = True
    symmetrize_heavy_sea: bool = True
    power: int = 16  # 2**power Sobol points per independently scrambled repeat
    repeats: int = 8
    seed: int = 24680
    reference_pb: float = 18440.0
    reference_error_pb: float = 97.17

    def validate(self):
        for key, val in asdict(self).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if not math.isfinite(val):
                    raise ValueError(f"{key} must be finite; use null for no upper cut")
        if self.ecm <= 0 or self.mb <= 0 or self.ptmin < 0 or self.mhatmin < 0:
            raise ValueError("Require ecm>0, mb>0, ptmin>=0, mhatmin>=0")
        for lower, upper, name in [(self.ptmin, self.ptmax, "pt"),
                                    (self.mhatmin, self.mhatmax, "mhat")]:
            if upper is not None and upper <= lower:
                raise ValueError(f"{name}max must be larger than {name}min")
        if any(v is not None and v <= 0 for v in [self.ymax, self.etamax]):
            raise ValueError("ymax and etamax must be positive or null")
        if not isinstance(self.nquark, int) or not 1 <= self.nquark <= 5:
            raise ValueError("nquark must be an integer in [1,5]")
        if self.renorm_scale not in range(1, 6) or self.factor_scale not in range(1, 6):
            raise ValueError("Supported PYTHIA scale codes are 1..5")
        if min(self.renorm_mult_fac, self.factor_mult_fac,
               self.renorm_fix_scale, self.factor_fix_scale) <= 0:
            raise ValueError("Scale multipliers and fixed Q^2 must be positive")
        if self.kfactor < 0 or self.alphas_mz <= 0 or self.alpha_fixed < 0:
            raise ValueError("Invalid coupling or K factor")
        if self.alpha_backend not in ("pythia", "table", "lhapdf", "fixed"):
            raise ValueError("Unknown alpha_backend")
        if self.alpha_order not in (0, 1, 2, 3) or self.alpha_nfmax not in (5, 6):
            raise ValueError("Invalid alpha_s order or maximum running flavour")
        if not isinstance(self.member, int) or self.member < 0:
            raise ValueError("member must be a nonnegative integer")
        if not isinstance(self.power, int) or not 6 <= self.power <= 23:
            raise ValueError("power must be an integer in [6,23]")
        if not isinstance(self.repeats, int) or self.repeats < 2:
            raise ValueError("At least two independent scrambles are required")
        if self.reference_pb <= 0 or self.reference_error_pb < 0:
            raise ValueError("Invalid reference cross section/error")


def dsigma_dcostheta(shat, costheta, mb, alpha_s):
    """Spin/colour averaged LO d(sigma_hat)/dcos(theta), in pb.

    PDG Eq. (51.20), or PYTHIA Sigma2qqbar2QQbar::sigmaKin:
    pi alpha_s^2 beta/(9 sHat) * [1+c^2+rho(1-c^2)].
    Incoming partons are massless. theta is in their centre-of-mass frame.
    """
    rho = 4.0 * mb * mb / shat
    if rho >= 1.0:
        return 0.0
    return (math.pi * alpha_s**2 * math.sqrt(1.0 - rho) / (9.0 * shat)
            * (1.0 + costheta**2 + rho * (1.0 - costheta**2)) * GEV2_TO_PB)


def sigma_hat_total(shat, mb, alpha_s):
    """No cuts, fixed alpha_s, analytic angular integral; output pb."""
    rho = 4.0 * mb * mb / shat
    if rho >= 1.0:
        return 0.0
    return (4.0 * math.pi * alpha_s**2 * math.sqrt(1-rho) * (2+rho)
            / (27.0 * shat) * GEV2_TO_PB)


def scale_q2(code, multiplier, fixed_q2, pt2, mb2, shat):
    if code == 5:
        return fixed_q2  # PYTHIA ignores multiplier for its fixed-scale option
    return multiplier * (pt2 + mb2 if code in (1, 2, 3) else shat)


def load_pythia_module():
    for name in ("pythia8", "pythia8mc"):
        try:
            return importlib.import_module(name)
        except ImportError:
            pass
    raise RuntimeError("Native alpha_s needs pythia8 or pythia8mc. Install its Python "
                       "bindings, or use an exported snapshot (alpha_backend=table).")


def make_alpha(cfg, pdf):
    if cfg.alpha_backend == "fixed":
        return lambda q2: cfg.alpha_fixed
    if cfg.alpha_backend == "lhapdf":
        # This is deliberately opt-in: NNLO PDF alpha_s is not necessarily the
        # coupling/order used by PYTHIA's LO hard matrix element.
        return pdf.alphasQ2
    if cfg.alpha_backend == "pythia":
        module = load_pythia_module()
        alpha = module.AlphaStrong()
        alpha.init(cfg.alphas_mz, cfg.alpha_order, cfg.alpha_nfmax, False)
        return alpha.alphaS  # retain the native object through its bound method
    if cfg.alpha_table is None:
        raise ValueError("table backend requires alpha_table from the exporter")
    values = np.asarray(cfg.alpha_table, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 4
            or not np.isfinite(values).all() or (values <= 0).any()
            or not (np.diff(values[:, 0]) > 0).all()):
        raise ValueError("alpha_table must contain increasing positive [Q2, alphaS] rows")
    interp = PchipInterpolator(np.log(values[:, 0]), values[:, 1], extrapolate=False)
    def alpha(q2):
        if not values[0, 0] <= q2 <= values[-1, 0]:
            raise ValueError(f"Q2={q2:g} outside exported alpha_s table; re-export "
                             "a wider range or select the native backend")
        return float(interp(math.log(q2)))
    return alpha


class ProtonPDF:
    """LHAPDF adapter matching PYTHIA's proton PDF wrapper conventions."""
    def __init__(self, raw, cfg):
        self.raw, self.cfg = raw, cfg
        self.frozen_calls = 0
        self.negative_values = 0

    def all_quarks(self, x, q2):
        xeval = min(x, self.raw.xMax)
        if not self.cfg.pdf_extrapolate:
            xeval = max(xeval, self.raw.xMin)
        qeval = min(max(q2, self.raw.q2Min), self.raw.q2Max)
        if xeval != x or qeval != q2:
            self.frozen_calls += 1
        # Two vectors: x*f_q and x*f_antiquark. Never divide by x here:
        # the log(tau) phase-space Jacobian cancels the 1/(x1*x2).
        pos, neg = [], []
        for q in range(1, self.cfg.nquark + 1):
            a = self.raw.xfxQ2(q, xeval, qeval)
            b = a if q >= 4 and self.cfg.symmetrize_heavy_sea else self.raw.xfxQ2(-q, xeval, qeval)
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("PDF returned a non-finite value")
            self.negative_values += int(a < 0) + int(b < 0)
            if self.cfg.clip_negative_pdf:
                a, b = max(a, 0.0), max(b, 0.0)
            pos.append(a)
            neg.append(b)
        return pos, neg


class BornIntegrand:
    """Map unit cube -> log(tau), pair boost Y, allowed |cos(theta)|.

    dx1 dx2 = d(tau) dY = tau dlog(tau) dY.
    f1*f2 = (x1*f1)(x2*f2)/tau. These tau factors cancel exactly.
    Both beam orientations are summed in the luminosity. The QCD matrix
    element and BOTH-quark absolute y/eta cuts are even in cos(theta), so
    integrate the positive angular interval with a factor 2.
    """
    def __init__(self, cfg, pdf, alpha):
        self.cfg, self.pdf, self.alpha = cfg, pdf, alpha
        self.mb2, self.s = cfg.mb**2, cfg.ecm**2
        lo = max(4 * (self.mb2 + cfg.ptmin**2), cfg.mhatmin**2)
        hi = min(self.s, cfg.mhatmax**2 if cfg.mhatmax is not None else self.s)
        self.empty = lo >= hi
        self.loglo = math.log(lo / self.s)
        self.logwidth = 0.0 if self.empty else math.log(hi / lo)

    def __call__(self, point):
        cfg = self.cfg
        if self.empty:
            return np.zeros(cfg.nquark)
        logtau = self.loglo + self.logwidth * point[0]
        tau = math.exp(logtau)
        shat = tau * self.s
        ymax_kin = -0.5 * logtau
        boost = (2.0 * point[1] - 1.0) * ymax_kin
        x1 = math.exp(0.5 * logtau + boost)
        x2 = math.exp(0.5 * logtau - boost)
        p2 = max(0.0, shat / 4.0 - self.mb2)
        if p2 <= cfg.ptmin**2 or ymax_kin <= 0:
            return np.zeros(cfg.nquark)
        cmax = math.sqrt(max(0.0, 1.0 - cfg.ptmin**2 / p2))
        cmin = 0.0
        if cfg.ptmax is not None and cfg.ptmax**2 < p2:
            cmin = math.sqrt(1.0 - cfg.ptmax**2 / p2)
        # Stable subtraction for very narrow forward angular intervals.
        span = cmax - cmin
        if cmin > 0:
            span = (cfg.ptmax**2 - cfg.ptmin**2) / (p2 * (cmax + cmin))
        c = cmin + span * point[2]
        pt2 = p2 * (1.0 - c) * (1.0 + c)
        beta = math.sqrt(1.0 - 4.0 * self.mb2 / shat)
        ystar = math.atanh(min(beta * c, 1.0 - 1e-15))
        y1, y2 = boost + ystar, boost - ystar
        if cfg.ymax is not None and max(abs(y1), abs(y2)) > cfg.ymax:
            return np.zeros(cfg.nquark)
        if cfg.etamax is not None:
            if pt2 <= 0:
                return np.zeros(cfg.nquark)
            mt_over_pt = math.sqrt((pt2 + self.mb2) / pt2)
            eta1 = math.asinh(mt_over_pt * math.sinh(y1))
            eta2 = math.asinh(mt_over_pt * math.sinh(y2))
            if max(abs(eta1), abs(eta2)) > cfg.etamax:
                return np.zeros(cfg.nquark)
        qr2 = scale_q2(cfg.renorm_scale, cfg.renorm_mult_fac,
                      cfg.renorm_fix_scale, pt2, self.mb2, shat)
        qf2 = scale_q2(cfg.factor_scale, cfg.factor_mult_fac,
                      cfg.factor_fix_scale, pt2, self.mb2, shat)
        a = self.alpha(qr2)
        aq, abar = self.pdf.all_quarks(x1, qf2)
        bq, bbar = self.pdf.all_quarks(x2, qf2)
        weight = (self.logwidth * 2*ymax_kin * 2*span * cfg.kfactor
                  * dsigma_dcostheta(shat, c, cfg.mb, a))
        return np.array([weight * (aq[i]*bbar[i] + abar[i]*bq[i])
                         for i in range(cfg.nquark)])


def integrate(cfg, raw_pdf=None, progress=True):
    cfg.validate()
    if raw_pdf is None:
        try:
            import lhapdf
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import LHAPDF6 using {sys.executable} "
                f"(Python {sys.version.split()[0]}): {exc}. "
                "Use a Python interpreter with matching LHAPDF bindings. "
                "For the project-local setup run .venv/bin/python "
                "qqbar_bbbar_xsec.py (with your original options); see README.md. "
                "LHAPDF_DATA_PATH locates PDF data, not Python bindings."
            ) from exc
        raw_pdf = lhapdf.mkPDF(cfg.pdf, cfg.member)
    pdf = ProtonPDF(raw_pdf, cfg)
    alpha = make_alpha(cfg, raw_pdf)
    fn = BornIntegrand(cfg, pdf, alpha)
    repeats = []
    for r in range(cfg.repeats):
        points = qmc.Sobol(d=3, scramble=True, seed=cfg.seed+r).random_base2(cfg.power)
        sums = np.zeros(cfg.nquark)
        for point in points:
            sums += fn(point)
        repeats.append(sums / len(points))
        if progress:
            print(f"  repeat {r+1}/{cfg.repeats}: {sum(repeats[-1])/1000:.6f} nb", flush=True)
    repeats = np.asarray(repeats)
    means = repeats.mean(axis=0)
    errs = repeats.std(axis=0, ddof=1) / math.sqrt(cfg.repeats)
    total_repeats = repeats.sum(axis=1)
    total = float(total_repeats.mean())
    error = float(total_repeats.std(ddof=1) / math.sqrt(cfg.repeats))
    # Sum repeat totals before estimating error: flavours are correlated.
    config_record = asdict(cfg)
    table = config_record.pop("alpha_table", None)
    if table is not None:
        config_record["alpha_table_points"] = len(table)
    return {
        "config": config_record, "sigma_pb": total, "error_pb": error,
        "sigma_nb": total / 1000, "error_nb": error / 1000,
        "relative_error_percent": 100*error/total if total else None,
        "ratio_to_reference": total / cfg.reference_pb,
        "difference_over_combined_integration_error": (
            (total-cfg.reference_pb) / math.hypot(error, cfg.reference_error_pb)
            if math.hypot(error, cfg.reference_error_pb) > 0 else None),
        "by_flavour_pb": {FLAVOUR_NAMES[q]: {"sigma": float(means[q-1]),
                                             "error": float(errs[q-1])}
                          for q in range(1, cfg.nquark+1)},
        "repeat_totals_pb": total_repeats.tolist(),
        "pdf_frozen_calls": pdf.frozen_calls,
        "negative_pdf_values": pdf.negative_values,
        "alpha_s_at_Q2_1178": alpha(1178.0),
        "reference_note": "18.44 nb applies only to the original settings. "
                          "Cut/parameter scans need new matching PYTHIA runs.",
    }


def show_result(result):
    print(f"\nTOTAL = {result['sigma_nb']:.6f} +/- {result['error_nb']:.6f} nb")
    print(f"      = {result['sigma_pb']:.3f} +/- {result['error_pb']:.3f} pb")
    for name, val in result["by_flavour_pb"].items():
        print(f"  {name} {name}bar: {val['sigma']/1000:.6f} +/- {val['error']/1000:.6f} nb")
    print(f"Ratio to supplied reference: {result['ratio_to_reference']:.6f}")
    print(f"alpha_s(Q^2=1178 GeV^2) = {result['alpha_s_at_Q2_1178']:.8f}")
    print("Errors are integration errors, not QCD scale/PDF uncertainties.")
    if result["pdf_frozen_calls"]:
        print(f"PDF evaluations frozen at grid boundaries: {result['pdf_frozen_calls']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON configuration or exported snapshot")
    for name, typ in [("ecm", float), ("mb", float), ("pdf", str), ("member", int),
                      ("ptmin", float), ("ptmax", float), ("mhatmin", float), ("mhatmax", float),
                      ("ymax", float), ("etamax", float), ("nquark", int),
                      ("renorm-scale", int), ("factor-scale", int),
                      ("renorm-mult-fac", float), ("factor-mult-fac", float),
                      ("renorm-fix-scale", float), ("factor-fix-scale", float),
                      ("kfactor", float), ("alpha-backend", str), ("alphas-mz", float),
                      ("alpha-order", int), ("alpha-nfmax", int), ("alpha-fixed", float),
                      ("power", int), ("repeats", int), ("seed", int),
                      ("reference-pb", float), ("reference-error-pb", float)]:
        p.add_argument("--"+name, type=typ, default=None)
    p.add_argument("--mur-factor", type=float, help="Multiply mu_R by this factor; squares it for Q^2")
    p.add_argument("--muf-factor", type=float, help="Multiply mu_F by this factor; squares it for Q^2")
    p.add_argument("--scan-ptmin", help="Comma-separated lower pTHat cuts")
    p.add_argument("--signed-pdfs", action="store_true", help="Keep negative PDFs (changes PYTHIA convention)")
    p.add_argument("--output", type=Path, help="Write full results to JSON")
    args = p.parse_args()
    cfg = Config()
    if args.config:
        data = json.loads(args.config.read_text())
        unknown = set(data) - {f.name for f in fields(Config)} - {"metadata"}
        if unknown:
            p.error(f"Unknown configuration fields: {sorted(unknown)}")
        cfg = replace(cfg, **{k: v for k, v in data.items() if k != "metadata"})
    for field in fields(Config):
        val = getattr(args, field.name, None)
        if val is not None:
            setattr(cfg, field.name, val)
    if cfg.alpha_backend == "table" and any(getattr(args, x) is not None
                                            for x in ("alphas_mz", "alpha_order", "alpha_nfmax")):
        p.error("Changing alpha_s parameters requires --alpha-backend pythia or a newly exported table")
    if args.mur_factor is not None:
        if args.mur_factor <= 0:
            p.error("--mur-factor must be positive")
        if cfg.renorm_scale == 5:
            cfg.renorm_fix_scale *= args.mur_factor**2
        else:
            cfg.renorm_mult_fac *= args.mur_factor**2
    if args.muf_factor is not None:
        if args.muf_factor <= 0:
            p.error("--muf-factor must be positive")
        if cfg.factor_scale == 5:
            cfg.factor_fix_scale *= args.muf_factor**2
        else:
            cfg.factor_mult_fac *= args.muf_factor**2
    if args.signed_pdfs:
        cfg.clip_negative_pdf = False
    cuts = [cfg.ptmin] if args.scan_ptmin is None else [float(x) for x in args.scan_ptmin.split(",")]
    for cut in cuts:
        replace(cfg, ptmin=cut).validate()
    print(f"PDF: {cfg.pdf}/{cfg.member}; eCM={cfg.ecm:g}; mb={cfg.mb:g}; alpha_s={cfg.alpha_backend}")
    print("Reference 18.44 nb is meaningful only for matching the original PYTHIA settings.")
    results = []
    for cut in cuts:
        print(f"\npTHat >= {cut:g} GeV; pTHat upper limit={cfg.ptmax}", flush=True)
        result = integrate(replace(cfg, ptmin=cut))
        show_result(result)
        results.append(result)
    if len(results) > 1:
        print("\npTHatMin [GeV]     sigma [nb]       integration error [nb]")
        for result in results:
            print(f"{result['config']['ptmin']:12g} {result['sigma_nb']:16.6f} {result['error_nb']:16.6f}")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
