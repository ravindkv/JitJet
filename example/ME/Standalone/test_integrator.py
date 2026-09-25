"""Numerical physics checks: python test_integrator.py (no LHAPDF/PYTHIA needed)."""
import math
from dataclasses import replace
import numpy as np
from scipy.integrate import quad
from qqbar_bbbar_xsec import Config, integrate, dsigma_dcostheta, sigma_hat_total, scale_q2


def angular_integral(shat, mb, alpha, ptmin=0.0, ptmax=None):
    p2 = shat/4-mb**2
    if p2 <= ptmin**2:
        return 0.0
    rho = 4*mb**2/shat
    hi = math.sqrt(1-ptmin**2/p2)
    lo = math.sqrt(1-ptmax**2/p2) if ptmax is not None and ptmax**2 < p2 else 0.0
    # Analytic antiderivative, independent of production integration map.
    primitive = lambda z: (1+rho)*z + (1-rho)*z**3/3
    return (2*math.pi*alpha**2*math.sqrt(1-rho)/(9*shat)
            * (primitive(hi)-primitive(lo)) * 3.89379372e8)


class FlatDensityPDF:
    """Test-only f_q(x)=f_qbar(x)=1. This is NOT a physical proton PDF."""
    xMin, xMax, q2Min, q2Max = 1e-20, 1.0, 1e-20, 1e30
    def xfxQ2(self, flavour, x, q2):
        return x


def run():
    shat = 0.032001686*0.067846217*13000**2
    assert math.isclose(sigma_hat_total(shat, 4.8, .108), 11.5215613118, rel_tol=1e-10)
    for energy in (9.7, 20.0, 120.24, 605.749):
        val = quad(lambda c: dsigma_dcostheta(energy**2, c, 4.8, .108), -1, 1)[0]
        assert math.isclose(val, sigma_hat_total(energy**2, 4.8, .108), rel_tol=1e-12)
    assert sigma_hat_total(8**2, 4.8, .108) == 0
    assert scale_q2(2, 4, 100, 100**2, 4.8**2, 200**2) == 4*(100**2+4.8**2)
    assert scale_q2(5, 4, 100, 100**2, 4.8**2, 200**2) == 100

    cfg = Config(ecm=100, ptmin=6, ptmax=25, alpha_backend="fixed",
                 alpha_fixed=.2, nquark=1, power=14, repeats=4)
    for c in (cfg, replace(cfg, ptmin=0, ptmax=None), replace(cfg, mhatmin=50, mhatmax=70)):
        tau_lo = max(4*(c.mb**2+c.ptmin**2), c.mhatmin**2)/c.ecm**2
        tau_hi = min(1, c.mhatmax**2/c.ecm**2 if c.mhatmax else 1)
        # Explicit tau integral for f=1; length of Y interval = -log(tau).
        def reference(tau):
            return (-2*math.log(tau) * angular_integral(tau*c.ecm**2, c.mb,
                                                       c.alpha_fixed, c.ptmin, c.ptmax))
        expected = quad(reference, tau_lo, tau_hi, epsabs=1e-5, epsrel=1e-9, limit=300)[0]
        actual = integrate(c, raw_pdf=FlatDensityPDF(), progress=False)
        assert abs(actual["sigma_pb"]-expected) < max(7*actual["error_pb"], 3e-4*expected)
        print(f"Jacobian/beam sum/cut check: {actual['sigma_pb']:.5f} vs {expected:.5f} pb")
    empty = integrate(replace(cfg, ptmin=60, ptmax=None), FlatDensityPDF(), False)
    assert empty["sigma_pb"] == 0
    print("All analytic normalization, threshold, scale and phase-space checks passed.")


if __name__ == "__main__":
    run()
