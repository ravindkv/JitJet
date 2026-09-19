#ifndef EXPORT_PYTHIA_XSEC_H
#define EXPORT_PYTHIA_XSEC_H

// Include this file in the user's existing generator and call immediately
// AFTER successful pythia.init(), on the hard-process generator instance:
//   exportPythiaXsec(pythia, "pythia_snapshot.json");
// For direct use of the resulting coupling table, Python needs only LHAPDF,
// NumPy and SciPy, not PYTHIA's Python bindings.

#include "Pythia8/Pythia.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

inline void exportPythiaXsec(Pythia8::Pythia& p, const std::string& filename) {
  auto& s = p.settings;
  if (!s.flag("ProcessLevel:all") || !s.flag("HardQCD:qqbar2bbbar"))
    throw std::runtime_error("Export from initialized hard-process PYTHIA with qqbar2bbbar enabled");
  if (s.mode("Beams:frameType") != 1 || s.mode("Beams:idA") != 2212 || s.mode("Beams:idB") != 2212)
    throw std::runtime_error("Exporter supports symmetric-energy proton-proton beams only");
  if (s.flag("PDF:useHardNPDFA") || s.flag("PDF:useHardNPDFB"))
    throw std::runtime_error("Nuclear PDF modifications are not implemented by this integrator");
  bool hard = s.flag("PDF:useHard");
  std::string pdfA = s.word(hard ? "PDF:pHardSet" : "PDF:pSet");
  std::string pdfB = s.word(hard ? "PDF:pHardSetB" : "PDF:pSetB");
  if (pdfB != "void" && pdfB != pdfA)
    throw std::runtime_error("This integrator requires the same proton PDF in both beams");
  const std::string prefix = "LHAPDF6:";
  if (pdfA.compare(0, prefix.size(), prefix) != 0)
    throw std::runtime_error("Use an explicit LHAPDF6:SET[/MEMBER] for a reproducible external PDF comparison");
  pdfA = pdfA.substr(prefix.size());
  int member = 0;
  auto slash = pdfA.find('/');
  if (slash != std::string::npos) {
    member = std::stoi(pdfA.substr(slash + 1));
    pdfA = pdfA.substr(0, slash);
  }
  if (pdfA.empty() || pdfA.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos)
    throw std::runtime_error("Use a named LHAPDF set in PDF:pSet, with optional /member");
  if (s.mode("SigmaProcess:renormScale2") > 5 || s.mode("SigmaProcess:factorScale2") > 5)
    throw std::runtime_error("This integrator supports scale choices 1..5");

  std::ofstream f(filename);
  if (!f) throw std::runtime_error("Cannot create snapshot file");
  f << std::setprecision(17) << "{\n";
  auto number = [&](const std::string& key, double value) {
    f << "  \"" << key << "\": " << value << ",\n";
  };
  auto upper = [&](const std::string& key, double value, double lower) {
    f << "  \"" << key << "\": ";
    if (value > lower) f << value; else f << "null";
    f << ",\n";
  };
  number("ecm", s.parm("Beams:eCM"));
  number("mb", p.particleData.m0(5));
  f << "  \"pdf\": \"" << pdfA << "\",\n";
  number("member", member);
  number("ptmin", s.parm("PhaseSpace:pTHatMin"));
  upper("ptmax", s.parm("PhaseSpace:pTHatMax"), s.parm("PhaseSpace:pTHatMin"));
  number("mhatmin", s.parm("PhaseSpace:mHatMin"));
  upper("mhatmax", s.parm("PhaseSpace:mHatMax"), s.parm("PhaseSpace:mHatMin"));
  number("nquark", s.mode("PDFinProcess:nQuarkIn"));
  number("renorm_scale", s.mode("SigmaProcess:renormScale2"));
  number("factor_scale", s.mode("SigmaProcess:factorScale2"));
  number("renorm_mult_fac", s.parm("SigmaProcess:renormMultFac"));
  number("factor_mult_fac", s.parm("SigmaProcess:factorMultFac"));
  number("renorm_fix_scale", s.parm("SigmaProcess:renormFixScale"));
  number("factor_fix_scale", s.parm("SigmaProcess:factorFixScale"));
  number("kfactor", s.parm("SigmaProcess:Kfactor"));
  number("alphas_mz", s.parm("SigmaProcess:alphaSvalue"));
  number("alpha_order", s.mode("SigmaProcess:alphaSorder"));
  number("alpha_nfmax", s.mode("StandardModel:alphaSnfmax"));
  f << "  \"pdf_extrapolate\": " << (s.flag("PDF:extrapolate") ? "true" : "false") << ",\n";
  f << "  \"alpha_backend\": \"table\",\n";
  f << "  \"alpha_table\": [\n";
  // Fine logarithmic Q^2 grid with room for scale/energy variations.
  const double q2min = 1.0;
  const double q2max = std::max(1e8, 16.0 * std::pow(s.parm("Beams:eCM"), 2));
  for (int i = 0; i <= 4096; ++i) {
    double q2 = std::exp(std::log(q2min) + (std::log(q2max/q2min) * i / 4096.0));
    f << "    [" << q2 << ", " << p.coupSM.alphaS(q2) << "]" << (i == 4096 ? "\n" : ",\n");
  }
  f << "  ],\n  \"metadata\": {\"source\": \"initialized PYTHIA; hard-process coupling sampled directly\"}\n}\n";
  if (!f) throw std::runtime_error("Failed writing snapshot");
}
#endif
