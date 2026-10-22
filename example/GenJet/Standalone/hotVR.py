#!/usr/bin/env python3
"""HOTVR clustering, step by step, on the nine particles of antiKt.py.

The Heavy Object Tagger with Variable R (Lapsien, Kogler, Haller, arXiv:1606.04961, Section 2):
Cambridge/Aachen ordering (d_ij = DeltaR_ij^2, no momentum in the distance) with a beam distance that
shrinks with the pseudojet's pT,
    d_iB = R_eff(pT_i)^2,     R_eff = clip(rho / pT, R_min, R_max),
and a decision at every merge whose mass passes mu: the mass jump theta m_ij > max(m_i, m_j) must hold
and both branches must carry pT > pT_sub, otherwise the lighter (or the softer) branch is thrown away.
The defaults are the paper's top-tagging values (rho = 600 GeV, R_min = 0.1, R_max = 1.5, mu = 30 GeV,
theta = 0.7, pT_sub = 30 GeV); every one is a command-line option.

The particles are the nine of antiKt.py and caKt.py (rows of ../../Decay/event_Decay.lhe, massless): the
two cores of our b jet (A B, C D), three soft particles at its border (E F G), the leading particle of the
neighbouring gluon jet [4] (H, 1.1 away) and of jet [5] (I, 1.6 away). With R_eff = 1.5 for everything
below 400 GeV the whole neighbourhood is in reach: H is merged as an ordinary particle (the pair mass stays
below mu) and I, whose pair mass passes mu, is groomed away because it is softer than pT_sub. The script is
O(N^3), recomputes every distance at every step and prints every decision; it is a reading aid, not the
validated FastJet-contrib HOTVR plugin. plot_algorithms_GenJet.py runs the same decisions on the 749
particles of the event.

    python3 hotVR.py                                  # the branch checks, the history and a window with the picture
    python3 hotVR.py --no-show --output hotVR.pdf     # the picture as a file (make figures)
    python3 hotVR.py --rho 40 --mu 10 --ptsub 5       # parameters scaled down to a 100 GeV family
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


EPS = 1.0e-12


def delta_phi(phi_a: float, phi_b: float) -> float:
    """Return the signed azimuthal difference in [-pi, pi)."""

    return (phi_a - phi_b + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class FourVector:
    """Minimal Cartesian Lorentz four-vector with (+, -, -, -) metric."""

    energy: float
    px: float
    py: float
    pz: float

    @classmethod
    def from_pt_y_phi_m(
        cls, pt: float, rapidity: float, phi: float, mass: float = 0.0
    ) -> "FourVector":
        if pt < 0.0 or mass < 0.0:
            raise ValueError("pt and mass must be non-negative")
        transverse_mass = math.hypot(pt, mass)
        return cls(
            energy=transverse_mass * math.cosh(rapidity),
            px=pt * math.cos(phi),
            py=pt * math.sin(phi),
            pz=transverse_mass * math.sinh(rapidity),
        )

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            self.energy + other.energy,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
        )

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)

    @property
    def phi(self) -> float:
        return math.atan2(self.py, self.px)

    @property
    def mass(self) -> float:
        mass2 = (
            self.energy * self.energy
            - self.px * self.px
            - self.py * self.py
            - self.pz * self.pz
        )
        return math.sqrt(max(0.0, mass2))

    @property
    def rapidity(self) -> float:
        numerator = self.energy + self.pz
        denominator = self.energy - self.pz
        if numerator <= 0.0 or denominator <= 0.0:
            return math.copysign(math.inf, self.pz)
        return 0.5 * math.log(numerator / denominator)


@dataclass(frozen=True)
class PseudoJet:
    """A four-vector plus clustering ancestry and HOTVR subjets."""

    uid: int
    label: str
    p4: FourVector
    leaves: tuple[str, ...]
    subjets: tuple["PseudoJet", ...] = field(default_factory=tuple, repr=False)

    @property
    def pt(self) -> float:
        return self.p4.pt

    @property
    def mass(self) -> float:
        return self.p4.mass

    @property
    def rapidity(self) -> float:
        return self.p4.rapidity

    @property
    def phi(self) -> float:
        return self.p4.phi


@dataclass(frozen=True)
class HOTVRConfig:
    """HOTVR parameters (the defaults are the paper's top-tagging values)."""

    r_min: float = 0.1
    r_max: float = 1.5
    rho: float = 600.0  # GeV
    mu: float = 30.0  # GeV
    theta: float = 0.7
    pt_sub: float = 30.0  # GeV

    def __post_init__(self) -> None:
        if not 0.0 < self.r_min <= self.r_max:
            raise ValueError("Require 0 < r_min <= r_max")
        if self.rho <= 0.0 or self.mu < 0.0 or self.pt_sub < 0.0:
            raise ValueError("rho must be positive; mu and pt_sub non-negative")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")


@dataclass(frozen=True)
class HistoryStep:
    number: int
    action: str
    distance: float
    message: str


@dataclass(frozen=True)
class DiscardedPseudoJet:
    pseudojet: PseudoJet
    reason: str


@dataclass(frozen=True)
class ClusterResult:
    jets: tuple[PseudoJet, ...]
    discarded: tuple[DiscardedPseudoJet, ...]
    history: tuple[HistoryStep, ...]


def make_pseudojet(
    uid: int,
    label: str,
    pt: float,
    rapidity: float,
    phi: float,
    mass: float = 0.0,
) -> PseudoJet:
    """Build one input pseudojet from familiar collider coordinates."""

    return PseudoJet(
        uid=uid,
        label=label,
        p4=FourVector.from_pt_y_phi_m(pt, rapidity, phi, mass),
        leaves=(label,),
    )


def effective_radius(pt: float, config: HOTVRConfig) -> float:
    """Equation (4): clip rho / pt to [R_min, R_max]."""

    if pt <= 0.0:
        return config.r_max
    return min(config.r_max, max(config.r_min, config.rho / pt))


def pair_distance(a: PseudoJet, b: PseudoJet) -> float:
    """Equation (1) for HOTVR's n=0 choice: d_ij = DeltaR_ij^2."""

    dy = a.rapidity - b.rapidity
    dphi = delta_phi(a.phi, b.phi)
    return dy * dy + dphi * dphi


def beam_distance(a: PseudoJet, config: HOTVRConfig) -> float:
    """Equation (2) for n=0: d_iB = R_eff(pT_i)^2."""

    radius = effective_radius(a.pt, config)
    return radius * radius


def _subjet_leaves(pseudojet: PseudoJet) -> tuple[PseudoJet, ...]:
    """Flatten an existing HOTVR subjet list, or treat the object as one subjet."""

    return pseudojet.subjets if pseudojet.subjets else (pseudojet,)


def _merge(
    a: PseudoJet,
    b: PseudoJet,
    uid: int,
    record_subjets: bool,
) -> PseudoJet:
    if record_subjets:
        subjets = _subjet_leaves(a) + _subjet_leaves(b)
    elif a.subjets or b.subjets:
        # This is rarely reached for physical time-like four-vectors, because a
        # pseudojet that already passed a mass jump normally has mass > mu.
        # Propagating the existing leaves is the least surprising behaviour.
        subjets = _subjet_leaves(a) + _subjet_leaves(b)
    else:
        subjets = ()

    return PseudoJet(
        uid=uid,
        label=f"({a.label}+{b.label})",
        p4=a.p4 + b.p4,
        leaves=a.leaves + b.leaves,
        subjets=subjets,
    )


def hotvr_cluster(
    inputs: Sequence[PseudoJet], config: HOTVRConfig
) -> ClusterResult:
    """Cluster pseudojets using the Section-2 HOTVR decision sequence.

    The implementation deliberately recomputes every distance at each step so
    the correspondence with the paper is easy to inspect.  At exact mass ties,
    which Section 2 does not specify, the lower-pT object is removed; the uid is
    a final deterministic tie-breaker.
    """

    if len({pseudojet.uid for pseudojet in inputs}) != len(inputs):
        raise ValueError("Each input pseudojet must have a unique uid")

    active = list(inputs)
    jets: list[PseudoJet] = []
    discarded: list[DiscardedPseudoJet] = []
    history: list[HistoryStep] = []
    next_uid = max((pseudojet.uid for pseudojet in inputs), default=-1) + 1

    def remove(pseudojet: PseudoJet) -> None:
        active.remove(pseudojet)

    def log(action: str, distance: float, message: str) -> None:
        history.append(HistoryStep(len(history) + 1, action, distance, message))

    while active:
        # Candidate tuple: (distance, kind_order, uid_i, uid_j, kind, i, j).
        # kind_order only resolves the measure-zero case of an exact tie.
        candidates: list[
            tuple[float, int, int, int, str, PseudoJet, PseudoJet | None]
        ] = []
        for a in active:
            candidates.append((beam_distance(a, config), 0, a.uid, -1, "beam", a, None))
        for index, a in enumerate(active):
            for b in active[index + 1 :]:
                uid_a, uid_b = sorted((a.uid, b.uid))
                candidates.append((pair_distance(a, b), 1, uid_a, uid_b, "pair", a, b))

        distance, _, _, _, kind, a, b = min(candidates, key=lambda item: item[:4])

        # Step 1: a beam distance wins, so a is a completed jet.
        if kind == "beam":
            remove(a)
            jets.append(a)
            log(
                "finalize_jet",
                distance,
                f"d_iB = R_eff^2 smallest for {a.label}: finalize jet "
                f"(pT={a.pt:.1f}, m={a.mass:.1f}, R_eff={effective_radius(a.pt, config):.3f}).",
            )
            continue

        assert b is not None
        combined_p4 = a.p4 + b.p4
        combined_mass = combined_p4.mass

        # Step 2: below the mass threshold there is an ordinary recombination.
        if combined_mass <= config.mu + EPS:
            remove(a)
            remove(b)
            merged = _merge(a, b, next_uid, record_subjets=False)
            next_uid += 1
            active.append(merged)
            log(
                "merge_below_mu",
                distance,
                f"Pair {a.label} + {b.label} (dR={math.sqrt(distance):.3f}): m_ij={combined_mass:.1f} <= "
                f"mu={config.mu:.1f}; merge without declaring a new mass-jump subjet boundary.",
            )
            continue

        # Step 3: above mu, test the mass jump theta*m_ij > max(m_i,m_j).
        jump_lhs = config.theta * combined_mass
        jump_rhs = max(a.mass, b.mass)
        if not jump_lhs > jump_rhs + EPS:
            # The paper removes the lower-mass pseudojet.  For exact equal masses,
            # use lower pT and then uid as deterministic pedagogical tie-breaks.
            loser = min((a, b), key=lambda obj: (obj.mass, obj.pt, -obj.uid))
            remove(loser)
            discarded.append(
                DiscardedPseudoJet(loser, "mass-jump condition failed; lower-mass branch removed")
            )
            log(
                "discard_mass_jump",
                distance,
                f"Pair {a.label} + {b.label} (dR={math.sqrt(distance):.3f}): m_ij={combined_mass:.1f} > mu, but "
                f"theta*m_ij={jump_lhs:.1f} <= max(m_i,m_j)={jump_rhs:.1f}; "
                f"discard lower-mass {loser.label}.",
            )
            continue

        # Step 3b(i): a valid jump still requires both branches above pT,sub.
        soft = [obj for obj in (a, b) if obj.pt <= config.pt_sub + EPS]
        if soft:
            for loser in soft:
                remove(loser)
                discarded.append(
                    DiscardedPseudoJet(loser, "mass jump passed, but branch failed pT,sub")
                )
            soft_names = ", ".join(obj.label for obj in soft)
            log(
                "discard_soft",
                distance,
                f"Pair {a.label} + {b.label} (dR={math.sqrt(distance):.3f}): mass jump passes "
                f"(m_ij={combined_mass:.1f} > mu, theta*m_ij={jump_lhs:.1f} > {jump_rhs:.1f}), but {soft_names} has "
                f"pT <= pT,sub={config.pt_sub:.1f}; discard the soft branch.",
            )
            continue

        # Step 3b(ii): keep both branches, merge, and remember their subjets.
        remove(a)
        remove(b)
        merged = _merge(a, b, next_uid, record_subjets=True)
        next_uid += 1
        active.append(merged)
        log(
            "merge_mass_jump",
            distance,
            f"Pair {a.label} + {b.label} (dR={math.sqrt(distance):.3f}): m_ij={combined_mass:.1f} > mu, "
            f"theta*m_ij={jump_lhs:.1f} > {jump_rhs:.1f}, and both branches "
            f"have pT > {config.pt_sub:.1f}; merge and retain subjets.",
        )

    return ClusterResult(
        jets=tuple(sorted(jets, key=lambda jet: jet.pt, reverse=True)),
        discarded=tuple(discarded),
        history=tuple(history),
    )


# ------------------------------------------------------------
# Input particles: the nine of antiKt.py (rows of ../../Decay/event_Decay.lhe,
# massless, rapidity = eta); the comments give the row of the record.
# ------------------------------------------------------------
PARTICLES = (
    ("A", 14.269, -1.3358, -2.7211),   # pi+ [2181], the hardest particle of the b jet
    ("B", 5.414, -1.3592, -2.7120),    # pi- [2239], 0.025 from A
    ("C", 10.567, -1.3893, -2.7700),   # K- [2211], the D0's kaon
    ("D", 4.451, -1.3832, -2.7698),    # pi- [2241], 0.006 from C: the first merge of the whole event
    ("E", 0.680, -1.8008, -2.8491),    # pi+ [1217], the last particle anti-kT added to our real jet
    ("F", 1.857, -1.7408, -2.6406),    # gamma [1538], at the border
    ("G", 0.969, -1.7131, -2.6160),    # gamma [2183], 0.037 from F
    ("H", 8.282, -2.4166, -2.4959),    # Lambda0 [1546], the leading particle of the gluon jet [4], 1.1 away
    ("I", 24.572, -2.0434, -1.3043),   # pi+ [1592], the leading particle of jet [5], 1.6 away
)


def build_toy_events() -> dict[str, tuple[PseudoJet, ...]]:
    """Return the nine particles of our event as pseudojets."""

    return {"Our event": tuple(make_pseudojet(k, label, pt, y, phi) for k, (label, pt, y, phi) in enumerate(PARTICLES))}


def _branch_examples(config: HOTVRConfig) -> dict[str, str]:
    """Exercise the four non-beam branches and fail loudly if logic regresses."""

    cases = {
        "m_ij <= mu -> ordinary merge": (
            (
                make_pseudojet(100, "a", 80.0, 0.00, 0.00),
                make_pseudojet(101, "b", 60.0, 0.05, 0.08),
            ),
            "merge_below_mu",
        ),
        "mass jump + hard branches -> subjets": (
            (
                make_pseudojet(110, "a", 120.0, 0.00, -0.20),
                make_pseudojet(111, "b", 80.0, 0.00, 0.20),
            ),
            "merge_mass_jump",
        ),
        "mass jump + soft branch -> discard": (
            (
                make_pseudojet(120, "hard", 120.0, 0.00, 0.00),
                make_pseudojet(121, "soft", 15.0, 0.80, 0.00),
            ),
            "discard_soft",
        ),
        "failed mass jump -> discard lighter mass": (
            (
                make_pseudojet(130, "massive_core", 500.0, 0.00, 0.00, 170.0),
                make_pseudojet(131, "radiation", 20.0, 0.40, 0.40),
            ),
            "discard_mass_jump",
        ),
    }

    observed: dict[str, str] = {}
    for description, (particles, expected_action) in cases.items():
        result = hotvr_cluster(particles, config)
        actions = {step.action for step in result.history}
        if expected_action not in actions:
            raise AssertionError(
                f"Branch example '{description}' missed expected action "
                f"'{expected_action}'; observed {sorted(actions)}"
            )
        observed[description] = expected_action
    return observed


def _format_labels(labels: Iterable[str]) -> str:
    values = list(labels)
    return ", ".join(values) if values else "none"


def print_result(
    name: str,
    inputs: Sequence[PseudoJet],
    result: ClusterResult,
    config: HOTVRConfig,
) -> None:
    """Print kinematics and the complete clustering history."""

    print(f"\n{'=' * 78}\n{name}: HOTVR with rho = {config.rho:g} GeV, R_min = {config.r_min:g}, R_max = {config.r_max:g}, "
          f"mu = {config.mu:g} GeV, theta = {config.theta:g}, pT_sub = {config.pt_sub:g} GeV")
    print("Input pseudojets:")
    for particle in inputs:
        print(
            f"  {particle.label:>5s}: pT={particle.pt:6.2f} GeV, "
            f"y={particle.rapidity:+7.4f}, phi={particle.phi:+7.4f}, "
            f"m={particle.mass:5.1f} GeV, R_eff={effective_radius(particle.pt, config):.3f}"
        )

    print("Clustering history:")
    for step in result.history:
        print(f"  [{step.number:02d}] {step.action:>18s}  d={step.distance:7.4f}  {step.message}")

    print("Final jets:")
    for index, jet in enumerate(result.jets, start=1):
        subjet_labels = ["+".join(subjet.leaves) for subjet in jet.subjets]
        print(
            f"  J{index}: pT={jet.pt:6.1f} GeV, y={jet.rapidity:+6.3f}, "
            f"phi={jet.phi:+6.3f}, m={jet.mass:6.1f} GeV, "
            f"R_eff={effective_radius(jet.pt, config):.3f}, "
            f"constituents=[{''.join(sorted(jet.leaves))}], "
            f"subjets=[{_format_labels(subjet_labels)}]"
        )
    print(
        "Discarded (groomed away): "
        + _format_labels(
            leaf
            for removed in result.discarded
            for leaf in removed.pseudojet.leaves
        )
    )


def _leaf_outcomes(result: ClusterResult) -> tuple[dict[str, int], set[str]]:
    jet_for_leaf: dict[str, int] = {}
    for jet_index, jet in enumerate(result.jets):
        for leaf in jet.leaves:
            jet_for_leaf[leaf] = jet_index
    discarded = {
        leaf
        for removed in result.discarded
        for leaf in removed.pseudojet.leaves
    }
    return jet_for_leaf, discarded


def _draw_event(
    ax,
    title: str,
    inputs: Sequence[PseudoJet],
    result: ClusterResult,
    config: HOTVRConfig,
) -> None:
    """Draw the nine particles and the HOTVR jets in the (y, phi) plane."""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    palette = ("#1479C9", "#E4572E", "#2E8B57", "#8E5EA2")
    jet_for_leaf, discarded = _leaf_outcomes(result)
    hardest = max(inputs, key=lambda p: sum(pair_distance(p, q) < 0.01 for q in inputs))   # the core's centre (A)
    core: list[PseudoJet] = []

    for particle in inputs:
        if particle.label in discarded:
            color = "#8B8F97"
            marker = "X"
            zorder = 6
        else:
            color = palette[jet_for_leaf.get(particle.label, 0) % len(palette)]
            marker = "o"
            zorder = 5
        size = 30.0 + 6.0 * particle.pt
        ax.scatter(
            particle.rapidity,
            particle.phi,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.3,
            alpha=0.92,
            zorder=zorder,
        )
        if pair_distance(particle, hardest) < 0.01:          # the core: too close to label one by one
            core.append(particle)
            continue
        ax.annotate(
            f"{particle.label} {particle.pt:.1f} GeV",
            (particle.rapidity, particle.phi),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
            color="#20242A",
            ha="left",
            zorder=8,
        )
    if core:
        ax.annotate("the core, within 0.07:\n" + ", ".join(f"{p.label} {p.pt:.1f}" for p in core) + " GeV",
                    (hardest.rapidity, hardest.phi), xytext=(0.62, 0.70), textcoords="axes fraction", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"), zorder=9)

    for jet_index, jet in enumerate(result.jets):
        color = palette[jet_index % len(palette)]
        radius = effective_radius(jet.pt, config)
        ax.add_patch(
            Circle(
                (jet.rapidity, jet.phi),
                radius,
                fill=False,
                color=color,
                linestyle=(0, (5, 4)),
                linewidth=2.0,
                alpha=0.8,
                zorder=1,
            )
        )
        ax.scatter(
            jet.rapidity,
            jet.phi,
            marker="*",
            s=230,
            color="#111111",
            edgecolor="white",
            linewidth=0.8,
            zorder=9,
        )
        ax.annotate(
            f"jet {jet_index + 1}: {jet.pt:.1f} GeV, R_eff = {radius:.2f}",
            (jet.rapidity, jet.phi),
            xytext=(7, -14),
            textcoords="offset points",
            fontsize=9,
            color=color,
            zorder=8,
        )
        for subjet in jet.subjets:
            ax.scatter(
                subjet.rapidity,
                subjet.phi,
                marker="D",
                s=48,
                facecolor="none",
                edgecolor="#111111",
                linewidth=1.2,
                zorder=8,
            )

    leading = result.jets[0]
    removed_labels = [
        leaf
        for removed in result.discarded
        for leaf in removed.pseudojet.leaves
    ]
    summary = (
        f"Leading jet: pT={leading.pt:.1f} GeV, m={leading.mass:.1f} GeV\n"
        f"R_eff={effective_radius(leading.pt, config):.2f}, constituents {''.join(sorted(leading.leaves))}\n"
        f"N_sub={len(leading.subjets)}, groomed away: {_format_labels(removed_labels)}"
    )
    ax.text(
        0.03,
        0.03,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#D5D8DC", alpha=0.94),
        zorder=10,
    )

    ax.set_title(title, fontsize=14, weight="bold", pad=10)
    ax.set_xlabel(r"rapidity $y$")
    ax.set_ylabel(r"azimuth $\phi$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.set_facecolor("#FAFBFC")

    r_lead = effective_radius(leading.pt, config)
    ys = [p.rapidity for p in inputs] + [leading.rapidity - r_lead, leading.rapidity + r_lead]
    ps = [p.phi for p in inputs] + [leading.phi - r_lead, leading.phi + r_lead]
    ax.set_xlim(min(ys) - 0.25, max(ys) + 0.45)
    ax.set_ylim(min(ps) - 0.25, max(ps) + 0.25)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1479C9", markeredgecolor="white", markersize=9, label="kept input"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#8B8F97", markeredgecolor="white", markersize=9, label="groomed away"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="white", markersize=13, label="jet axis"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="none", markeredgecolor="#111111", markersize=7, label="subjet axis"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8.5)


def make_figure(
    events: dict[str, tuple[PseudoJet, ...]],
    results: dict[str, ClusterResult],
    config: HOTVRConfig,
    output: Path,
    dpi: int,
) -> None:
    """Create the explanatory HOTVR figure and save it."""

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.6), height_ratios=(1.0, 1.12))
    radius_ax = fig.add_subplot(grid[0, 0])
    flow_ax = fig.add_subplot(grid[1, 0])
    event_ax = fig.add_subplot(grid[:, 1])

    # Variable-radius curve, using logarithmic sampling without numpy.
    pts = [10.0 ** (0.0 + index * (4.0 - 0.0) / 299.0) for index in range(300)]
    radii = [effective_radius(pt, config) for pt in pts]
    radius_ax.plot(pts, radii, color="#1479C9", linewidth=2.7)
    radius_ax.axhline(config.r_min, color="#6B7280", linewidth=1.0, linestyle=":")
    radius_ax.axhline(config.r_max, color="#6B7280", linewidth=1.0, linestyle=":")
    for name, result in results.items():
        jet = result.jets[0]
        radius_ax.scatter(jet.pt, effective_radius(jet.pt, config), s=65, color="#E4572E", zorder=5)
        radius_ax.annotate(
            f"our jet: {jet.pt:.0f} GeV",
            (jet.pt, effective_radius(jet.pt, config)),
            xytext=(7, -15),
            textcoords="offset points",
            fontsize=9,
            color="#E4572E",
        )
    radius_ax.set_xscale("log")
    radius_ax.set_xlim(1.0, 10000.0)
    radius_ax.set_ylim(0.0, config.r_max * 1.12)
    radius_ax.set_xlabel(r"pseudojet $p_T$ [GeV]")
    radius_ax.set_ylabel(r"$R_{\mathrm{eff}}$")
    radius_ax.set_title("Variable-R rule", fontsize=14, weight="bold")
    radius_ax.grid(True, which="both", color="#E5E7EB", linewidth=0.8)
    radius_ax.text(
        0.04,
        0.08,
        rf"$R_{{\rm eff}}=\mathrm{{clip}}(\rho/p_T)$" "\n"
        rf"$\rho={config.rho:.0f}$ GeV,  $R_{{\min}}={config.r_min:.1f}$,  $R_{{\max}}={config.r_max:.1f}$",
        transform=radius_ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D5D8DC", alpha=0.94),
    )

    flow_ax.axis("off")
    flow_text = (
        "At every iteration (n = 0):\n"
        r"$d_{ij}=\Delta R_{ij}^2$;   $d_{iB}=R_{\rm eff}^2$" "\n\n"
        "Smallest is beam distance\n"
        "  -> finalize pseudojet as a jet\n\n"
        "Smallest is pair distance\n"
        r"  $m_{ij}\leq\mu$ -> merge" "\n"
        r"  $m_{ij}>\mu$ -> test mass jump" "\n"
        r"     $\theta m_{ij}\leq\max(m_i,m_j)$" "\n"
        "       -> remove lower-mass branch\n"
        r"     $\theta m_{ij}>\max(m_i,m_j)$" "\n"
        r"       any $p_T\leq p_{T,\rm sub}$" "\n"
        "         -> remove soft branch\n"
        "       otherwise\n"
        "         -> merge + retain subjets"
    )
    flow_ax.text(
        0.02,
        0.98,
        flow_text,
        va="top",
        ha="left",
        fontsize=11.0,
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="#F7F9FB", edgecolor="#CBD1D8"),
    )
    flow_ax.set_title("Section-2 decision logic", fontsize=14, weight="bold", pad=10)

    for name, particles in events.items():
        _draw_event(event_ax, f"{name}: nine particles under HOTVR", particles, results[name], config)

    fig.suptitle(
        "HOTVR: variable-radius clustering, mass-jump subjets, and in-sequence grooming",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.5,
        0.003,
        "Dashed circles show R_eff around the final jet axis as a schematic scale; exact active areas require ghost particles.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555B65",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Illustrate the HOTVR jet algorithm on nine particles of our event.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hotVR.pdf"),
        help="output figure (default: hotVR.pdf)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="figure resolution (default: 180)")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    parser.add_argument("--no-plot", action="store_true", help="the log only")
    parser.add_argument("--quiet", action="store_true", help="suppress the detailed clustering log")
    parser.add_argument("--rho", type=float, default=600.0, help="rho of R_eff = rho/pT [GeV] (default 600)")
    parser.add_argument("--rmin", type=float, default=0.1, help="R_min (default 0.1)")
    parser.add_argument("--rmax", type=float, default=1.5, help="R_max (default 1.5)")
    parser.add_argument("--mu", type=float, default=30.0, help="mass threshold mu [GeV] (default 30)")
    parser.add_argument("--theta", type=float, default=0.7, help="mass-jump parameter theta (default 0.7)")
    parser.add_argument("--ptsub", type=float, default=30.0, help="pT_sub [GeV] (default 30)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")

    # Select a non-interactive backend before pyplot is imported when requested.
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    config = HOTVRConfig(r_min=args.rmin, r_max=args.rmax, rho=args.rho, mu=args.mu, theta=args.theta, pt_sub=args.ptsub)
    branch_checks = _branch_examples(config)
    events = build_toy_events()
    results = {name: hotvr_cluster(particles, config) for name, particles in events.items()}

    if not args.quiet:
        print("HOTVR branch checks:")
        for description, action in branch_checks.items():
            print(f"  PASS  {description}  [{action}]")
        for name, particles in events.items():
            print_result(name, particles, results[name], config)

    if args.no_plot:
        return
    make_figure(events, results, config, args.output, args.dpi)
    print(f"\nSaved HOTVR illustration to: {args.output.resolve()}")

    if not args.no_show:
        # Reopen the saved figure so interactive and headless executions share
        # exactly the same rendered output.
        import matplotlib.pyplot as plt

        image = plt.imread(args.output)
        plt.figure(figsize=(14.5, 8.0))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
