#!/usr/bin/env python3
"""Cambridge/Aachen clustering, step by step, on the nine particles of antiKt.py.

The same sequential recombination as antiKt.py with the power p = 0 of the generalised-kT family:
    d_ij = DeltaR_ij^2 / R^2,     d_iB = 1.
The momenta no longer enter the distance at all: the closest pair in angle is merged first, whatever
its pT, and a pseudojet becomes a jet once nothing is left within R of it. Same nine particles of our
event (rows of ../../Decay/event_Decay.lhe, massless, so rapidity = eta), same E scheme, same printout,
so the two logs can be read side by side; hotVR.py adds the variable radius and the mass jump.

    python3 caKt.py                         # the step-by-step log and a window with the picture
    python3 caKt.py --no-show -o caKt.pdf   # the log and the picture as a file (make figures)
    python3 caKt.py --radius 0.8
"""
import argparse
import math

R = 0.4

# ------------------------------------------------------------
# Input particles: the nine of antiKt.py (rows of the record)
# ------------------------------------------------------------
particles = [
    {"name": "A", "tag": "pi+ [2181]",     "pt": 14.269, "eta": -1.3358, "phi": -2.7211},
    {"name": "B", "tag": "pi- [2239]",     "pt": 5.414,  "eta": -1.3592, "phi": -2.7120},
    {"name": "C", "tag": "K- [2211]",      "pt": 10.567, "eta": -1.3893, "phi": -2.7700},
    {"name": "D", "tag": "pi- [2241]",     "pt": 4.451,  "eta": -1.3832, "phi": -2.7698},
    {"name": "E", "tag": "pi+ [1217]",     "pt": 0.680,  "eta": -1.8008, "phi": -2.8491},
    {"name": "F", "tag": "gamma [1538]",   "pt": 1.857,  "eta": -1.7408, "phi": -2.6406},
    {"name": "G", "tag": "gamma [2183]",   "pt": 0.969,  "eta": -1.7131, "phi": -2.6160},
    {"name": "H", "tag": "Lambda0 [1546]", "pt": 8.282,  "eta": -2.4166, "phi": -2.4959},
    {"name": "I", "tag": "pi+ [1592]",     "pt": 24.572, "eta": -2.0434, "phi": -1.3043},
]


def four_vector(pt, eta, phi):
    """Massless four-vector (E, px, py, pz) from pT, eta, phi."""
    return [pt * math.cosh(eta), pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)]


def kinematics(p4):
    """pT, rapidity and phi of a four-vector (rapidity = eta for a massless particle)."""
    e, px, py, pz = p4
    pt = math.hypot(px, py)
    y = 0.5 * math.log((e + pz) / (e - pz))
    return pt, y, math.atan2(py, px)


def make_object(p):
    p4 = four_vector(p["pt"], p["eta"], p["phi"])
    return {"name": p["name"], "tag": p["tag"], "p4": p4, "pt": p["pt"], "eta": p["eta"], "phi": p["phi"],
            "constituents": [p["name"]]}


def delta_phi(phi1, phi2):
    """phi1 - phi2 wrapped into [-pi, pi]: phi is periodic."""
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi < -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r2(obj1, obj2):
    """Delta R^2 = Delta y^2 + Delta phi^2."""
    deta = obj1["eta"] - obj2["eta"]
    dphi = delta_phi(obj1["phi"], obj2["phi"])
    return deta * deta + dphi * dphi


def pair_distance(obj1, obj2, radius):
    """Cambridge/Aachen: d_ij = DeltaR_ij^2 / R^2, no momentum anywhere."""
    return delta_r2(obj1, obj2) / (radius * radius)


def beam_distance(obj):
    """Cambridge/Aachen: d_iB = 1, i.e. a pseudojet is a jet once every other object is further than R."""
    return 1.0


def merge_objects(obj1, obj2):
    """The E scheme: add the four-vectors."""
    p4 = [a + b for a, b in zip(obj1["p4"], obj2["p4"])]
    pt, y, phi = kinematics(p4)
    return {"name": f"({obj1['name']}+{obj2['name']})", "tag": "", "p4": p4, "pt": pt, "eta": y, "phi": phi,
            "constituents": obj1["constituents"] + obj2["constituents"]}


def print_objects(objects, title):
    print("\n" + title)
    print("-" * len(title))
    for obj in objects:
        print(f"{obj['name']:16s} {obj['tag']:15s} pT = {obj['pt']:7.3f} GeV,  y = {obj['eta']:7.4f},  "
              f"phi = {obj['phi']:7.4f},  constituents = {''.join(sorted(obj['constituents']))}")


def all_distances(objects, radius):
    """Every pair and beam distance of the current step, smallest first (ties keep the input order)."""
    rows = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            rows.append((pair_distance(objects[i], objects[j], radius), "pair", i, j,
                         math.sqrt(delta_r2(objects[i], objects[j]))))
    for i in range(len(objects)):
        rows.append((beam_distance(objects[i]), "beam", i, None, None))
    return sorted(rows, key=lambda r: r[0])


def print_distance_table(objects, rows, n_show):
    print(f"\nDistances (the {n_show} smallest of {len(rows)})")
    print("---------")
    for d, kind, i, j, dr in rows[:n_show]:
        if kind == "pair":
            print(f"d({objects[i]['name']}, {objects[j]['name']}) = {d:.4e}   (dR = {dr:.3f})")
        else:
            print(f"d({objects[i]['name']}, beam) = {d:.4e}")


def run_clustering(input_particles, radius, n_show=6):
    """Cambridge/Aachen step by step: at every step the smallest of all d_ij and d_iB wins."""
    active = [make_object(p) for p in input_particles]
    final_jets, summary = [], []
    step = 1
    print_objects(active, f"Initial particles (Cambridge/Aachen, R = {radius})")
    while active:
        print("\n" + "=" * 70 + f"\nStep {step}\n" + "=" * 70)
        print_objects(active, "Current active objects")
        rows = all_distances(active, radius)
        print_distance_table(active, rows, n_show)
        d, kind, i, j, dr = rows[0]
        if kind == "pair":
            obj1, obj2 = active[i], active[j]
            merged = merge_objects(obj1, obj2)
            print(f"\nSmallest distance is the pair distance d({obj1['name']}, {obj2['name']}) = {d:.4e}: "
                  f"merge them -> pT = {merged['pt']:.3f} GeV")
            summary.append(f"step {step}: d({obj1['name']}, {obj2['name']}) = {d:.3f} (dR = {dr:.3f}) wins: "
                           f"merge -> {''.join(sorted(merged['constituents']))} with pT = {merged['pt']:.2f} GeV")
            for index in sorted([i, j], reverse=True):
                active.pop(index)
            active.append(merged)
        else:
            obj = active[i]
            print(f"\nSmallest distance is the beam distance d({obj['name']}, beam) = {d:.4e}: "
                  f"{obj['name']} is a final jet (nothing within R of it)")
            summary.append(f"step {step}: d({obj['name']}, beam) = 1 wins: jet {''.join(sorted(obj['constituents']))} "
                           f"with pT = {obj['pt']:.2f} GeV")
            final_jets.append(obj)
            active.pop(i)
        step += 1
    print("\n" + "=" * 70 + "\nSummary\n" + "=" * 70)
    for line in summary:
        print(line)
    print_objects(final_jets, "Final Cambridge/Aachen jets")
    return final_jets


def plot_event(input_particles, final_jets, radius, output=None, show=True):
    """The nine particles and the jets in the (y, phi) plane: a circle of radius R around every jet axis."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    # the core: the particle with most neighbours within 0.1 (A here), and those neighbours
    hardest = max(input_particles, key=lambda p: sum(math.sqrt(delta_r2(p, q)) < 0.1 for q in input_particles))
    core, box = [], []
    for p in input_particles:
        ax.scatter(p["eta"], p["phi"], s=20.0 + 6.0 * p["pt"], alpha=0.7)
        if math.sqrt(delta_r2(p, hardest)) < 0.1:          # the core: too close to label one by one
            core.append(p)
        else:
            ax.text(p["eta"] + 0.02, p["phi"] + 0.02, f"{p['name']} {p['pt']:.1f} GeV", fontsize=9)
    if core:
        ax.annotate("the core, within 0.07:\n" + ", ".join(f"{p['name']} {p['pt']:.1f}" for p in core) + " GeV",
                    (hardest["eta"], hardest["phi"]), xytext=(0.04, 0.08), textcoords="axes fraction", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    for idx, jet in enumerate(final_jets, start=1):
        ax.scatter(jet["eta"], jet["phi"], s=120, marker="x", linewidths=2)
        ax.add_patch(plt.Circle((jet["eta"], jet["phi"]), radius, fill=False, linestyle="--", linewidth=2))
        ax.text(jet["eta"] + 0.06, jet["phi"] - 0.14 - 0.05 * len(jet["constituents"]) ** 0.5,
                f"jet {idx}: {jet['pt']:.1f} GeV\n{''.join(sorted(jet['constituents']))}", fontsize=9)
    ax.set_xlabel("rapidity $y$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(rf"Cambridge/Aachen, $R={radius}$: nine particles of our event")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.75, -0.95)
    ax.set_ylim(-3.15, -1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output:
        fig.savefig(output)
        print(f"\nwrote {output}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-R", "--radius", type=float, default=R)
    ap.add_argument("-o", "--output", default=None, help="save the picture (pdf or png)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--no-plot", action="store_true", help="the log only")
    args = ap.parse_args()
    jets = run_clustering(particles, args.radius)
    if not args.no_plot:
        plot_event(particles, jets, args.radius, args.output, show=not args.no_show)
