#!/usr/bin/env python3
"""The film of chapter 10: the hits appear in the order in which the detector receives them.

Four scenes, narrated, all in the r-z view of the detector of standalone_SIM.py:
  1 the family crosses        the hits of the signal appear as the clock runs from 0 to 30 ns after the collision
  2 the crowd of our crossing the 63 in-time collisions, simulated separately, on their own
  3 the other crossings       the 897 out-of-time collisions, each on the clock of its own bunch crossing
  4 added by hand             one clock for all: from -300 ns to +100 ns, the sixteen crossings arrive one after the
                              other and the family is one flash among them; then the sum, and the circle of our jet

Every number of the narration is read from add_hits_SIM.json (written by add_hits_SIM.py).

Narration and music: as in the other chapter films (example/soundtrack.py): one line per phase (--print-script),
edge-tts or the macOS `say` voice, clips cached in build/narration/SIM/, an .srt next to the .mp4.

  python3 animate_standalone_SIM.py --print-script
  python3 animate_standalone_SIM.py --stills animate_standalone_SIM_stills.pdf --no-video     # the four key frames
  python3 animate_standalone_SIM.py --fast -o /tmp/preview.mp4                               # a quick look
  python3 animate_standalone_SIM.py --jobs 8                                                 # the film, 1920x1080

In the film the out-of-time hits are drawn one in four (--thin); the counters count all of them.
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
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
import add_hits_SIM as add  # noqa: E402
import plot_simhits_SIM as disp  # noqa: E402
import standalone_SIM as sim  # noqa: E402

sys.path.insert(0, str(HERE.parent.parent))
import soundtrack as snd  # noqa: E402

COL = {0: "black", 1: "#999900", 2: "#6699ff", 3: "#ff6666"}
LABEL = {0: "our family", 1: "earlier crossings", 2: "our crossing, the crowd", 3: "later crossings"}
TEXT, TEXT2 = "#0b0b0b", "#52514e"
FIGSIZE = (12.8, 7.2)
SELECT = {"signal": [0], "intime": [2], "oot": [1, 3], "mixed": [1, 3, 2, 0]}
# the key frames of the chapter (--stills): title, sample, the clock [ns]
SCENES = [("1. the family crosses the border", "signal", 0.0, 30.0, 60),
          ("2. the crowd of our own crossing, simulated on its own", "intime", 0.0, 30.0, 40),
          ("3. the fifteen other crossings, each on its own clock", "oot", 0.0, 30.0, 40),
          ("4. added by hand: sixteen crossings on one clock", "mixed", -300.0, 100.0, 160)]


def data(S, thin: int = 1):
    """The hits and cells of the three samples on both clocks. With thin > 1 only every thin-th out-of-time hit is
    kept for drawing; the sorted times (`count`) always hold all of them."""
    M = add.mix(S)
    t, o, bx = M["trk"]["t"], M["trk"]["origin"], M["trk"]["bx"]
    tl = t - 25.0 * bx
    z, r = M["trk"]["z"], np.hypot(M["trk"]["x"], M["trk"]["y"])
    count = {(k, c): np.sort(v[o == k]) for k in range(4) for c, v in (("t", t), ("tl", tl))}
    keep = np.ones(t.size, dtype=bool)
    if thin > 1:
        oot = np.flatnonzero((o == 1) | (o == 3))
        keep[oot] = False
        keep[oot[::thin]] = True
    _, cx, cy, cz = sim.cell_position(M["cells"])
    ecal = np.isin(M["det"], (sim.EB, sim.EE, sim.ES))
    return dict(t=t[keep], tl=tl[keep], o=o[keep], z=z[keep], r=r[keep], count=count, cz=cz, cr=np.hypot(cx, cy), E=M["E"], Ebx=M["Ebx"],
                det=M["det"], ecal=ecal, ecal_bx=np.concatenate([[0.0], M["Ebx"][ecal].sum(axis=0).cumsum()]), ecal_o=M["E"][ecal].sum(axis=0))


def crossings_done(tnow: float) -> int:
    """How many of the sixteen crossings (bx -12 ... +3) have put their energy into the cells at the common time tnow."""
    return int(np.clip(np.floor((tnow - 5.0) / 25.0) + 13, 0, 16))


def shower_fraction(tnow: float) -> float:
    """The cells light up between 4 and 16 ns after their own collision."""
    return float(np.clip((tnow - 4.0) / 12.0, 0, 1))


def hits_so_far(D, sample: str, tnow: float) -> Dict[int, int]:
    c = "t" if sample == "mixed" else "tl"
    return {k: int(np.searchsorted(D["count"][(k, c)], tnow, side="right")) for k in SELECT[sample]}


def ecal_so_far(D, sample: str, tnow: float) -> float:
    if sample == "mixed":
        return float(D["ecal_bx"][crossings_done(tnow)])
    return float(sum(D["ecal_o"][k] for k in SELECT[sample]) * shower_fraction(tnow))


def frame(ax, D, scene, tnow, with_title: bool = True):
    title, sample = scene[0], scene[1]
    ax.clear()
    disp.draw_detector(ax, "rz", "ALL")
    sel = SELECT[sample]
    tt = D["t"] if sample == "mixed" else D["tl"]
    for k in sel:
        m = (D["o"] == k) & (tt <= tnow)
        if m.any():
            ax.plot(D["z"][m], D["r"][m], ls="none", marker=".", ms=2.5 if k == 0 else 1.2, mew=0, color=COL[k], rasterized=True, zorder=3 + (k == 0))
    # the cells light up when the crossing of their energy has passed (the shower needs 4 to 40 ns to get there)
    if sample == "mixed":
        e = D["Ebx"][:, :crossings_done(tnow)].sum(axis=1)
        col = "#999900"
    else:
        e = sum(D["E"][:, k] for k in sel) * shower_fraction(tnow)
        col = COL[sel[0]]
    m = e > 0.05
    if m.any():
        w = np.where(D["det"][m] == sim.HF, 0.3, np.where(np.isin(D["det"][m], (sim.HB, sim.HE)), 150.0, 1.0))
        ax.scatter(D["cz"][m], D["cr"][m], s=np.clip(2 + 4 * np.log10(1 + e[m] * w), 1, 40), color=col, alpha=0.5, lw=0, rasterized=True, zorder=2)
    ax.set_xlim(-1300, 1300); ax.set_ylim(0, 480); ax.set_xlabel("z [cm]"); ax.set_ylabel("r [cm]")
    if with_title:
        n = sum(hits_so_far(D, sample, tnow).values())
        ax.set_title(f"{title}\n$t$ = {tnow:+.0f} ns: {n:,d} tracker and muon hits so far".replace(",", " "), fontsize=11, loc="left")


# --- the film: scenes on the clock of soundtrack.py --------------------------------------------------------------
SCENE_NAMES = {1: "family", 2: "intime", 3: "oot", 4: "mixed"}


class Scene:
    """A scene is a list of segments (phase key, clock from, clock to [ns], seconds at the nominal speed)."""

    def __init__(self, number, title, sample, segments, caption, D, N):
        self.number, self.title, self.sample, self.caption, self.D, self.N = number, title, sample, caption, D, N
        self.seg, k = [], 0
        for key, ta, tb, seconds in segments:
            n = int(round(seconds * snd.NOMINAL_FPS))
            self.seg.append((key, k, k + n, ta, tb))
            k += n
        self.nframes = k

    def phases(self) -> List[Tuple[str, int, int]]:
        return [(key, k0, k1) for key, k0, k1, _, _ in self.seg]

    def at(self, k: float) -> Tuple[str, float]:
        """(phase key, the clock [ns]) at the frame index k (a float)."""
        for key, k0, k1, ta, tb in self.seg:
            if k < k1 or (key, k0, k1, ta, tb) == self.seg[-1]:
                u = min(max((k - k0) / max(k1 - 1 - k0, 1), 0.0), 1.0)
                if ta == tb or (u >= 1.0 and self.sample != "mixed" and k1 == self.nframes):
                    return key, (ta if ta == tb else math.inf)       # the last frame of a sample: every hit, also the late spirals
                return key, ta + (tb - ta) * u
        raise RuntimeError("no segments")

    def draw(self, fig, k: float):
        key, tnow = self.at(k)
        D, N = self.D, self.N
        fig.clf()
        fig.patch.set_facecolor("white")
        fig.text(0.02, 0.965, f"Scene {self.number}: {self.title}", fontsize=15, weight="bold", color=TEXT, va="center")
        fig.text(0.98, 0.965, "standalone_SIM.py + add_hits_SIM.py", fontsize=9.5, color=TEXT2, ha="right", va="center")
        fig.text(0.02, 0.03, self.caption, fontsize=10.5, color=TEXT, va="bottom")
        ax = fig.add_axes([0.065, 0.165, 0.665, 0.745])
        frame(ax, D, (self.title, self.sample), tnow, with_title=False)
        # the panel on the right: the clock, the hits and the energy so far
        x = 0.765
        fig.text(x, 0.875, "the clock" if self.sample == "mixed" else "the clock of each collision", fontsize=10, color=TEXT2, va="center")
        t_end = max(tb for *_, tb in self.seg if math.isfinite(tb))
        clock = f"t = {round(tnow) + 0:+d} ns" if math.isfinite(tnow) else f"t > {t_end:+.0f} ns"
        fig.text(x, 0.825, clock, fontsize=22, color=TEXT, va="center", weight="bold")
        if not math.isfinite(tnow):
            fig.text(x, 0.782, "all sixteen crossings, every hit" if self.sample == "mixed" else "every hit, also the late spirals", fontsize=11, color=TEXT2, va="center")
        elif self.sample == "mixed":
            b = int(np.clip(np.floor((tnow + 12.5) / 25.0), -12, 3))
            fig.text(x, 0.782, f"crossing {b:+d}" if b else "crossing 0: ours", fontsize=11, color=TEXT2, va="center")
        y = 0.71
        fig.text(x, y, "tracker and muon hits so far", fontsize=10, color=TEXT2, va="center")
        hits = hits_so_far(D, self.sample, tnow)
        for o in sorted(hits, key=lambda j: (1, 2, 3, 0).index(j)):
            y -= 0.052
            fig.text(x, y, f"{hits[o]:,}", fontsize=16, color=COL[o], va="center", weight="bold")
            fig.text(x + 0.105, y, LABEL[o], fontsize=9.5, color=TEXT2, va="center")
        if len(hits) > 1:
            y -= 0.052
            fig.text(x, y, f"{sum(hits.values()):,}", fontsize=16, color=TEXT, va="center", weight="bold")
            fig.text(x + 0.105, y, "sum", fontsize=9.5, color=TEXT2, va="center")
        y -= 0.075
        fig.text(x, y, "energy in the ECAL so far", fontsize=10, color=TEXT2, va="center")
        fig.text(x, y - 0.05, f"{ecal_so_far(D, self.sample, tnow):,.0f} GeV", fontsize=16, color=TEXT, va="center", weight="bold")
        if key in ("sum", "jet"):
            tot = N["tracker"]["total"]
            fig.text(x, y - 0.13, f"the family owns {100 * tot['signal'] / sum(tot.values()):.2f} %\nof the tracker hits", fontsize=11.5, color=TEXT, va="center")
        if key == "jet":
            eta = N["jet"]["eta"]
            rr = np.array([0.0, 300.0])
            for de, ls in ((-0.4, ":"), (0.0, "-"), (0.4, ":")):
                ax.plot(rr * math.sinh(eta + de), rr, color="#d62728", lw=1.6 if de == 0 else 1.0, ls=ls, zorder=6)
            c = {d["origin"]: d["scalar"] for d in N["circle_ledger"]}
            crowd = c["intime"] + c["early"] + c["late"]
            ax.text(-1250, 455, f"our jet, $\\eta$ = {eta:.2f}: the cells within $\\Delta R$ < 0.4 hold\n"
                                f"{c['signal']:.0f} GeV of $E_T$ from the family, {crowd:.0f} GeV from the crowd",
                    fontsize=11, color="#d62728", va="top", bbox=dict(facecolor="white", edgecolor="#d62728", alpha=0.92, boxstyle="round,pad=0.4"), zorder=7)


def build_scenes(D, N) -> List[Scene]:
    dots = "Dots: hits in the tracker and the muon chambers. Circles: cells of the calorimeters, their size the logarithm of the energy."
    return [
        Scene(1, "the family crosses the border", "signal", [("tracker", 0.0, 10.0, 6.0), ("calo", 10.0, 18.0, 5.0), ("count", 18.0, 30.0, 4.0)], dots, D, N),
        Scene(2, "the crowd of our own crossing, simulated on its own", "intime", [("crowd", 0.0, 14.0, 6.0), ("count", 14.0, 30.0, 4.0)], dots, D, N),
        Scene(3, "the fifteen other crossings, each on its own clock", "oot", [("clocks", 0.0, 14.0, 6.0), ("count", 14.0, 30.0, 4.0)],
              dots + " Olive: earlier crossings, red: later ones.", D, N),
        Scene(4, "added by hand: sixteen crossings on one clock", "mixed",
              [("early", -300.0, -12.5, 8.0), ("own", -12.5, 12.5, 5.0), ("late", 12.5, 100.0, 4.0), ("sum", math.inf, math.inf, 2.0), ("jet", math.inf, math.inf, 2.0)],
              "Every hit keeps its place and its energy; only its time is shifted by 25 ns x the number of its crossing. Black: our family.", D, N),
    ]


def narration(N: dict) -> Dict[str, str]:
    c, p, tr = N["collisions"], N["primaries"], N["tracker"]["total"]
    n = N["counters"]["signal"]
    ec, hc, hf = N["ECAL"], N["HBHEHO"]["e"], N["HF"]["e"]
    oot = c["early"] + c["late"]
    total = sum(tr.values())
    circ = {d["origin"]: d["scalar"] for d in N["circle_ledger"]}
    crowd = circ["intime"] + circ["early"] + circ["late"]
    return {
        "family_tracker": (f"The gate has opened. ... {p['signal']} members of our family step into the detector. This is not Geant4 itself, but a small simulation written by hand, with the same physics in simple formulas. "
                           "A field of three point eight tesla bends every charged track into a helix, and every layer of silicon a particle crosses writes down a hit: a place, a time, and a little lost energy. "
                           f"By the time the family is across, the tracker will hold {tr['signal']:,} of them."),
        "family_calo": (f"Not everybody arrives as they left. {n['decays']} members decay in flight. {n['conversions']} photons convert into electron pairs in the material, electrons radiate {n['brems']} photons, "
                        f"and {n['nuclear']} hadrons break up on a nucleus. Those who reach the calorimeters shower there: {ec['e']['signal']:.0f} GeV in the crystals of the electromagnetic calorimeter."),
        "family_count": (f"The hadron calorimeter writes down far less, {hc['signal']:.1f} GeV, because it only samples: what counts is the light of its thin scintillator tiles, about one percent of the shower. "
                         f"And far forward, the quartz fibres count {hf['signal']:,.0f} photoelectrons. Thirty nanoseconds after the collision, our family is across; only a few slow spirals are still turning in the field."),
        "intime_crowd": (f"Now the same border, crossed by somebody else. The {c['intime']} other collisions of our own bunch crossing are simulated on their own, with the same rules and their own random numbers: "
                         f"{p['intime']:,} primaries. Nothing here knows about our family."),
        "intime_count": (f"They leave {tr['intime']:,} hits in the tracker, {tr['intime'] / tr['signal']:.0f} times ours, and {ec['e']['intime']:,.0f} GeV in the crystals, "
                         "most of it in the endcaps, where the crowd is all energy."),
        "oot_clocks": (f"And then the crossings before and after ours. {oot} collisions, in twelve earlier and three later bunch crossings, {p['early'] + p['late']:,} primaries. "
                       "Each collision is simulated on its own clock, as if it had happened at time zero."),
        "oot_count": (f"{tr['early'] + tr['late']:,} hits. Nothing in this sample says that it is early or late. That is decided when the samples are added."),
        "mixed_early": ("Now the addition, and it is done by hand. Every hit keeps its place and its energy; only its time is shifted, by twenty five nanoseconds for every crossing of distance. One clock for all. ... "
                        f"Three hundred nanoseconds before our collision the first crossing arrives, and then one more every twenty five nanoseconds: {c['early']} collisions before ours."),
        "mixed_own": (f"Time zero. Our family crosses, together with the {c['intime']} collisions of its own crossing. Watch the black dots: they are one flash among sixteen."),
        "mixed_late": (f"Three more crossings follow, {c['late']} collisions, whose particles are still on their way while the detector is reading us out."),
        "mixed_sum": (f"The sum: {total:,} hits in the tracker, and our family owns {100 * tr['signal'] / total:.2f} percent of them. "
                      f"In the crystals, {ec['hit_all']:,} cells hold energy, and in only {ec['dom']['signal']:,} of them is the family the largest contributor."),
        "mixed_jet": (f"And our jet? It points at the corner between the barrel and the endcap. In a circle of radius zero point four around it, the cells hold {circ['signal']:.0f} GeV of transverse energy from the family, "
                      f"and {crowd:.0f} from the crowd: {circ['intime']:.0f} from our own crossing, the rest from the others. The jet is still there. It is no longer alone."),
        "outro": ("Nothing has been measured yet. A hit is deposited energy, not a signal. ... Turning sixteen crossings of hits into the pulses of the electronics is the next film."),
    }


def score_spec(film: snd.Film) -> snd.ScoreSpec:
    return snd.film_score(film, [snd.Dm, snd.Bb, snd.F, snd.Gm])


_LAST: Dict[int, Tuple[int, float]] = {}


def draw(fig, sc, k, t):
    """One frame; a frame that repeats the one before (a hold under the narration) is not drawn again."""
    key = (sc.number, round(float(k), 3))
    if _LAST.get(id(fig)) != key:
        sc.draw(fig, k)
        _LAST[id(fig)] = key


def write_stills(D, out: Path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.2))
    for ax, sc, tk in zip(axs.flat, SCENES, (12.0, 30.0, 30.0, 12.0)):
        frame(ax, D, sc, tk)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"{out} written")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=HERE / "animate_standalone_SIM.mp4")
    p.add_argument("--scene", default="all", help="all, or numbers like 1,4")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--fast", action="store_true", help="a preview: every fourth frame at 60 dpi")
    p.add_argument("--thin", type=int, default=4, help="draw one out-of-time hit in this many (film only)")
    p.add_argument("--stills", type=Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, default=REPO / "build" / "narration" / "SIM")
    snd.add_narration_args(p)
    args = p.parse_args()
    if args.fast:
        args.step, args.dpi = max(args.step, 4), min(args.dpi, 60)
    snd.narration_defaults(args)
    N = json.loads((HERE / "add_hits_SIM.json").read_text())
    lines = narration(N)
    if args.print_script:
        snd.print_script(lines)
        return
    S = add.load(HERE.parent)
    if args.stills is not None:
        write_stills(data(S), args.stills)
    if args.no_video:
        return
    scenes = build_scenes(data(S, args.thin), N)
    if args.scene != "all":
        want = {int(s) for s in args.scene.split(",")}
        scenes = [sc for sc in scenes if sc.number in want]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and not args.no_narration:
        sys.exit("ffmpeg not found on PATH (needed for the narration clips)")
    clips = snd.load_narration(lines, args, args.cache_dir, ffmpeg) if ffmpeg else {}
    film = snd.Film(scenes, snd.clip_lengths(clips), SCENE_NAMES, slow_max=4.0)
    print(f"film: {film.describe()}")
    score = None if args.no_music else score_spec(film)
    audio = None
    tmp = tempfile.TemporaryDirectory(prefix="jitjet_sim_")
    if ffmpeg and (clips or score):
        audio = Path(tmp.name) / "soundtrack.wav"
        snd.mix_audio(film.duration, film.voice_at, clips, score, args.music_db, audio)
        if clips:
            snd.write_srt(film.voice_at, clips, lines, args.output.with_suffix(".srt"))
    snd.render_video(film, args.output, args.fps / args.step, args.dpi, audio, draw, FIGSIZE, "b bbar event, the crossing of the border: the simulated hits", jobs=args.jobs)
    print(f"wrote {args.output} ({film.duration:.1f} s)")


if __name__ == "__main__":
    main()
