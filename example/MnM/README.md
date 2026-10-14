# Chapter 3: Whose Child Is It? Matching and merging

Chapter 3 is a stop on the road at which our event does not get off: a
`HardQCD:qqbar2bbbar` sample is one leading-order 2->2 process, with no
extra-parton subprocess and no NLO matrix element, so every parton beyond the
b and bbar has exactly one parent, the shower, and there is nothing to match
or merge. The ledger row repeats the previous one (100.45 GeV, Delta pT = 0).
There is no standalone script here; the chapter's sketches are
`chapter/tikz/mm_*.tex`.

`animate_whose_child.py` is the ten-second film of the stop for the JitJet
playlist: the road of the book with its stops, the cart carrying our event,
the sign that says who stop 3 is for (MLM, CKKW-L, MC@NLO, POWHEG, FxFx),
the samples that do get off here, the card "not for us" with the unchanged
ledger row, and the cart rolling on to stop 4.

```bash
python3 animate_whose_child.py                                   # -> animate_whose_child.mp4 (1280x720, 24 fps, 10 s)
python3 animate_whose_child.py --fast                            # low-resolution preview
python3 animate_whose_child.py --stills animate_whose_child_stills.pdf --no-video   # one frame per beat (make figures)
```

On this Mac use `/usr/bin/python3` (matplotlib); the film takes about half a
minute to render and needs `ffmpeg`. The `.mp4` is not committed, the stills
PDF is (chapter 3 includes its three pages as a figure).
