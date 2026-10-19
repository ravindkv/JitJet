# Chapter 3: Whose Child Is It? Matching and merging

Chapter 3 is a stop on the road at which our event does not get off: a
`HardQCD:qqbar2bbbar` sample is one leading-order 2->2 process, with no
extra-parton subprocess and no NLO matrix element, so every parton beyond the
b and bbar has exactly one parent, the shower, and there is nothing to match
or merge. The ledger row repeats the previous one (100.45 GeV, Delta pT = 0).
There is no standalone script here; the chapter's sketches are
`chapter/tikz/mm_*.tex`.

`animate_whose_child.py` is the narrated film of the stop for the JitJet
playlist: the road of the book with its stops, the cart carrying our event,
the sign that says who stop 3 is for (MLM, CKKW-L, MC@NLO, POWHEG, FxFx),
the samples that do get off here, the card "not for us" with the unchanged
ledger row, and the cart rolling on to stop 4. The script clock is 12.5 s in
three beats; each phase of a beat lasts at least its line of narration (the
voice and the score come from `../soundtrack.py`, shared by all the films),
so the narrated film is about 110 s and the silent one 12.5 s.

```bash
python3 animate_whose_child.py                                   # -> animate_whose_child.mp4 + .srt (1920x1080, 24 fps, ~110 s)
python3 animate_whose_child.py --fast                            # low-resolution preview
python3 animate_whose_child.py --print-script                    # the narration, to record it yourself
python3 animate_whose_child.py --no-narration                    # music only, the 12.5 s script clock
python3 animate_whose_child.py --stills animate_whose_child_stills.pdf --no-video   # one frame per beat (make figures)
```

On this Mac use `/usr/bin/python3` (matplotlib); the film takes about ten
minutes to render and needs `ffmpeg`, plus `edge-tts` for the voice (it
fetches Microsoft's neural voices over the network; the clips are cached in
`build/narration/MnM/`). The `.mp4` and `.srt` are not committed, the stills
PDF is (chapter 3 includes its three pages as a figure).
