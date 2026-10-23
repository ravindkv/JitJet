#!/usr/bin/env python3
"""Pages per chapter from the table of contents, against the Constitution's budget.

Rule E1: a stop is 25 +- 5 pages (the target is 25, the hard cap 30), an
interlude at most 8. The script reads JitJet.toc (chapter and part starts) and
JitJet.log (the last page) after a build and prints one line per chapter, marks
a chapter above the target, and exits with status 1 when one is over the cap,
so `make pages` can be used as a check.
"""
import re
import sys
from pathlib import Path

TARGET = 25               # a stop aims at this many pages ...
TOLERANCE = 5             # ... and may exceed it by this much
BUDGET = TARGET + TOLERANCE
INTERLUDE_BUDGET = 8
INTERLUDES = {3}          # chapters that are interludes (no standalone script)

ROOT = Path(__file__).resolve().parent.parent
TOC = ROOT / "JitJet.toc"
LOG = ROOT / "JitJet.log"


def roman_or_int(s):
    try:
        return int(s)
    except ValueError:
        return None  # front matter pages (roman) are ignored


def main():
    if not TOC.exists():
        sys.exit(f"{TOC.name} not found: build the book first")
    starts = []  # (page, kind, number, title)
    pat = re.compile(r"\\contentsline \{(chapter|part)\}\{(?:\\numberline \{([^}]*)\})?(.*?)\}\{([^}]*)\}\{[^}]*\}%?$")
    for line in TOC.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        kind, number, title, page = m.groups()
        page = roman_or_int(page)
        if page is None:
            continue
        starts.append((page, kind, number, title))
    # the last arabic page: the largest "[n" page marker pdflatex wrote to the log
    total = None
    if LOG.exists():
        pages = [int(n) for n in re.findall(r"\[(\d+)[\]\s{]", LOG.read_text(errors="ignore"))]
        if pages:
            total = max(pages) + 1   # the page after the last one
    bad = 0
    print(f"{'chapter':>8}  {'start':>5}  {'pages':>5}  {'cap':>3}  title")
    for i, (page, kind, number, title) in enumerate(starts):
        if kind != "chapter" or not number or not number.isdigit():
            continue
        nxt = starts[i + 1][0] if i + 1 < len(starts) else total
        pages = (nxt - page) if nxt else None
        interlude = int(number) in INTERLUDES
        cap = INTERLUDE_BUDGET if interlude else BUDGET
        flag = ""
        if pages is not None and pages > cap:
            flag = "  <-- over the cap"
            bad += 1
        elif pages is not None and not interlude and pages > TARGET:
            flag = f"  (above the target of {TARGET})"
        title = re.sub(r"\\[a-zA-Z]+|[{}$]", "", title).strip()
        print(f"{number:>8}  {page:>5}  {pages if pages is not None else '?':>5}  {cap:>3}  {title}{flag}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
