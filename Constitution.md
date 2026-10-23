This is the constitution of the JitJet book, which every chapter has to follow.
The author is a tour guide and the readers are passengers; the family trip is
the metaphor; the guide keeps the passengers engaged. The constitution is
documented, with polishing, in chapter/chapter_00_frontmatter.tex, and it is
the only place that documents it.

Rule A: Every chapter (a "stop") has three homes
    (A1) example/XYZ/            the scripts, logs, records, plots and films
    (A2) chapter/chapter_XYZ.tex the book chapter: physics, in the guide's voice
    (A3) chapter/chapter_XYZ.md  the technical companion, not compiled into
                                 the book, reached from the .tex by exactly
                                 one \url to its GitHub page
    (A4) chapter/chapter_00_frontmatter.tex holds the JitJet World table
     (label tab:jitjet-world): repository, standalone scripts, chapter
     companions, CMSSW material, YouTube channel and playlist. Chapters
     refer to it by \ref through the \jitjetworld macro and never repeat
     its URLs. The only URL a chapter prints itself is the one to its
     own .md (A3).

Rule B: The .tex chapter opens with
    (B0) the title
    (B1) the stage in the journey (the metaphor, one paragraph)
    (B2) the physics in this chapter (one paragraph, and the one number to
         take away from the stop)

Rule C: The .tex chapter then has exactly these sections
    (C1) What we will see at this stop
        (C1-1) motivation, connected to the previous chapter
        (C1-2) the animated journey: the YouTube playlist link, ONE paragraph,
               and at most one figure with relevant frames. The films are
               fully narrated, so the .tex does not retell them.
        (C1-3) a sketch or diagram, and a picture to carry if it helps
        (C1-4) the agreement table: one row per quantity, columns standalone /
               official package / CMSSW, explained in one or two paragraphs at
               most. How the numbers are produced is in the .md, not here.
    (C2) Let us dive into the details: the theory
        Every subsection follows the same order:
        (C2-1) the motivation
        (C2-2) a sketch or Feynman diagram in TikZ
        (C2-3) the equation
        (C2-4) the numbers the equation gives for our event, from the -v log of
               the standalone script, quoted inline in the text, never as a
               log listing
        (C2-5) a plot of the equation over the phase space, from the standalone
               script or a script next to it; one figure per equation
        The theory ends with the recipe, step by step, that the script runs.
    (C3) The energy ledger: our jet at this stage
        (C3-1) the ledger table and the eta-phi display side by side,
               standalone and CMSSW, cumulative since chapter 1; details in
               the .md
        (C3-2) what this stop is, and what it is not
    (C4) Further reading
        (C4-1) pitfalls and FAQs for newcomers
        (C4-2) exercises
        (C4-3) open problems for the Phase-2 LHC, after a literature survey,
               with citations

Rule D: The .md companion has exactly these sections
    (D1) Running the standalone script: command, options, what -v/-vv print
    (D2) Reading the record: the LHE/HepMC rows, with the excerpts the .tex
         quotes numbers from
    (D3) Drawing the event and rendering the film: plot and animate commands
    (D4) Validation from official packages: the PYTHIA/MadGraph/FastJet script,
         the CMSSW configuration and log, and how each number was read out
    (D5) The full tables: every number the .tex quotes, with the command that
         produced it

Rule E: Length
    (E1) a stop is at most 25 +-5 pages; an interlude (a stop with no standalone
         script, e.g. matching and merging) is at most 8 and follows only
         C1, C3-2 and C4
    (E2) no minted block in the .tex except TikZ figures; commands, logs and
         configurations live in the .md
