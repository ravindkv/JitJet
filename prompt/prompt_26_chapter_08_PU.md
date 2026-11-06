IDEA: Since we are writing the book in 2026 where the Run-3 data-taking is already ended. So the book will have more relevance for Phase2 HL-LHC. I am thinking of writing the next parts (Part-II,  Part-III, etc) by passing our event (which is for the Run-3 conditions, i think except the slight change in the energy everything else will remain the same) to the Phase2 Geometry. In summary:

Original plan: Run3 event in Run3 Geometry
Updated but a bit conservative/to be discussed plan: Run3 event in Phase2 Geometry
More accurate but then we will have to  rewrite Part-1: Phase2 event in Phase2 Geometry.

What is your take on this IDEA?

Run-3 event, Phase-2

Yes, start chapter 8

A MAJOR update on Run-3 vs Phase-2 road: I discussed with Mikko and Theo about which road we should take from their point of view. They both suggested staying on the Run-3 road. The argument was the fact that the Phase-2 is still in an evolving situation so by next year it might change so the book will  not reflect the updated settings. Since  the Run-3 ended this year, it is kind of frozen and a book on  Run-3 settings would reflect more accurately. We anyway mention the open problems in the context of phase-2 HL-LHC for every chapter, we can even be more thorough focusing more details on CMS. And when Phase-2 starts/ends, we can have a  new  book  if needed.

In  view of this, please write chapter/chapter_08_PU* and example/PU/* for Run-3 (2024). I already have some  inputs from CMSSW in the example/PU/CMSSW/README.md file. For now let us NOT create  new .mp4 just still should be fine to get plots to include in the tex.


thanks. Please implement the following:
(a)
Below the figure \label{fig:crowd_on_the_road:minbias_sketch}, show a table that lists 10 minimum bias events with columns indicating:
event	proc.	interactions	stable	charged	central	|eta| > 5.5	lead pT [GeV]	sum pT [GeV]	jets > 10 GeV	PPS protons	z [cm]	t [ns]

similar to what we have from CMSSW here:
https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.md#6-the-250-events-at-generator-level

include tables similar to the next two shown in the above link

(b)
of figure \label{fig:crowd_on_the_road:region_sketch}, separate 3rd figurer and list number of events each bunch crossing would produce and the total events which will pileup on our signal events. Then in the text explain that inside CMSSW we can check which event comes from signal and which from other bx at least in MC.

(c) Create also the animated video for 2024. The video should also explain the ND, SD, DD, CD collision with the table. These should not be included as introductory frames in the begining of the chapter as we will already have them in  later section

Feel free to tweak (a), (b),  (c) a bit if needed :-)
