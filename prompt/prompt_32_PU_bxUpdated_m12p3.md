
In the example/PU/CMSSW/README.md, we have 
https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.tex

which I have updated to  include bx = -12 to +3 (earlier it used  -1 to +1) so the corresponding txt in the chapter/chapter_08_PU.tex has to  be updated. Also I would like to have a table from Standalone on “The ledger of the overlay” similar to what we have from CMSSW:
https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.tex#L214-L260



Thanks. I just updated the
https://github.com/ravindkv/JitJetSW/blob/main/Journey/doc/chapter/chapter_08_PU.tex#L215-L266
where I added corresponding columns for out-time pileup and dropped the E (Energy) column. The signal + pileup column is renamed as signal + in-time pileup. Please implement this in our new table. Now, In our book, we can either in caption or text mention that the out-time pileup is removed at the later stage from ECAL/HCAL so we do not overlap for now. Maybe we can mention that the intime-pileup is also removed later at the RECO stage of CHS/PUPPI. The idea is to give the readers clear msg that at this stage,  in MC, we have separate signal, in-time, out-time pileup. The separation exist at the next chapters: G4, SIM, DIGI. Then the DIGI are mixed and separation is lost. Then at RECO stage, we first remove the out-time and later in-time pileup.

