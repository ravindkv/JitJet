
I think I have enough material to exapand chapter/chapter_01_ancestral_home.tex. In this chapter and in future for other chapters also, my plan is to:
+ Theory: Explain theoretical detail and enough steps to compute qqbar -> bbar xsec. Please take the material from example/ME/Standalone/standalone_qqbar_bbbar_xsec.py and also mention the caveats from example/ME/Standalone/qqbar_bbbar_xsec.py.
+  Standalone code: Next I would like to include the example/ME/Standalone/standalone_qqbar_bbbar_xsec.py code itself in the book for readers who could simply python example/ME/Standalone/standalone_qqbar_bbbar_xsec.py and then they will see the xsec value that we got from Theory stage.
+ CMSSW code: Then we also give an example from offical CMSSW code. You can see an example in example/ME/CMSSW/step1_GEN_cfg.py and corresponding README.md for setup. 

In the book, the .py files can be included using maybe the minted latex package (you can also use other packages if worth).  The idea is to include .py file from path which would help if later thoses files are updated.

the excersise section we can be based on the Theory caveats. Feel free to organise the excersize as you prefer.

The whole idea is to give the reader full confidence. Finally the journey of event is listed in example/ME/event_ME.lhe file. Please create a python script example/ME/plot_journey_ME.py file which would produce a 2d plot in eta-phi showing the location of b, bbar and pT values. The plot (in pdf format) can be included in the book.
