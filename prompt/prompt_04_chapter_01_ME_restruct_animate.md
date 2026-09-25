Let us implement the following in chapter/chapter_01_ancestral_home.tex:

(1) the 30 GeV pT cut leads the both b quarks towards the endcap. I wanted to have at least one b-jet closer to the barrel so I increased minPt cut to 100. I also ran Madgraph following the instructions mentioned in example/ME/MadGraph/qqbar_bbbar_xsec_MadGraph.md. Please include another row in the first table using the xsec from this file. Then the madgraph section in the book can be expanded a bit.

(2) The new xsec for 100 GeV cut:
CMSSW: step1_GEN_cfg.py: 3.451e+02 +- 6.078e+00 pb
I have updated the event_ME.lhe file and  also the journey_ME.pdf
The standalone script is giving also similar result:
python standalone_qqbar_bbbar_xsec.py gives 345.320 pb
.venv/bin/python qqbar_bbbar_xsec.py --output result.json gives 345.324 +/- 0.001 pb
xsec from MadGraph in 344.3 +- 0.9587 pb 
Please update the full chaper_01 to reflect 100 GeV min pT scenario.

(3)I would  like to be the “tour guide” and readers as passengers who are going to witness  the entire journey of the jet. The tour guide will explain every step making it interesting and enjoyable for the readers. So please rewrite the first chapter (and in  future all  chapters we will write too) in this theme. As an example, I am a big fan of Griffiths book on electrodynamics where the reader feels a teacher besides him/her while reading. Taking inspiration from that,  the passenger (reader) should feel a tour guide at every  moment of this book.

(4) Since the JitJet package is hosted on github(https://github.com/ravindkv/JitJet), let us NOT include python code  inside the book,  instead just write the instructions on how to run them. For more details about those files could be found on github. This way the number of pages on book could be minimised. That would allow us to expand the mathematics formulae a bit so that  the reader is not lost at every step, rather they should feel the equations. 

(5) While reading the current content, I needed to understand a few explanations which I have put in example/ME/Standalone/explanation_born_weights.md and example/ME/Standalone/explanation_jacobian_better_variables.md files. In chapter 01, we can expand the corresponding equations for ease. We do not need to include everything from these two files. Just enough content to help readers. Then we can put these .md files in .gitignore. Maybe we can also explain in more details in chapter/appendix_A_kinematics.tex



(6) In the “Further reading” section, please include a few points/projects under the subsection titled “Open problems in the context of phase-2 LHC”. The idea is to outline what needs to be updated or tuned for Run-4. The idea is also to let students think about these problems and if they will be willing to explore them in future and provide more feedback to the developers of generators. Probably in later chapters this will be more relevant and important.

(7) Now we also have the script to animate the journey:
example/ME/Standalone/animate_standalone_qqbar_bbbar_xsec.py
Please include this also in the book. I am  uploading the .mp4 video on youtube which can be found in the following playlist:
https://www.youtube.com/playlist?list=PLFtvDH9g5Km0
In the book we can refer to this playlist. You can describe briefly what is shown in the video inside the book.

Please implement all these 7 points in the book.

