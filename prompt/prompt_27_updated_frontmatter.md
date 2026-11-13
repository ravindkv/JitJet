From Director’s Desk:

Journey is the Jet: JitJet. The book explicitly covers the journey of quarks which form a jet in the end. However, the idea of writing a book has its own journey. It emanated years ago when I started my PhD career in 2013 at IIT Kanpur in particle physics phenomenology. We had to generate Monte Carlo samples using available software packages which we downloaded and ran as black box with little or no understanding what happens inside. I wanted to learn and understand that, but did not have enough time or resources.

Later in 2016, I switched to experimental particle physics working in the CMS experiment at TIFR Mumbai. I was AlCa-FTV (2016-18) and AlCa-TSG(2018-20) convener where we submitted and managed more than 200 hundred workflows which pass multiple stages of the CMS chain. We, as a convener, used to look only at the high level plots in the end and decide if the given condition update in CMS changed things as expected. There also, we wanted to learn/understand each stage of the complex CMS workflow but did not have enough time and resources. 

Then in 2022 while I was a post doc at Florida Tech based at Fermilab, I became an EGM-HLT convener, where we used to check the efficiencies of HLT paths on a regular basis whenever there was change in the detector conditions. The HLT reconstruction itself comprises so many steps that understanding and learning each stage was not feasible due to lack of time and resources given the fact that the data-taking was going on.

In 2024, I joined the University of Helsinki as post doc where I lead the JME–JERC group as convener. In JERC, we derive and apply Jet energy resolutions and corrections. However, the derivation of corrections happens mostly when we already have a jet. How we got the jet in the first place was not so well known to me. Since a jet is the entire journey and the journey comprises more than 15 stages where we either add energy or remove it or do nothing. Again I wanted to understand the full journey of the jet but did not have enough time and resources for that.

The lack of time and resources was fundamentally correlated with the complexity of the CMS experiment. To understand any stage, we needed to have the technical capability to dig down to the CMS software or MC tools such as Pythia, Geant4. So it required time, technical expertise, and most importantly support from the institute.  In 2026 all these three requirement met: my JERC convenership ended in August 2026 so I had time, we then had great progress in agentic AI (e..g Claude Fable) which had technical expertise, and finally I got full support from Mikko (who is my supervisor). Mikko wanted to write a JERC related book as part of the ERC grant. However,  the JERC Run2 CHS paper, JERC Run2 + Run3 Puppi paper was already work in progress which was supposed to come in late 2027. So a traditional book on JERC would largely  overlap with papers. In that sense the JitJet book would provide a broader understanding of the jet and its corrections that happen at different stages.

The name “JitJet” of the book came to my mind when we were on the summer HIP trip to Tammela region on June 2, 2026. We had many stops with a tour guide explaining to us different parts of the trip. I told Mikael (sitting next to me) and later to Mikko and Fikri that the quarks also have to go through so many stops and the entire journey becomes a jet. Unlike other objects in the CMS such as electrons, muons, photons which are there at the destination, a jet is a whole journey.  Motivated from the trip, we decided to keep the book in the same spirit. That is, the reader will be a passenger and the author as tour guide. At every stop of the journey, the reader will see a video first then the tour guide will explain what it is from the physics point of view. Special thanks to Theo for all his blissful ideas and suggestions.

We also have the ROOT version JitJet.root which holds all the necessary data in ROOT flat Tree format which a reader can simply visualise using TBrowser. In the complex workflow like that of CMS, a “message” very quickly gets hidden between the "messengers" and “medium” of different stages. For example, a simple event at patron level is stored in LHE format then as it travels, in HepMC, DIGI, RAW, RECO, etc format. It is extremely challenging to track the same event due to use of different software packages needed at different stages. In this sense, the JitJet.root will be very useful as the entire journey will be in a single format.

The JitJet book can comfortably fit in a one semester JitJet course for a master student who has done a basic course on particle physics. It can also serve as a review project for new PhD students who will be willing to pursue a career in experimental particle physics. The JitJet book is equally relevant for theory/pheno people who struggle to understand any paper from CMS (in the last 10 years, I have come across numerous pheno physicists who had hard times reading the CMS paper). Keeping this in mind, the entire journey is shown using “standalone” scripts and “CMSSW” packages so that both non-CMS and CMS users can run the scripts.

The current JitJet book covers the Run-3 CMS (mostly 2024 and 2025) and lists out many open problems in the context of Phase2 HL-LHC. A few years later, we can have a JitJet@Phase2 book covering the Phase2 aspect. I would also like to write a MitMET (M is the MET) book in addition to the JitJet. The “M” would be unknown as MET is. The reader will pick what they prefer for M: Missing, Misbehaved, Miscalibrated, Many, etc. Or I might choose a name for “M” which suits the MET best  once we have the MitMET book mostly ready.

Finally, in the age of AI, I see “human” and “AI” relations as “director” and “producer”,  respectively. A director has the core ideas and visions and the producer is the one who implements them. A director has to ensure that every frame of a movie or line of a book is consistent, coherent. Then only we can have a good movie or a book. As humans, whatever fantasy we had,  whatever dream we wanted to realise, whatever craving we had could be fulfilled in the age of agentic AI. Next I would request the producer for his words.


Thank you,

[Photo]
Ravindra Kumar Verma
PostDoc Researcher
University of Helsinki

[Signature]

Date:


From Producer’s Desk:


I have added words from my Desk in the chapter/chapter_00_frontmatter.tex. I request you to please add yours since you are the Producer of the book :-)

Also skim down a bit the words from JitJet Desk.


thanks a lot. Please implement the folllowing:

* Fix typos, grammers, add paragraph spacing from Director's desk.
* add paragraph spacing for Producers desk
* Except the JitJet world table, wrap everything in plain paragraph format. Merge 2-3 paragraphs into one, reduce the number of words. 


The idea is to have uniform formats for all three Desk. If you prefer better (different) formating which are typically used in books, feel free to go for that.


