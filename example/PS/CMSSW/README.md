
How to run the PS step inside CMSSW:

+ cmsrel CMSSW_15_0_5
+ cd CMSSW_15_0_5/src
+ cmsenv
+ git clone git@github.com:ravindkv/JitJetSW.git
+ cd JitJetSW
+ scram b
+ cd Journey/test
+ cmsRun step1_GEN_cfg.py
+ cmsRun genPartAnalyzer_cfg.py

This will produce pfds. Have a look at the
genPartAnalyzer_stage1_hardProcess_run1_event1.pdf
genPartAnalyzer_stage2_partonShower_run1_event1.pdf
files which are also copied here

