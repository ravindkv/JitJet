
How to run the PS step inside CMSSW:

+ cmsrel CMSSW_15_0_5
+ cd CMSSW_15_0_5/src
+ cmsenv
+ git clone git@github.com:ravindkv/JitJetSW.git
+ cd JitJetSW
+ scram b
+ cd Journey/test
+ ./runCMSDrivers_mc2024_BBbar_GEN.sh

This will produce pfds. Have a look at the
genPartAnalyzer_ME_PS_MPI_out.pdf
files which are also copied here

