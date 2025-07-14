from Studies.HME.new.hmeVariables import GetHMEVariables
from Analysis.DNN_Application import ApplyDNN
import Analysis.hh_bbww as analysis
import ROOT
import sys
import os

class HMEProducer:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, dfw):
        if "ncentralJet" not in dfw.df.GetColumnNames():
            dfw.Define("ncentralJet", "return centralJet_pt.size();")
        if self.cfg['channel'] == "DL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0")
        elif self.cfg['channel'] == "SL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 4 && lep1_pt > 0.0")
        
        dfw.df = GetHMEVariables(dfw.df, self.cfg['channel'])
        for col in self.cfg['columns']:
            if col != 'valid':
                dfw.DefineAndAppend(f"HME_{col}", f"return hme_output[static_cast<size_t>(HME::EstimOut::{col})];")
        if 'valid' in self.cfg['columns']:
            dfw.DefineAndAppend("HME_valid", "return HME_mass > 0.0;")
        return dfw

class DNNProducer:
    def __init__(self, cfg):
        self.cfg = cfg



        sys.path.append(os.environ['ANALYSIS_PATH'])
        ROOT.gROOT.ProcessLine(".include "+ os.environ['ANALYSIS_PATH'])
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/pnetSF.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisMath.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')
        self.counter = 0

    def run(self, dfw):
        print("Running DNN producer")
        print(self.cfg)
        print(os.environ["ANALYSIS_PATH"])
        print(self.counter)
        self.counter += 1

        dfw.df = analysis.defineAllP4(dfw.df)
        dfw.df = analysis.AddDNNVariables(dfw.df)

        dfw.df = ApplyDNN(dfw.df, self.cfg, self.counter)
        for col in self.cfg['columns']:
            dfw.DefineAndAppend(f"DNN_{col}", f"return {col};")
            dfw.df.Display(f'DNN_{col}').AsString() # I need this so the rdf doesn't segfault in final saving for some reason
        return dfw