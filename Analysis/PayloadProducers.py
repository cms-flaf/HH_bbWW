from Studies.HME.new.hmeVariables import GetHMEVariables
from Analysis.DNN_Application import ApplyDNN
import Analysis.hh_bbww as analysis
import ROOT
import sys
import os

class HMEProducer:
    def __init__(self, cfg, payload_name):
        self.cfg = cfg
        self.payload_name = payload_name

    def run(self, dfw):
        if "ncentralJet" not in dfw.df.GetColumnNames():
            dfw.Define("ncentralJet", "return centralJet_pt.size();")

        ch = self.cfg['channel']
        if ch == "DL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0")
        elif ch == "SL":
            dfw.Define("has_necessary_inputs", "ncentralJet >= 4 && lep1_pt > 0.0")
        else:
            raise RuntimeError(f"Illegal channel in config: {ch}")

        dfw.df = GetHMEVariables(dfw.df, ch)
        for col in self.cfg['columns']:
            if col != 'valid':
                dfw.DefineAndAppend(f"{self.payload_name}_{col}", f"return hme_output[static_cast<size_t>(HME::EstimOut::{col})];")
        if 'valid' in self.cfg['columns']:
            dfw.DefineAndAppend(f"{self.payload_name}_valid", f"return {self.payload_name}_mass > 0.0;")
        return dfw

class DNNProducer:
    def __init__(self, cfg, payload_name):
        import yaml
        self.cfg = cfg
        self.payload_name = payload_name

        sys.path.append(os.environ['ANALYSIS_PATH'])
        ROOT.gROOT.ProcessLine(".include "+ os.environ['ANALYSIS_PATH'])
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisMath.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')

        dnnConfig = {}
        dnnFolder = os.path.join(os.environ["ANALYSIS_PATH"], "config", "DNN", self.cfg['version'])
        with open(os.path.join(dnnFolder, "dnn_config.yaml"), 'r') as file:
            dnnConfig = yaml.safe_load(file)

        #Features to use for DNN application (single vals)
        features = dnnConfig['features']
        #Features to use for DNN application (vectors and index)
        list_features = dnnConfig['listfeatures']
        #Features to use for DNN application (high level names to create)
        highlevel_features = dnnConfig['highlevelfeatures']

        #Features to load from df to awkward
        load_features = set()
        load_features.update(features)
        for feature in list_features:
            load_features.update([feature[0]])
        load_features.update(highlevel_features)

        load_features.update(["FullEventId"])

        # What to save in tmp file
        self.vars_to_save = load_features
        # What to save for final output
        self.cols_to_save = [ f"{self.payload_name}_{col}" for col in self.cfg['columns'] ]

    def prepare_dfw(self, dfw):
        print("Running DNN preparer")

        dfw.df = analysis.defineAllP4(dfw.df)
        dfw.df = analysis.AddDNNVariables(dfw.df)

        return dfw


    def run(self, array):
        print("Running DNN producer")

        array = ApplyDNN(array, self.cfg)


        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg['columns']:
                if col != 'FullEventId':
                    del array[col]
                    
        # Rename the branches
        for col in self.cfg['columns']:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")


        return array
