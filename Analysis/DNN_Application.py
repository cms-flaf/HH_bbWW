#python3 /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/AnaProd/NNInterface.py --inModelDir /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/nn_models --inFile /tmp/prsolank/luigi-tmp-416131263.root --outFileName /tmp/prsolank/luigi-tmp-862152055.root --uncConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/Run2_2018/weights.yaml --globalConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/global.yaml --EraName e2018 --Mass 400 --Spin 2 --PairType 2

from __future__ import annotations
import os
import enum
from typing import Any, Dict
import ROOT
import pandas as pd
ROOT.ROOT.EnableImplicitMT()
import numpy as np
import awkward as ak
import tensorflow as tf
import sys
import tqdm
import FLAF.Common.Utilities as Utilities
import uproot
import onnxruntime as ort
from FLAF.RunKit.run_tools import ps_call
import shutil



def getKeyNames(root_file_name):
    root_file = ROOT.TFile(root_file_name, "READ")
    key_names = [str(k.GetName()) for k in root_file.GetListOfKeys() if root_file.Get(str(k.GetName())).InheritsFrom("TTree")]
    root_file.Close()
    return key_names

def createCentralQuantities(df_central, central_col_types, central_columns):
    map_creator = ROOT.analysis.MapCreator(*central_col_types)()
    df_central = map_creator.processCentral(ROOT.RDF.AsRNode(df_central), Utilities.ListToVector(central_columns))
    return df_central


def run_inference_on_events_tree(df_begin, models, globalConfig, dnnConfig, snapshotOptions):
    print(f"Running inference on the Events tree")
    # Create variables for DNN
    df = analysis.defineAllP4(df_begin)
    df = analysis.AddDNNVariables(df)
    out_names = []
    outFileName = 'tmp_Events.root'
    out_name = run_inference_for_tree(
        tree_name="Events",
        rdf=df,
        models=models,
        globalConfig=globalConfig,
        dnnConfig=dnnConfig,
        snapshotOptions=snapshotOptions,
        outFileName=outFileName
    )
    out_names.append(out_name)
    print(f"NN scores for Events tree saved in the output ROOT file")
    return out_names

# Not done yet
def run_inference_on_uncertainty_trees(df_begin, models, globalConfig, dnnConfig, unc_cfg_dict, scales, inFileName, snapshotOptions):
    dfWrapped_central = Utilities.DataFrameBuilderBase(df_begin)
    colNames = dfWrapped_central.colNames
    colTypes = dfWrapped_central.colTypes
    dfWrapped_central.df = createCentralQuantities(df_begin, colTypes, colNames)

    defaultColToSave = ["entryIndex", "luminosityBlock", "run", "event"]

    if dfWrapped_central.df.Filter("map_placeholder > 0").Count().GetValue() <= 0:
        raise RuntimeError("No events passed the map placeholder")

    snapshotOptions.fLazy = False
    out_names = []
    for uncName in unc_cfg_dict['shape']:
        for scale in scales:
            treeName = f"Events_{uncName}{scale}"
            for suffix in ["_noDiff", "_Valid", "_nonValid"]:
                treeName_with_suffix = f"{treeName}{suffix}"
                if treeName_with_suffix in getKeyNames(inFileName):
                    print(f"  Processing {treeName_with_suffix}")
                    df_unc = ROOT.RDataFrame(treeName_with_suffix, inFileName)
                    dfWrapped_unc = Utilities.DataFrameBuilderBase(df_unc)
                    if "_nonValid" not in treeName_with_suffix:
                        dfWrapped_unc.CreateFromDelta(colNames, colTypes)
                    dfWrapped_unc.AddMissingColumns(colNames, colTypes)    
                    # dfW_unc = Utilities.DataFrameWrapper(dfWrapped_unc.df, defaultColToSave)
                    
                    # Create variables for DNN
                    dfW_unc = analysis.defineAllP4(dfWrapped_unc.df)
                    dfW_unc = analysis.AddDNNVariables(dfW_unc)
                    outFileName = f'tmp_{treeName_with_suffix}.root'
                    out_name = run_inference_for_tree(
                        tree_name=treeName_with_suffix,
                        rdf=dfW_unc,
                        models=models,
                        globalConfig=globalConfig,
                        dnnConfig=dnnConfig,
                        snapshotOptions=snapshotOptions,
                        outFileName=outFileName
                    )
                    out_names.append(out_name)
    
    print(f"NN scores saved to {output_file_name}")
    return out_names

def run_inference_for_tree(tree_name, rdf, models, globalConfig, dnnConfig, snapshotOptions, outFileName):
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

    features_to_drop = load_features.copy() #We don't need to save these in the final file

    load_features.update(["entryIndex", "luminosityBlock", "run", "event"])

    # print("Loading these features")
    # print(load_features)


    nClasses = dnnConfig['nClasses'] if 'nClasses' in dnnConfig.keys() else 3
    nParity = dnnConfig['nParity'] if 'nParity' in dnnConfig.keys() else 4

    use_parametric = dnnConfig['use_parametric']
    param_mass_list = [250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000 ]

    class_names_list = dnnConfig['class_names'] if 'class_names' in dnnConfig.keys() else ['Signal', 'TT', 'DY']

    if not use_parametric:
        param_mass_list = [0]

    def run_inference_and_save(rdf_filtered):
        # This rdf thing is so horrible with the jet branch being a vector
        # It is actually better to just save the thing and then re open the root file with uproot

        vars_to_save = Utilities.ListToVector(load_features)
        rdf_filtered.Snapshot(tree_name, "test.root", vars_to_save, snapshotOptions)

        # Now open it with uproot!
        events = uproot.open("test.root")
        branches = events[tree_name].arrays(load_features)

        all_predictions = np.zeros((len(param_mass_list), len(branches.event), nParity, nClasses))

        for parityIdx, [model, parityfunc] in enumerate(models):
            #We want to only apply the 3 models that are NOT trained on this parity
            ones = np.ones_like(all_predictions)
            zeros = np.zeros_like(all_predictions)

            sess = ort.InferenceSession(f"{model}.onnx")

            #Get single value array
            array = np.array([getattr(branches, feature_name) for feature_name in features]).transpose()

            #Get vector value array
            default_value = 0.0
            if list_features != None:
                array_listfeatures = np.array([ak.fill_none(ak.pad_none(getattr(branches, feature_name), index+1), default_value)[:,index] for [feature_name,index] in list_features]).transpose()
                #Need to append the value features and the listfeatures together
                array = np.append(array, array_listfeatures, axis=1)

            #Need to append the high level features and the other features together
            if highlevel_features != None: 
                array_highlevelfeatures = np.array([getattr(branches, feature_name) for feature_name in highlevel_features]).transpose()
                array = np.append(array, array_highlevelfeatures, axis=1)


            #Add parametric mass point to the array
            for param_idx, param_mass in enumerate(param_mass_list):
                param_array = np.array([[param_mass for x in array]]).transpose()
                if use_parametric:
                    final_array = np.append(array, param_array, axis=1)
                else:
                    final_array = array

                # prediction = model.predict(final_array)
                prediction = sess.run(None, {'x': final_array}) # Take only first entry, prediction is [ [Sig, TT, DY], [mBB_SR] ]

                class_prediction = prediction[0]
                adv_prediction = prediction[1]

                # Now we need to set the trained parity to 0
                # But if there is only one model, then skip parity
                event_branch = np.expand_dims(branches.event, axis=-1)
                parity_filter = np.repeat(event_branch, nClasses, axis=-1)
                if nParity != 1:
                    class_prediction = np.where(
                        parity_filter % nParity != parityIdx,
                        class_prediction,
                        0.0
                    )

                all_predictions[param_idx,:,parityIdx,:] = class_prediction


        all_predictions = np.sum(all_predictions, axis=2) # Need to take average of the existing parity branches
        if nParity != 1: all_predictions = all_predictions/(nParity-1) # So we want to divide by nParity-1 (4 parity -> train with 1, apply with remaining 3)


        # Last save the branches
        for param_idx, param_mass in enumerate(param_mass_list):
            this_param_prediction = all_predictions[param_idx,:,:] # Now we want to get the individual param masses predictions for filling

            for class_idx, class_name in enumerate(class_names_list):
                branches[f'dnn_M{param_mass}_{class_name}'] = this_param_prediction.transpose()[class_idx]

                # branches[f'dnn_M{param_mass}_Signal'] = prediction.transpose()[0]
                # branches[f'dnn_M{param_mass}_TT'] = prediction.transpose()[1]
                # branches[f'dnn_M{param_mass}_DY'] = prediction.transpose()[2]

        #But we want to drop the features from this outfile
        # print("Dropping ", features_to_drop)
        for feature in features_to_drop:
            del branches[feature]



        # if os.path.isfile(outFileName):
        #     with uproot.update(outFileName) as outfile:
        #         print("Opened with update")
        #         outfile[tree_name] = branches
        #         print("Filled with tree")
        #         print(branches)

        # else:
        #     print(f"File doesn't exist, creating {outFileName}")
        #     with uproot.recreate(outFileName) as outfile:
        #         outfile[tree_name] = branches
        #         outfile.close()

        with uproot.recreate(outFileName) as outfile:
            outfile[tree_name] = branches
            outfile.close()
        
        return outFileName




    return run_inference_and_save(rdf)

if __name__ == "__main__":

    sys.path.append(os.environ['ANALYSIS_PATH'])
    ROOT.gROOT.ProcessLine(".include "+ os.environ['ANALYSIS_PATH'])
    ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisTools.h"')
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/pnetSF.h"')
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisMath.h"')
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')

    import yaml
    import argparse
    import numpy as np
    from FLAF.Analysis.HistHelper import *
    from FLAF.Common.Utilities import *
    import Analysis.hh_bbww as analysis
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--inFileName', required=True, type=str)
    parser.add_argument('--outFileName', required=True, type=str)
    parser.add_argument('--dnnFolder', required=True, type=str)
    parser.add_argument('--globalConfig', required=True, type=str)  
    parser.add_argument('--uncConfig', required=True, type=str)
    parser.add_argument('--compute_unc_variations', required=False, type=bool, default=None) # Dummy var for now

    args = parser.parse_args()

    with open(args.globalConfig, 'r') as f:
        globalConfig = yaml.safe_load(f)

    with open(args.uncConfig, 'r') as f:
        unc_cfg_dict = yaml.safe_load(f)

    startTime = time.time()

    outFileNameFinal = f'{args.outFileName}'

    # Check if the Events branch is in the inFile
    inFile_root = ROOT.TFile.Open(args.inFileName,"READ")
    inFile_keys = [k.GetName() for k in inFile_root.GetListOfKeys()]
    events_tree_exists = "Events" in inFile_keys
    if not events_tree_exists:
        print("No events tree, file is empty will create empty output")
        outfile = uproot.recreate(f'{args.outFileName}')

    
    else:
        scales = ['Up', 'Down']
        df_begin = ROOT.RDataFrame("Events", args.inFileName)
        snapshotOptions = ROOT.RDF.RSnapshotOptions()

        print("Loading Models----")
        dnnConfig = {}
        with open(os.path.join(args.dnnFolder, "dnn_config.yaml"), 'r') as file:
            dnnConfig = yaml.safe_load(file)  
        modelname_parity = dnnConfig['modelname_parity']

        models = [[os.path.join(args.dnnFolder, x),y] for x,y in modelname_parity]
        print("Models Loaded----")

        output_file_name = f'{args.outFileName}'

        # Run inference on the Events tree
        events_file_names = run_inference_on_events_tree(
            df_begin=df_begin,
            models=models,
            globalConfig=globalConfig,
            dnnConfig=dnnConfig,
            snapshotOptions=snapshotOptions,
        )
        
        # Run inference on uncertainty trees
        unc_file_names = run_inference_on_uncertainty_trees(
            df_begin=df_begin,
            models=models,
            globalConfig=globalConfig,
            dnnConfig=dnnConfig,
            unc_cfg_dict=unc_cfg_dict,
            scales=scales,
            inFileName=args.inFileName,
            snapshotOptions=snapshotOptions,
        )
        

        #ps call
        all_files = events_file_names + unc_file_names
        hadd_str = f'hadd -f209 -j 6 -n 0 {output_file_name} '
        hadd_str += ' '.join(f for f in all_files)
        print(len(all_files))
        print(hadd_str)
        if len(all_files) > 1:
            ps_call([hadd_str], True)
        else:
            shutil.copy(all_files[0],output_file_name)

        print(f"Processed and saved NN scores for all trees into {output_file_name}.")


