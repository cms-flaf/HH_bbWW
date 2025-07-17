#python3 /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/AnaProd/NNInterface.py --inModelDir /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/nn_models --inFile /tmp/prsolank/luigi-tmp-416131263.root --outFileName /tmp/prsolank/luigi-tmp-862152055.root --uncConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/Run2_2018/weights.yaml --globalConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/global.yaml --EraName e2018 --Mass 400 --Spin 2 --PairType 2

from __future__ import annotations
import os
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import FLAF.Common.Utilities as Utilities



def ApplyDNN(df, cfg):
    dnnConfig = {}
    dnnFolder = os.path.join(os.environ["ANALYSIS_PATH"], "config", "DNN", cfg['version'])
    with open(os.path.join(dnnFolder, "dnn_config.yaml"), 'r') as file:
        dnnConfig = yaml.safe_load(file)  
    modelname_parity = dnnConfig['modelname_parity']

    models = [[os.path.join(dnnFolder, x),y] for x,y in modelname_parity]

    #Features to use for DNN application (single vals)
    features = dnnConfig['features']
    #Features to use for DNN application (vectors and index)
    list_features = dnnConfig['listfeatures']
    #Features to use for DNN application (high level names to create)
    highlevel_features = dnnConfig['highlevelfeatures']

    nClasses = dnnConfig['nClasses'] if 'nClasses' in dnnConfig.keys() else 3
    nParity = dnnConfig['nParity'] if 'nParity' in dnnConfig.keys() else 4


    use_parametric = dnnConfig['use_parametric']
    param_mass_list = [ 250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000 ]
    
    class_names_list = dnnConfig['class_names'] if 'class_names' in dnnConfig.keys() else ['Signal', 'TT', 'DY']

    if not use_parametric:
        param_mass_list = [0]

    #Features to load from df to awkward
    load_features = set()
    load_features.update(features)
    for feature in list_features:
        load_features.update([feature[0]])
    load_features.update(highlevel_features)

    features_to_drop = load_features.copy() #We don't need to save these in the final file

    load_features.update(["FullEventId"])

    # Sometimes (often) the ak.from_rdataframe() line will crash with MemoryError on the first uncertainty tree
    # It ALWAYS passes the central call, but will often (90%) crash on the very first uncertainty (JER_UP)
    # It doesn't seem to be python memory (psutil prints show only 2gb of ram usage), but instead ROOT/C++ memory
    # df.Display(var).AsString() works, so it isn't corrupted, just MemoryErrors
    # Saving a local file tmp.root seems to help the problem
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists=True
    snapshotOptions.fMode="RECREATE"
    varToSave = Utilities.ListToVector(list(load_features))
    df.Snapshot(f"tmp", f'tmp.root', varToSave, snapshotOptions)

    branches = ak.from_rdataframe(df, load_features)

    # Initialize the final predictions per parametric per event per class
    all_predictions = np.zeros((len(param_mass_list), len(branches.FullEventId), nClasses))

    # Move to do sub-loops of the array to avoid huge RAM usage
    iter_length = 500_000
    print(f"Total Events: {len(branches)}")
    for lower_bound in range(0, len(branches), iter_length):
        print(f"Loading {lower_bound} to {lower_bound+iter_length}")
        small_branches = branches[lower_bound:lower_bound+iter_length]
        small_event_branch = small_branches.FullEventId & 0xFFFFFFFF

        #Get single value array
        small_array = np.array([getattr(small_branches, feature_name) for feature_name in features]).transpose()

        #Get vector value array
        default_value = 0.0
        if list_features != None:
            array_listfeatures = np.array([ak.fill_none(ak.pad_none(getattr(small_branches, feature_name), index+1), default_value)[:,index] for [feature_name,index] in list_features]).transpose()
            #Need to append the value features and the listfeatures together
            small_array = np.append(small_array, array_listfeatures, axis=1)

        #Need to append the high level features and the other features together
        if highlevel_features != None: 
            array_highlevelfeatures = np.array([getattr(small_branches, feature_name) for feature_name in highlevel_features]).transpose()
            small_array = np.append(small_array, array_highlevelfeatures, axis=1)

        # Initialize the local predictions including per parity, but this will be summed out later
        local_predictions = np.zeros((len(param_mass_list), len(small_array), nParity, nClasses))

        for parityIdx, [model, parityfunc] in enumerate(models):
            #We want to only apply the 3 models that are NOT trained on this parity
            sess = ort.InferenceSession(f"{model}.onnx")

            #Add parametric mass point to the array
            for param_idx, param_mass in enumerate(param_mass_list):
                param_array = np.array([[param_mass for x in small_array]]).transpose()
                if use_parametric:
                    final_array = np.append(small_array, param_array, axis=1)
                else:
                    final_array = small_array

                # prediction = model.predict(final_array)
                prediction = sess.run(None, {'x': final_array}) # Take only first entry, prediction is [ [Sig, TT, DY], [mBB_SR] ]

                class_prediction = prediction[0]
                adv_prediction = prediction[1]

                # Now we need to set the trained parity to 0
                # But if there is only one model, then skip parity
                event_num = np.expand_dims(small_event_branch, axis=-1) # We now get event_branch from the FullEventId branch earlier
                parity_filter = np.repeat(event_num, nClasses, axis=-1)
                if nParity != 1:
                    class_prediction = np.where(
                        parity_filter % nParity != parityIdx,
                        class_prediction,
                        0.0
                    )
                local_predictions[param_idx,:,parityIdx,:] = class_prediction

        # Reduce dimension by sum the parity axis from local to global
        all_predictions[:,lower_bound:lower_bound+iter_length,:] = np.sum(local_predictions, axis=2)


    if nParity != 1: all_predictions = all_predictions/(nParity-1) # So we want to divide by nParity-1 (4 parity -> train with 1, apply with remaining 3)


    # Last save the branches
    for param_idx, param_mass in enumerate(param_mass_list):
        this_param_prediction = all_predictions[param_idx,:,:] # Now we want to get the individual param masses predictions for filling

        for class_idx, class_name in enumerate(class_names_list):
            branches[f'M{param_mass}_{class_name}'] = this_param_prediction.transpose()[class_idx].astype(np.float32)


    #But we want to drop the features from this outfile
    for feature in features_to_drop:
        del branches[feature]

    # to_rdataframe requires a different format than the output of from_rdataframe (dumb)
    dict_for_df = {}
    for field_name in branches.fields:
        dict_for_df[field_name] = branches[field_name]

    new_df = ak.to_rdataframe(dict_for_df)

    return new_df
