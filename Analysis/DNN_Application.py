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


def ApplyDNN(branches, cfg):
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
    if 'ncentralJet' in branches.fields: features_to_drop.update(["ncentralJet"]) # I don't know why, but sometimes this is there?

    load_features.update(["FullEventId"])

    nEvents = len(branches)
    print(f"Running DNN Over {nEvents} events")

    # Initialize the final predictions per parametric per event per class
    all_predictions = np.zeros((len(param_mass_list), nEvents, nClasses))

    event_number = branches.FullEventId & 0xFFFFFFFF

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

    # Initialize the local predictions including per parity, but this will be summed out later
    local_predictions = np.zeros((len(param_mass_list), len(array), nParity, nClasses))

    for parityIdx, [model, parityfunc] in enumerate(models):
        #We want to only apply the 3 models that are NOT trained on this parity
        sess = ort.InferenceSession(f"{model}.onnx")

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
            event_num = np.expand_dims(event_number, axis=-1) # We now get event_number from the FullEventId branch earlier
            parity_filter = np.repeat(event_num, nClasses, axis=-1)
            if nParity != 1:
                class_prediction = np.where(
                    parity_filter % nParity != parityIdx,
                    class_prediction,
                    0.0
                )
            local_predictions[param_idx,:,parityIdx,:] = class_prediction

    # Reduce dimension by sum the parity axis from local to global
    all_predictions = np.sum(local_predictions, axis=2)


    if nParity != 1: all_predictions = all_predictions/(nParity-1) # So we want to divide by nParity-1 (4 parity -> train with 1, apply with remaining 3)

    # Last save the branches
    for param_idx, param_mass in enumerate(param_mass_list):
        this_param_prediction = all_predictions[param_idx,:,:] # Now we want to get the individual param masses predictions for filling

        for class_idx, class_name in enumerate(class_names_list):
            branches[f'M{param_mass}_{class_name}'] = this_param_prediction.transpose()[class_idx].astype(np.float32)

    for feature in features_to_drop:
        del branches[feature]

    print("Finishing call, memory?")
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {mem_mb:.2f} MB")

    return branches
    