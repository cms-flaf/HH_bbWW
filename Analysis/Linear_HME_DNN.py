#python3 /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/AnaProd/NNInterface.py --inModelDir /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/nn_models --inFile /tmp/prsolank/luigi-tmp-416131263.root --outFileName /tmp/prsolank/luigi-tmp-862152055.root --uncConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/Run2_2018/weights.yaml --globalConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/global.yaml --EraName e2018 --Mass 400 --Spin 2 --PairType 2

from __future__ import annotations
import os, sys
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import FLAF.Common.Utilities as Utilities
import Analysis.hh_bbww as analysis


class LinHMEDNN:
    def __init__(self, cfg, payload_name):

        self.cfg = cfg
        self.payload_name = payload_name

        self.HME_coarse_bins = [-10, 250, 280, 300, 350, 400, 500, 600, 800, 10_000]

        # What to save in tmp file
        self.vars_to_save = set(['FullEventId', 'DoubleLepHME_mass', 'DNN_M0_Signal'])

    def run(self, array):
        print("Running LinHMEDNN producer")

        hme = array['DoubleLepHME_mass']
        dnn = array['DNN_M0_Signal']

        def find_index_between(num):
            lst = self.HME_coarse_bins
            for i in range(len(lst) - 1):
                if lst[i] <= num < lst[i + 1]:
                    return i
            return None

        hme_binned = [ find_index_between(hme_val) for hme_val in hme ]
        linear_hme_dnn = hme_binned + dnn

        array['linear_hme_dnn'] = np.array(linear_hme_dnn, dtype=np.float32)

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

        print("Finished my linHMEDNN, columns are")
        print(self.cfg['columns'])
        print("And array is ")
        print(array)

        return array

