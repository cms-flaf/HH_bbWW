import pandas as pd
import numpy as np
import awkward as ak
import uproot
import vector
from sklearn.utils import shuffle


class DataWrapper():
    def __init__(self, cfg):
        jet_obs = cfg['jet_observables']
        self.n_jets = cfg['n_jets']

        self.tree_name = cfg['tree_name']
        self.features = [f"centralJet{i}_{obs}" for i in range(self.n_jets) for obs in jet_obs]
        self.labels = cfg['labels']
        self.extra_data = cfg['extra_data']

        # pd.DataFrame containing full dataset
        self.data = pd.DataFrame(columns=[*self.features, *self.labels, *self.extra_data])


    def ReadFile(self, file_name):
        file = uproot.open(file_name)
        tree = file[self.tree_name]
        branches = tree.arrays()

        d1 = {name: np.array(branches[name]) for name in self.extra_data}
        d1["X_mass"] = np.array(branches['X_mass'], dtype=float)

        centralJet_p4 = vector.zip({'pt': branches['centralJet_pt'], 
                                    'eta': branches['centralJet_eta'], 
                                    'phi': branches['centralJet_phi'], 
                                    'mass': branches['centralJet_mass']})

        d2 = {}
        for i in range(self.n_jets):
            d2[f"centralJet{i}_px"] = ak.to_numpy(ak.fill_none(ak.pad_none(centralJet_p4.px[:, :self.n_jets], self.n_jets), 0.0))[:, i]
            d2[f"centralJet{i}_py"] = ak.to_numpy(ak.fill_none(ak.pad_none(centralJet_p4.py[:, :self.n_jets], self.n_jets), 0.0))[:, i]
            d2[f"centralJet{i}_pz"] = ak.to_numpy(ak.fill_none(ak.pad_none(centralJet_p4.pz[:, :self.n_jets], self.n_jets), 0.0))[:, i]
            d2[f"centralJet{i}_E"] = ak.to_numpy(ak.fill_none(ak.pad_none(centralJet_p4.E[:, :self.n_jets], self.n_jets), 0.0))[:, i]
            d2[f"centralJet{i}_btagPNetB"] = ak.to_numpy(ak.fill_none(ak.pad_none(branches['centralJet_btagPNetB'][:, :self.n_jets], self.n_jets), 0.0))[:, i] 
            d2[f"centralJet{i}_PNetRegPtRawCorr"] = ak.to_numpy(ak.fill_none(ak.pad_none(branches['centralJet_PNetRegPtRawCorr'][:, :self.n_jets], self.n_jets), 0.0))[:, i]
            d2[f"centralJet{i}_PNetRegPtRawCorrNeutrino"] = ak.to_numpy(ak.fill_none(ak.pad_none(branches['centralJet_PNetRegPtRawCorrNeutrino'][:, :self.n_jets], self.n_jets), 0.0))[:, i]  
            d2[f"centralJet{i}_PNetRegPtRawRes"] = ak.to_numpy(ak.fill_none(ak.pad_none(branches['centralJet_PNetRegPtRawRes'][:, :self.n_jets], self.n_jets), 0.0))[:, i] 

        HVV_p4 = vector.zip({'pt': branches['genHVV_pt'],
                            'eta': branches['genHVV_eta'],
                            'phi': branches['genHVV_phi'],
                            'mass': branches['genHVV_mass']})

        d2["H_VV_px"] = ak.to_numpy(HVV_p4.px)
        d2["H_VV_py"] = ak.to_numpy(HVV_p4.py)
        d2["H_VV_pz"] = ak.to_numpy(HVV_p4.pz)
        d2["H_VV_E"] = ak.to_numpy(HVV_p4.E)

        data_dict = d1 | d2

        df = pd.DataFrame.from_dict(data_dict)
        print(f"Reading {df.shape[0]} entries from {file_name}")
        self.data = pd.concat([self.data, df]) 


    def ReadFiles(self, input_files):
        for file_name in input_files:
            self.ReadFile(file_name)


    def Shuffle(self):
        self.data = shuffle(self.data)


    def TrainTestSplit(self):
        self.Shuffle()
        train_df = self.SelectEvents(0, 2)
        test_df = self.SelectEvents(1, 2)

        test_df = test_df.drop(self.labels, axis=1)
        
        self.train_events = train_df['event']
        self.test_events = test_df['event']

        test_df = test_df.drop(['event'], axis=1)
        train_df = train_df.drop(['event'], axis=1)
       
        self.train_features = train_df[self.features]
        self.test_features = test_df[self.features]
        self.train_labels = train_df[self.labels]


    def SelectEvents(self, value, modulo):
        return self.data[self.data['event'] % modulo == value]
