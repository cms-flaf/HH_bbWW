import pandas as pd
import numpy as np
import awkward as ak
import uproot
import vector
from sklearn.utils import shuffle
from Common.DataWrapper_utils import *


class DataWrapper():
    def __init__(self, cfg):
        self.n_jets = cfg['n_jets']
        self.jet_obs = cfg['jet_observables']

        self.tree_name = cfg['tree_name']
        self.features = [f"centralJet{i}_{obs}" for i in range(self.n_jets) for obs in self.jet_obs]
        self.labels = cfg['labels']
        self.extra_data = cfg['extra_data']

        self.modulo = cfg['modulo']
        self.train_val = cfg['train_val']
        self.test_val = cfg['test_val']

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

        PxPyPzE = ['px', 'py', 'pz', 'E']
        func_map = {'px': Px, 'py': Py, 'pz': Pz, 'E': E}

        d2 = {}
        for i in range(self.n_jets):
            for var in self.jet_obs:
                if var in PxPyPzE:
                    func = func_map[var]
                    var_awkward_array = func(centralJet_p4)
                else:
                    branch_name = f"centralJet_{var}"
                    var_awkward_array = branches[branch_name]
                d2[f"centralJet{i}_{var}"] = GetNumPyArray(var_awkward_array, self.n_jets, i)

        HVV_p4 = vector.zip({'pt': branches['genHVV_pt'],
                            'eta': branches['genHVV_eta'],
                            'phi': branches['genHVV_phi'],
                            'mass': branches['genHVV_mass']})

        for var in PxPyPzE:
            branch_name = f"H_VV_{var}"
            func = func_map[var]
            var_awkward_array = func(HVV_p4)
            d2[branch_name] = ak.to_numpy(var_awkward_array)

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
        train_df = self.SelectEvents(self.train_val, self.modulo)
        test_df = self.SelectEvents(self.test_val, self.modulo)

        self.test_labels = test_df[self.labels] # contain px, py, pz, E of H->VV and true X_mass
        test_df = test_df.drop(self.labels, axis=1)
        
        self.train_events = train_df['event'] 
        self.test_events = test_df['event']

        test_df = test_df.drop(['event'], axis=1)
        train_df = train_df.drop(['event'], axis=1)
       
        self.train_features = train_df[self.features] # contain all possible centralJet variables for all central jets (for train)
        self.test_features = test_df[self.features] # contain all possible centralJet variables for all central jets (for test)
        self.train_labels = train_df[self.labels] # contain X_mass and all variables of H->VV


    def SelectEvents(self, value, modulo):
        return self.data[self.data['event'] % modulo == value]
