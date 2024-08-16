import pandas as pd
import numpy as np
import uproot
from sklearn.utils import shuffle


class JetNetData():
    # features: list of strings with names of features to be used in training
    # labels: list of strings with names of labels to be used in training
    # train_test_fraction: float how much split full dataset into train/test parts
    def __init__(self, features, labels, train_test_fraction=0.8):
        self.features = features
        self.labels = labels 

        # pd.DataFrame containing full dataset
        self.data = pd.DataFrame(columns=[*features, *labels])

        self.train_frac = train_test_fraction

        self.train_features = pd.DataFrame()
        self.test_features = pd.DataFrame()

        self.train_labels = pd.DataFrame()
        self.test_labels = pd.DataFrame()


    def ReadFile(self, file_name):
        file = uproot.open(file_name)
        tree = file['JetNetTree']
        branches = tree.arrays()

        data_dict = {name: np.array(branches[name]) for name in self.features + self.labels}
            
        df = pd.DataFrame.from_dict(data_dict)
        print(f"Reading {file_name}, containing {df.shape[0]} entries")
        self.data = pd.concat([self.data, df]) 


    def ReadFiles(self, input_files):
        for file_name in input_files:
            self.ReadFile(file_name)


    def Shuffle(self):
        self.data = shuffle(self.data)


    def TrainTestSplit(self):
        train_df = self.data.sample(frac=self.train_frac, random_state=42)
        test_df = self.data.drop(train_df.index)

        self.train_features = train_df[self.features]
        self.test_features = test_df[self.features]
        self.train_labels = train_df[self.labels]
        self.test_labels = test_df[self.labels]