import tensorflow as tf
import pandas as pd
import numpy as np
import uproot
import pandas as pd

from Common.JetNet_utils import MXLossFunc, H_WW_MXLossFunc


class JetNet():
    def __init__(self, cfg):
        jet_obs = cfg['jet_observables']
        n_jets = cfg['n_jets']

        self.features = [f"centralJet{i}_{obs}" for i in range(n_jets) for obs in jet_obs]
        self.labels = cfg['labels']

        # training parameters
        self.lr = cfg['learning_rate']
        self.n_epochs = cfg['n_epochs']
        self.batch_size = cfg['batch_size']
        self.verbosity = cfg['verbosity']
        self.valid_split = cfg['valid_split']
        self.name = cfg['name']
        self.topology = cfg['topology']

        self.model = None


    def ConfigureModel(self, dataset_shape):
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(layer_size, activation='relu') for layer_size in self.topology])
        self.model.add(tf.keras.layers.Dense(3))

        self.model.compile(loss=MXLossFunc, optimizer=tf.keras.optimizers.Adam(self.lr))
        self.model.build(dataset_shape)


    def Fit(self, train_features, train_labels):
        if not self.model:
            raise RuntimeError("Model has not been configured before fitting")
        history = self.model.fit(train_features,
                                 train_labels,
                                 validation_split=self.valid_split,
                                 verbose=self.verbosity,
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs)
        return history


    def Predict(self, test_features):
        if not np.all(test_features.columns == self.features):
            raise RuntimeError(f"Features pased for prediction do not match expected features: passed {test_features.columns}, while expected {self.features}")
        # returns predicted variables: px, py, pz of H->bb
        pred_p3 = self.model.predict(test_features)
        pred_en = np.sqrt(125.0**2 + np.sum(np.square(pred_p3), axis=1))
        pred_df = pd.DataFrame({"H_bb_px": pred_p3[:, 0], "H_bb_py": pred_p3[0:, 1], "H_bb_pz": pred_p3[:, 2]}, "H_bb_E": pred_en)
        return pred_df


    def SaveModel(self, path):
        self.model.save(f"{path}{self.name}.keras")

    
    def LoadModel(self, model_name):
        self.model = tf.keras.models.load_model(model_name, compile=False)
        print(f"Loaded model {model_name}")
        print(self.model.summary())
