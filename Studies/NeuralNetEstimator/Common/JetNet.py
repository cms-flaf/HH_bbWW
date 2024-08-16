import tensorflow as tf
import pandas as pd
import numpy as np
import uproot

from Common.JetNet_utils import MXLossFunc, H_WW_MXLossFunc


class JetNet():
    def __init__(self, features, labels, cfg):
        self.features = features
        self.labels = labels

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
        return self.model.predict(test_features)


    def SaveModel(self, path):
        self.model.save(f"{path}{self.name}.keras")

    
    def LoadModel(self, model_name):
        # to load with compilation custom_objects needs to be passed
        # couldn't make it work
        # if model is loaded only for prediction, compilation is not needed
        self.model = tf.keras.models.load_model(model_name, compile=False)
        # self.model = tf.keras.models.load_model(model_name, custom_objects={'loss' : MXLossFunc(target, output)})
        print(f"Loaded model {model_name}")
        print(self.model.summary())
