import tensorflow as tf
import numpy as np
import uproot 
import awkward as ak 
import os
import matplotlib.pyplot as plt
import sklearn.metrics
import yaml
import ROOT
import tf2onnx
import onnx
import onnxruntime as ort
import shutil
import copy


import threading
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread

#Need to get train_features and train_labels
class DataWrapper():
    def __init__(self):
        print("Init data wrapper")

        self.feature_names = None
        self.listfeature_names = None
        self.highlevelfeatures_names = None
        self.label_name = None
        self.mbb_name = None

        self.features_no_param = None
        self.features = None
        self.listfeatures = None
        self.hlv = None
        self.param_values = None
        self.labels = None
        self.labels_binary = None
        self.mbb = None
        self.mbb_region = None

        self.class_weight = None
        self.adv_weight = None
        self.class_target = None
        self.adv_target = None
        self.adv_regression_target = None

        self.param_list = [250, 260, 270, 280, 300, 350, 450, 550, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000 ]
        self.use_parametric = False

        self.features_paramSet = None

    def UseParametric(self, use_parametric):
        self.use_parametric = use_parametric
        print(f"Parametric feature set to {use_parametric}")

    def SetParamList(self, param_list):
        self.param_list = param_list

    def SetPredictParamValue(self, param_value):
        #During predict, we want to use a truly random param value even for signal!
        if param_value not in self.param_list:
            print(f"This param value {param_value} is not an option!")
        new_params = np.array([[param_value for x in self.features]]).transpose()

        self.features_paramSet = np.append(self.features_no_param, new_params, axis=1)

    def AddInputFeatures(self, features):
        if self.feature_names == None:
            self.feature_names = features
        else:
            self.feature_names = self.feature_names + features

        print(f"Added features {features}")
        print(f"New feature list {self.feature_names}")

    def AddInputFeaturesList(self, features, index):
        if self.listfeature_names == None:
            self.listfeature_names = [[feature, index] for feature in features]
        else:
            self.listfeature_names = self.listfeature_names + [[feature, index] for feature in features]

        print(f"Added features {features} with index {index}")
        print(f"New feature list {self.listfeature_names}")

    def AddHighLevelFeatures(self, features):
        if self.highlevelfeatures_names == None:
            self.highlevelfeatures_names = features
        else:
            self.highlevelfeatures_names = self.highlevelfeatures_names + features

        print(f"Added high level features {features}")
        print(f"New feature list {self.highlevelfeatures_names}")


    def AddInputLabel(self, labels_name):
        if self.label_name != None:
            print("What are you doing? You already defined the input label branch")
        self.label_name = labels_name

    def SetMbbName(self, mbb_name):
        if self.mbb_name != None:
            print("What are you doing? You already defined the mbb branch")
        self.mbb_name = mbb_name
        

    def ReadFile(self, file_name, entry_start=None, entry_stop=None):
        if self.feature_names == None:
            print("Uknown branches to read! DefineInputFeatures first!")
            return

        print(f"Reading file {file_name}")

        features_to_load = []
        features_to_load = features_to_load + self.feature_names
        for listfeature in self.listfeature_names:
           if listfeature[0] not in features_to_load: features_to_load.append(listfeature[0])
        features_to_load = features_to_load + self.highlevelfeatures_names

        features_to_load.append(self.mbb_name)
        features_to_load.append('X_mass')

        print(f"Only loading these features {features_to_load}")

        file = uproot.open(file_name)
        tree = file['Events']
        branches = tree.arrays(features_to_load, entry_start=entry_start, entry_stop=entry_stop)

        self.features = np.array([getattr(branches, feature_name) for feature_name in self.feature_names]).transpose()
        print("Got features, but its a np array")

        default_value = 0.0
        if self.listfeature_names != None: 
            self.listfeatures = np.array([ak.fill_none(ak.pad_none(getattr(branches, feature_name), index+1), default_value)[:,index] for [feature_name,index] in self.listfeature_names]).transpose()
        print("Got the list features")

        #Need to append the value features and the listfeatures together
        if self.listfeature_names != None: 
            print("We have list features!")
            self.features = np.append(self.features, self.listfeatures, axis=1)

        if self.highlevelfeatures_names != None: 
            self.hlv = np.array([getattr(branches, feature_name) for feature_name in self.highlevelfeatures_names]).transpose()
            self.features = np.append(self.features, self.hlv, axis=1)


        self.mbb = np.array(getattr(branches, self.mbb_name))
        self.adv_regression_target = self.mbb

        #Add parametric variable
        self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
        print("Got the param values")


        self.features_no_param = self.features
        if self.use_parametric: self.features = np.append(self.features, self.param_values, axis=1)


    def ReadWeightFile(self, weight_name, entry_start=None, entry_stop=None):
        print(f"Reading weight file {weight_name}")
        file = uproot.open(weight_name)
        tree = file['weight_tree']
        branches = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
        self.class_weight = np.array(getattr(branches, 'class_weight'))
        self.adv_weight = np.array(getattr(branches, 'adv_weight'))
        self.class_target = np.array(getattr(branches, 'class_target'))
        self.adv_target = np.array(getattr(branches, 'adv_target'))


@tf.function
def binary_entropy(target, output):
  epsilon = tf.constant(1e-7, dtype=tf.float32)
  x = tf.clip_by_value(output, epsilon, 1 - epsilon)
  return - target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)

@tf.function
def binary_focal_crossentropy(target, output, y_class, y_pred_class):
    gamma = 2.0 # Default from keras
    gamma = 0.0

    # Use signal from multiclass for focal check
    if y_class is not None:
      y_class = y_class[:,0]
      y_pred_class = y_pred_class[:,0]


    # Un-nest the output (currently in shape [ [1], [2], [3], ...] and we want in shape [1, 2, 3])
    y_true = target
    y_pred = output[:,0]


    bce = binary_entropy(y_true, y_pred)


    return bce


@tf.function
def accuracy(target, output):
  target = tf.expand_dims(target, axis=-1)
  return tf.cast(tf.equal(target, tf.round(output)), tf.float32)



@tf.function
def categorical_crossentropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    output = output / tf.reduce_sum(output, axis=-1, keepdims=True)
    output = tf.clip_by_value(output, epsilon, 1.0 - epsilon)
    log_prob = tf.math.log(output)
    return - tf.reduce_sum(target * log_prob, axis=-1)

@tf.function
def categorical_accuracy(target, output):
    y_true = target
    y_pred = output

    y_true = tf.argmax(y_true, axis=-1)

    reshape_matches = False
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_true.dtype)

    y_true_org_shape = tf.shape(y_true)
    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(y_true.shape) == len(y_pred.shape))
    ):
        y_true = tf.squeeze(y_true, -1)
        reshape_matches = True
    y_pred = tf.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast
    # them to match.
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.cast(y_pred, dtype=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    if reshape_matches:
        matches = tf.reshape(matches, y_true_org_shape)
    return matches


def ks_test(x, y):
    # x and y are nested, unnest them
    x_sorted = tf.sort(x[:,0])
    y_sorted = tf.sort(y[:,0])
    combined = tf.concat([x[:,0], y[:,0]], axis=0)
    sorted_combined = tf.sort(combined)

    n_x = tf.shape(x)[0]
    n_y = tf.shape(y)[0]

    cdf_x = tf.cast(tf.searchsorted(x_sorted, sorted_combined, side='right'), tf.float32) / tf.cast(n_x, tf.float32)
    cdf_y = tf.cast(tf.searchsorted(y_sorted, sorted_combined, side='right'), tf.float32) / tf.cast(n_y, tf.float32)

    delta = tf.abs(cdf_x - cdf_y)
    return tf.reduce_max(delta)






class EpochCounterCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.epoch_counter.assign_add(1.0)
        return
    
class AdvOnlyCallback(tf.keras.callbacks.Callback):
  def __init__(self, train_dataset, nSteps=100, TrackerWindowSize=10, on_batch=True, on_epoch=False, continue_training=False, quiet=False):
    self.train_dataset = train_dataset.repeat()
    self.trackerWindowSize = TrackerWindowSize
    self.nSteps = nSteps
    self.generator = self.looper()
    self.on_batch = on_batch
    self.on_epoch = on_epoch
    self.continue_training = continue_training #self.setup['continue_training'] When we continue, there is no point to skipping first epoch
    self.quiet = quiet

  def looper(self):
    yield
    n_window = 0
    nStep = 0
    for data in self.train_dataset:
      self.model._step_adv_only(data, True)
      n_window += 1
      if n_window == self.trackerWindowSize:
        if not self.quiet: print(f'\nSubmodule loss {self.model.adv_loss_tracker_submodule.result()} and accuracy {self.model.adv_accuracy_tracker_submodule.result()} after {nStep+1} nSteps')
        self.model.adv_loss_tracker_submodule.reset_state()
        self.model.adv_accuracy_tracker_submodule.reset_state()
        n_window = 0
      nStep += 1
      if nStep == self.nSteps:
        nStep = 0 # This is only a counter, so its fine to reset
        yield

  def on_batch_end(self, batch, logs=None):
    if self.nSteps <= 0: return
    if self.model.epoch_counter == 0. and not self.continue_training: return
    if self.on_batch:
      next(self.generator)
    

  def on_epoch_end(self, epoch, logs=None):
    if self.nSteps <= 0: return
    if self.model.epoch_counter == 0. and not self.continue_training: return
    if self.on_epoch:
       next(self.generator)



class ModelCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, filepath, monitor="val_loss", verbose=0, mode="min", min_delta=None, min_rel_delta=None,
               save_callback=None, patience=None, predicate=None, input_signature=None):
    super().__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.epochs_since_last_save = 0
    self.msg = None
    self.save_callback = save_callback
    self.patience = patience
    self.predicate = predicate
    self.input_signature = input_signature

    if os.path.exists(filepath):
      shutil.rmtree(filepath)

    self.best = None
    self.monitor_op = self._make_monitor_op(mode, min_delta, min_rel_delta)

  def _make_monitor_op(self, mode, min_delta, min_rel_delta):
    if mode == "min":
      if min_delta is None and min_rel_delta is None:
        return lambda current, best: best is None or best - current > 0
      if min_delta is None:
        return lambda current, best: best is None or (best - current) > min_rel_delta * best
      if min_rel_delta is None:
        return lambda current, best: best is None or best - current > min_delta
      return lambda current, best: best is None or (best - current) > min_rel_delta * best or best - current > min_delta
    elif mode == "max":
      if min_delta is None and min_rel_delta is None:
        return lambda current, best: best is None or current - best > 0
      if min_delta is None:
        return lambda current, best: best is None or (current - best) > min_rel_delta * best
      if min_rel_delta is None:
        return lambda current, best: best is None or current - best > min_delta
      return lambda current, best: best is None or (current - best) > min_rel_delta * best or current - best > min_delta
    else:
      raise ValueError(f"Unrecognized mode: {mode}")

  def _print_msg(self):
    if self.msg is not None:
      print(self.msg)
      self.msg = None

  def on_epoch_begin(self, epoch, logs=None):
    self._print_msg()

  def on_train_end(self, logs=None):
    self._print_msg()
    if self.best == None:
      print("Never found a best model, just save the last one")

      dir_name = f'epoch_final.keras'
      onnx_dir_name = f"epoch_final.onnx"
      os.makedirs(self.filepath, exist_ok = True)
      path = os.path.join(self.filepath, f'{dir_name}')
      self.model.save(path)
      if self.input_signature is not None:
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, self.input_signature, opset=13)
        onnx.save(onnx_model, os.path.join(self.filepath, f"{onnx_dir_name}"))

      path_best = os.path.join(self.filepath, 'best.onnx')
      path_best_keras = os.path.join(self.filepath, 'best.keras')
      if os.path.exists(path_best):
        os.remove(path_best)
        os.remove(path_best_keras)

      os.symlink(onnx_dir_name, path_best)
      os.symlink(dir_name, path_best_keras)



  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    current = logs.get(self.monitor)
    if self.monitor_op(current, self.best) and (self.predicate is None or self.predicate(self.model, logs)):
      dir_name = f'epoch_{epoch+1}.keras'
      onnx_dir_name = f"epoch_{epoch+1}.onnx"
      os.makedirs(self.filepath, exist_ok = True)
      path = os.path.join(self.filepath, f'{dir_name}')
      if self.save_callback is None:
        self.model.save(path)
        if self.input_signature is not None:
          onnx_model, _ = tf2onnx.convert.from_keras(self.model, self.input_signature, opset=13)
          onnx.save(onnx_model, os.path.join(self.filepath, f"{onnx_dir_name}"))

      else:
        self.save_callback(self.model, path)
      path_best = os.path.join(self.filepath, 'best.onnx')
      path_best_keras = os.path.join(self.filepath, 'best.keras')
      if os.path.exists(path_best):
        os.remove(path_best)
        os.remove(path_best_keras)

      os.symlink(onnx_dir_name, path_best)
      os.symlink(dir_name, path_best_keras)

      if self.verbose > 0:
        self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
        if self.best is None:
          self.msg += f"= {current:.5f}."
        else:
          self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
        self.msg += f" Saving model to {path}\n"
      self.best = current
      self.epochs_since_last_save = 0
    if self.patience is not None and self.epochs_since_last_save >= self.patience:
      self.model.stop_training = True
      if self.verbose > 0:
        if self.msg is None:
          self.msg = '\n'
        self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."






class AdversarialModel(tf.keras.Model):
  '''Goal: discriminate class0 vs class1 vs class2 without learning features that can guess class_adv'''

  def __init__(self, setup, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setup = setup

    self.epoch_counter = tf.Variable(0.)

    self.adv_optimizer = tf.keras.optimizers.Nadam(
        learning_rate=setup['adv_learning_rate'],
        weight_decay=setup['adv_weight_decay']
    )

    self.apply_common_gradients = setup['apply_common_gradients']

    self.class_grad_factor = setup['class_grad_factor']

    self.class_loss = categorical_crossentropy
    self.class_accuracy = categorical_accuracy

    self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
    self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

    self.adv_grad_factor = setup['adv_grad_factor']

    self.adv_loss = binary_focal_crossentropy
    self.adv_accuracy = accuracy

    self.adv_loss_tracker = tf.keras.metrics.Mean(name="adv_loss")
    self.adv_accuracy_tracker = tf.keras.metrics.Mean(name="adv_accuracy")

    self.adv_loss_tracker_submodule = tf.keras.metrics.Mean(name="adv_loss")
    self.adv_accuracy_tracker_submodule = tf.keras.metrics.Mean(name="adv_accuracy")


    self.common_layers = []

    def add_layer(layer_list, n_units, activation, name):
      layer = tf.keras.layers.Dense(n_units, activation=activation, name=name)
      layer_list.append(layer)
      if setup['dropout'] > 0:
        dropout = tf.keras.layers.Dropout(setup['dropout'], name=name + '_dropout')
        layer_list.append(dropout)
      if setup['use_batch_norm']:
        batch_norm = tf.keras.layers.BatchNormalization(name=name + '_batch_norm')
        layer_list.append(batch_norm)

    for n in range(setup['n_common_layers']):
      add_layer(self.common_layers, setup['n_common_units'], setup['common_activation'], f'common_{n}')

    self.class_layers = []
    self.adv_layers = []
    for n in range(setup['n_class_layers']):
      add_layer(self.class_layers, setup['n_class_units'], setup['class_activation'], f'class_{n}')
    for n in range(setup['n_adv_layers']):
      add_layer(self.adv_layers, setup['n_adv_units'], setup['adv_activation'], f'adv_{n}')


    self.class_output = tf.keras.layers.Dense(setup['nClasses'], activation='softmax', name='class_output')

    self.adv_output = tf.keras.layers.Dense(1, activation='sigmoid', name='adv_output')

    self.output_names = ['class_output', 'adv_output']

  def call(self, x):
    x_common = self.call_common(x)
    class_output = self.call_class(x_common)
    adv_output = self.call_adv(x_common)
    return class_output, adv_output

  def call_common(self, x):
    for layer in self.common_layers:
      x = layer(x)
    return x

  def call_class(self, x_common):
    x = x_common
    for layer in self.class_layers:
      x = layer(x)
    class_output = self.class_output(x)
    return class_output

  def call_adv(self, x_common):
    x = x_common
    for layer in self.adv_layers:
      x = layer(x)
    adv_output = self.adv_output(x)
    return adv_output

  def _step(self, data, training):
    x, y = data

    y_class = tf.cast(y[0], dtype=tf.float32)
    y_adv = tf.cast(y[1], dtype=tf.float32)

    class_weight = tf.cast(y[2], dtype=tf.float32)
    adv_weight = tf.cast(y[3], dtype=tf.float32)
    

    def compute_losses():
      y_pred_class, y_pred_adv = self(x, training=training)

      class_loss_vec = self.class_loss(y_class, y_pred_class)

      class_loss = tf.reduce_mean(class_loss_vec * class_weight)

      adv_loss_vec = self.adv_loss(y_adv, y_pred_adv, y_class, y_pred_class) # Focal loss
      # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
      # We want to apply some weights onto the adv loss vector
      # This is to have the SignalRegion and ControlRegion have equal weights

      adv_loss = tf.reduce_mean(adv_loss_vec * adv_weight)

      # Experimental ks test loss
      # Combine both class and adv loss into one 'loss' and put into only one optimizer
      # new_loss = class_lost + k * ks_test

      # y_adv_SR_mask = (y_adv == 0) & (adv_weight != 0)
      # y_adv_CR_mask = (y_adv == 1) & (adv_weight != 0)
      # k = 0.0

      # new_loss = class_loss + k * ks_test(y_pred_class[y_adv_SR_mask], y_pred_class[y_adv_CR_mask])

      return y_pred_class, class_loss_vec, class_loss, y_pred_adv, adv_loss_vec, adv_loss

    if training:
      with tf.GradientTape() as class_tape, tf.GradientTape() as adv_tape:
        y_pred_class, class_loss_vec, class_loss, y_pred_adv, adv_loss_vec, adv_loss = compute_losses()
    else:
      y_pred_class, class_loss_vec, class_loss, y_pred_adv, adv_loss_vec, adv_loss = compute_losses()

    class_accuracy_vec = self.class_accuracy(y_class, y_pred_class)

    self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
    self.class_accuracy_tracker.update_state(class_accuracy_vec, sample_weight=class_weight)

    adv_accuracy_vec = self.adv_accuracy(y_adv, y_pred_adv)


    self.adv_loss_tracker.update_state(adv_loss_vec, sample_weight=adv_weight)
    self.adv_accuracy_tracker.update_state(adv_accuracy_vec, sample_weight=adv_weight)

    if training:
      common_vars = [ var for var in self.trainable_variables if "/common" in var.path ]
      class_vars = [ var for var in self.trainable_variables if "/class" in var.path ]
      adv_vars = [ var for var in self.trainable_variables if "/adv" in var.path ]
      n_common_vars = len(common_vars)


      grad_class = class_tape.gradient(class_loss, common_vars + class_vars)
      grad_class_excl = grad_class[n_common_vars:]

      grad_adv = adv_tape.gradient(adv_loss, common_vars + adv_vars)
      grad_adv_excl = grad_adv[n_common_vars:]

      grad_common = [ self.class_grad_factor * grad_class[i] - self.adv_grad_factor * grad_adv[i] \
                      for i in range(len(common_vars)) ]

      grad_common_no_adv = [ grad_class[i] \
                      for i in range(len(common_vars)) ]

      grad_common_only_adv = [ grad_adv[i] \
                      for i in range(len(common_vars)) ]

      @tf.function
      def cond_true_fn():
        if self.apply_common_gradients:
          tf.cond(
            self.epoch_counter == 0. and not self.setup['continue_training'],
            true_fn = apply_common_no_adv,
            false_fn = apply_common
          )
        return
      
      @tf.function
      def apply_common_no_adv():
        self.optimizer.apply_gradients(zip(grad_common_no_adv + grad_class_excl, common_vars + class_vars))
        return 
      
      @tf.function
      def apply_common():
        self.optimizer.apply_gradients(zip(grad_common + grad_class_excl, common_vars + class_vars))
        return 

      @tf.function
      def cond_false_fn():
        return

      cond_true_fn()
      self.adv_optimizer.apply_gradients(zip(grad_adv_excl, adv_vars))

    return { m.name: m.result() for m in self.metrics }




  def _step_adv_only(self, data, training):
    x, y = data

    y_adv = tf.cast(y[1], dtype=tf.float32)

    adv_weight = tf.cast(y[3], dtype=tf.float32)
    

    def compute_losses(x_common):
      y_pred_adv = self.call_adv(x_common)

      adv_loss_vec = self.adv_loss(y_adv, y_pred_adv, None, None) # Focal loss
      # adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
      # We want to apply some weights onto the adv loss vector
      # This is to have the SignalRegion and ControlRegion have equal weights

      adv_loss = tf.reduce_mean(adv_loss_vec * adv_weight)

      return y_pred_adv, adv_loss_vec, adv_loss

    if training:
      x_common = self.call_common(x)
      with tf.GradientTape() as adv_tape:
        y_pred_adv, adv_loss_vec, adv_loss = compute_losses(x_common)
    else:
      y_pred_adv, adv_loss_vec, adv_loss = compute_losses()


    adv_accuracy_vec = self.adv_accuracy(y_adv, y_pred_adv)

    self.adv_loss_tracker_submodule.update_state(adv_loss_vec, sample_weight=adv_weight)
    self.adv_accuracy_tracker_submodule.update_state(adv_accuracy_vec, sample_weight=adv_weight)

    if training:
      adv_vars = [ var for var in self.trainable_variables if "/adv" in var.path ]


      grad_adv = adv_tape.gradient(adv_loss, adv_vars)

      self.adv_optimizer.apply_gradients(zip(grad_adv, adv_vars))

    return

  def train_step(self, data):
    # self.batch_counter = self.batch_counter.assign_add(1)
    return self._step(data, training=True)

  def test_step(self, data):
    return self._step(data, training=False)

  @property
  def metrics(self):
    return [
          self.class_loss_tracker,
          self.class_accuracy_tracker,

          self.adv_loss_tracker,
          self.adv_accuracy_tracker,
    ]



def train_dnn(setup, training_file, weight_file, config_dict, test_training_file, test_weight_file, test_config_dict, output_file):
  batch_size = config_dict['meta_data']['batch_dict']['batch_size']
  test_batch_size = test_config_dict['meta_data']['batch_dict']['batch_size']

  output_dnn_name = output_file

  dw = DataWrapper()
  dw.AddInputFeatures(setup['features'])
  for list_feature in setup['listfeatures']:
     dw.AddInputFeaturesList(*list_feature)
  dw.AddHighLevelFeatures(setup['highlevelfeatures'])


  dw.UseParametric(setup['UseParametric'])
  dw.SetParamList(setup['parametric_list'])

  dw.SetMbbName('bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino')

  # Prep a test dw
  # Must copy before reading file so we can read the test file instead
  test_dw = copy.deepcopy(dw)

  entry_start = 0
  # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

  # Do you want to make a larger batch? May increase speed
  entry_stop = None


  dw.ReadFile(training_file, entry_start=entry_start, entry_stop=entry_stop)
  dw.ReadWeightFile(weight_file, entry_start=entry_start, entry_stop=entry_stop)
  print(config_dict)
  # dw.DefineTrainTestSet(batch_size, 0.0)


  test_dw.ReadFile(test_training_file, entry_start=entry_start, entry_stop=entry_stop)
  test_dw.ReadWeightFile(test_weight_file, entry_start=entry_start, entry_stop=entry_stop)
  # dw_val.DefineTrainTestSet(val_batch_size, 0.0)


  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  tf.random.set_seed(42)


  model = AdversarialModel(setup)
  model.compile(loss=None,
              # optimizer=tf.keras.optimizers.AdamW(learning_rate=setup['learning_rate'],
              #                                     weight_decay=setup['weight_decay']))
              optimizer=tf.keras.optimizers.Nadam(learning_rate=setup['learning_rate'],
                                                  weight_decay=setup['weight_decay']
              )
  )

  model(dw.features)

  model.summary()

  batch_size = setup['batch_compression_factor']*batch_size
  nClasses = setup['nClasses']
  train_tf_dataset = tf.data.Dataset.from_tensor_slices((dw.features, (tf.one_hot(dw.class_target, nClasses), dw.adv_target, dw.class_weight, dw.adv_weight))).batch(batch_size, drop_remainder=True)
  train_tf_dataset = train_tf_dataset.shuffle(len(train_tf_dataset), reshuffle_each_iteration=True)

  test_batch_size = setup['batch_compression_factor']*test_batch_size
  test_tf_dataset = tf.data.Dataset.from_tensor_slices((test_dw.features, (tf.one_hot(test_dw.class_target, nClasses), test_dw.adv_target, test_dw.class_weight, test_dw.adv_weight))).batch(test_batch_size, drop_remainder=True)
  test_tf_dataset = test_tf_dataset.shuffle(len(test_tf_dataset), reshuffle_each_iteration=True)



  @tf.function
  def new_param_map(*x):
    dataset = x
    features = dataset[0]

    # Need to randomize the features parametric mass
    parametric_mass_probability = np.ones(len(dw.param_list)) * 1.0/len(dw.param_list)
    random_param_mass = tf.random.categorical(tf.math.log([list(parametric_mass_probability)]), tf.shape(features)[0], dtype=tf.int64)

    mass_values = tf.constant(dw.param_list)
    mass_keys = tf.constant(np.arange(len(dw.param_list)))
    table = tf.lookup.StaticHashTable(
       tf.lookup.KeyValueTensorInitializer(mass_keys, mass_values),
       default_value = -1
    )

    actual_new_mass = table.lookup(random_param_mass)
    actual_new_mass = tf.cast(actual_new_mass, tf.float64)

    # Lastly we need to keep the signal events the correct mass
    class_targets = dataset[1][0]
    old_mass_mask = tf.cast(class_targets[:,0], tf.float64)
    new_mass_mask = tf.cast(class_targets[:,1], tf.float64)

    actual_mass = old_mass_mask * features[:,-1] + new_mass_mask * actual_new_mass
    actual_mass = tf.transpose(actual_mass)

    features = tf.concat([features[:,:-1], actual_mass], axis=-1)
    new_dataset = (features, dataset[1])
    return new_dataset

  train_tf_dataset = train_tf_dataset.map(new_param_map)
  test_tf_dataset = test_tf_dataset.map(new_param_map)

  def save_predicate(model, logs):
      return (abs(logs['val_adv_accuracy'] - 0.5) < 0.001) # How do we stop the model from always guessing 0.49 or 0.51?


  input_shape = [None, dw.features.shape[1]]
  input_signature = [tf.TensorSpec(input_shape, tf.double, name='x')]
  callbacks = [
      ModelCheckpoint(output_dnn_name, verbose=1, monitor="val_class_loss", mode='min', min_rel_delta=1e-3,
                      # patience=setup['patience'], save_callback=None, predicate=save_predicate, input_signature=input_signature),
                      patience=setup['patience'], save_callback=None, input_signature=input_signature),
      tf.keras.callbacks.CSVLogger(f'{output_dnn_name}_training_log.csv', append=True),
      EpochCounterCallback(),
      AdvOnlyCallback(train_tf_dataset, nSteps=setup['adv_submodule_steps'], TrackerWindowSize=setup['adv_submodule_tracker'], on_batch=True, on_epoch=False, continue_training=setup['continue_training'], quiet=False),
      # AdvOnlyCallback(train_tf_dataset, nSteps=5000, TrackerWindowSize=100, on_batch=False, on_epoch=True, skip_epoch0=False, quiet=False),
  ]
  print("Fit model")
  history = model.fit(
      train_tf_dataset,
      validation_data=test_tf_dataset,
      verbose=1,
      epochs=setup['n_epochs'],
      shuffle=False,
      callbacks=callbacks,
  )

  model.save(f"{output_dnn_name}.keras")

  input_shape = [None, dw.features.shape[1]]
  input_signature = [tf.TensorSpec(input_shape, tf.double, name='x')]
  onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
  onnx.save(onnx_model, f"{output_dnn_name}.onnx")


  return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create TrainTest Files for DNN.')
    parser.add_argument('--training_file', required=True, type=str, help="Training file")
    parser.add_argument('--weight_file', required=True, type=str, help="Weight file")
    parser.add_argument('--batch_config', required=True, type=str, help="Batch config file")
    parser.add_argument('--test_training_file', required=True, type=str, help="Test file")
    parser.add_argument('--test_weight_file', required=True, type=str, help="Test weight file")
    parser.add_argument('--test_batch_config', required=True, type=str, help="Test batch config file")
    parser.add_argument('--output_file', required=True, type=str, help="Output model")
    parser.add_argument('--setup-config', required=True, type=str, help='Setup config for training')

    args = parser.parse_args()

    setup = {}
    with open(args.setup_config, 'r') as file:
        setup = yaml.safe_load(file)  

    modelname_parity = []

    try:
       
        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        config_dict = {}
        with open(args.batch_config, 'r') as file:
            config_dict = yaml.safe_load(file)  
        test_config_dict = {}
        with open(args.test_batch_config, 'r') as file:
            test_config_dict = yaml.safe_load(file)

        model = train_dnn(setup, args.training_file, args.weight_file, config_dict, args.test_training_file, args.test_weight_file, test_config_dict, args.output_file)


    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
