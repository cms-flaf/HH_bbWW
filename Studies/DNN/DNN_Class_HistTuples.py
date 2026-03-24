import tensorflow as tf
import numpy as np
import uproot
import os
import yaml
import tf2onnx
import onnx
import copy
import psutil
import matplotlib.pyplot as plt
import onnxruntime as ort
import ROOT
import sklearn.metrics
import shutil


class DataWrapper:
    def __init__(self):
        print("Init data wrapper")

        self.feature_names = None

        self.features_no_param = None
        self.features = None

        self.param_values = None

        self.labels = None

        self.class_weight = None
        self.class_target = None

        self.res2b = None
        self.recovery = None
        self.boosted = None

        self.param_list = [
            250,
            260,
            270,
            280,
            300,
            350,
            450,
            550,
            600,
            650,
            700,
            800,
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2500,
            3000,
            4000,
            5000,
        ]
        self.use_parametric = False

        self.features_paramSet = None

        self.X_mass = None

    def UseParametric(self, use_parametric):
        self.use_parametric = use_parametric
        print(f"Parametric feature set to {use_parametric}")

    def SetParamList(self, param_list):
        self.param_list = param_list

    def SetPredictParamValue(self, param_value):
        # During predict, we want to use a truly random param value even for signal!
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

    def ReadFile(self, file_name, entry_start=None, entry_stop=None):
        if self.feature_names == None:
            print("Unknown branches to read! DefineInputFeatures first!")
            return

        print(f"Reading file {file_name}")

        features_to_load = self.feature_names.copy()

        features_to_load.append("X_mass")
        features_to_load.append("weight_Central")
        features_to_load.append("class_value")
        features_to_load.append("res2b")
        features_to_load.append("recovery")
        features_to_load.append("boosted")

        print(f"Only loading these features {features_to_load}")

        print(
            f"Going to open file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

        # file = uproot.open(file_name)
        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                features_to_load, entry_start=entry_start, entry_stop=entry_stop
            )

            print(
                f"Loaded branches. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            self.features = np.array(
                [
                    getattr(branches, feature_name)
                    for feature_name in self.feature_names
                ],
                dtype="float32",
            ).transpose()

            print(
                f"Set Features. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            # Add parametric variable
            # self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
            self.X_mass = getattr(branches, "X_mass")
            self.physics_weight = getattr(branches, "weight_Central")
            self.class_value = getattr(branches, "class_value")
            self.res2b = getattr(branches, "res2b")
            self.recovery = getattr(branches, "recovery")
            self.boosted = getattr(branches, "boosted")
            self.param_values = np.array(
                [getattr(branches, "X_mass")], dtype="float32"
            ).transpose()  # Init wrong parametric, later we will fill with random sample
            print("Got the param values")

        self.features_no_param = self.features
        if self.use_parametric:
            self.features = np.append(self.features, self.param_values, axis=1)

        print(
            f"End read. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

    def ReadWeightFile(self, weight_name, entry_start=None, entry_stop=None):
        print(f"Reading weight file {weight_name}")
        with uproot.open(weight_name) as file:
            tree = file["weight_tree"]
            branches = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
            self.class_weight = np.array(
                getattr(branches, "class_weight"), dtype="float32"
            )
            self.class_target = np.array(
                getattr(branches, "class_target"), dtype="float32"
            )
            file.close()


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        mode="min",
        min_delta=None,
        min_rel_delta=None,
        save_callback=None,
        patience=None,
        predicate=None,
        input_signature=None,
    ):
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
                return (
                    lambda current, best: best is None
                    or (best - current) > min_rel_delta * best
                )
            if min_rel_delta is None:
                return lambda current, best: best is None or best - current > min_delta
            return (
                lambda current, best: best is None
                or (best - current) > min_rel_delta * best
                or best - current > min_delta
            )
        elif mode == "max":
            if min_delta is None and min_rel_delta is None:
                return lambda current, best: best is None or current - best > 0
            if min_delta is None:
                return (
                    lambda current, best: best is None
                    or (current - best) > min_rel_delta * best
                )
            if min_rel_delta is None:
                return lambda current, best: best is None or current - best > min_delta
            return (
                lambda current, best: best is None
                or (current - best) > min_rel_delta * best
                or current - best > min_delta
            )
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

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best) and (
            self.predicate is None or self.predicate(self.model, logs)
        ):
            # if self.predicate is None or self.predicate(self.model, logs):
            os.makedirs(self.filepath, exist_ok=True)
            path_best = os.path.join(self.filepath, f"{self.monitor}.onnx")
            if os.path.exists(path_best):
                os.remove(path_best)
            if self.input_signature is not None:
                onnx_model, _ = tf2onnx.convert.from_keras(
                    self.model, self.input_signature, opset=13
                )
                onnx.save(onnx_model, path_best)

            if self.verbose > 0:
                self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
                if self.best is None:
                    self.msg += f"= {current:.5f}."
                else:
                    self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
                self.msg += f" Saving model to {path_best}\n"
            self.best = current
            self.epochs_since_last_save = 0
        if self.patience is not None and self.epochs_since_last_save >= self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                if self.msg is None:
                    self.msg = "\n"
                self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."


class WeightedBackgroundAtSignalYield(tf.keras.metrics.Metric):
    def __init__(
        self,
        threshold_yield,
        max_events=100000,
        nParity=4,
        name="weighted_bkg_at_sig_yield",
        dtype=tf.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.threshold_yield = tf.constant(threshold_yield, dtype=dtype)
        self.max_events = max_events
        self.nParity = nParity

        self.scores = self.add_weight(
            name="scores", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.labels = self.add_weight(
            name="labels", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.weights_var = self.add_weight(
            name="weights", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.count = self.add_weight(
            name="count", shape=(), dtype=tf.int32, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        else:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        batch_size = tf.shape(y_true)[0]
        start = self.count
        end = tf.minimum(start + batch_size, self.max_events)
        actual_batch = end - start

        y_true = y_true[:actual_batch]
        y_pred = y_pred[:actual_batch]
        sample_weight = sample_weight[:actual_batch]

        indices = tf.range(start, end)
        indices_2d = tf.expand_dims(indices, 1)

        self.scores.assign(tf.tensor_scatter_nd_update(self.scores, indices_2d, y_pred))
        self.labels.assign(tf.tensor_scatter_nd_update(self.labels, indices_2d, y_true))
        self.weights_var.assign(
            tf.tensor_scatter_nd_update(self.weights_var, indices_2d, sample_weight)
        )

        self.count.assign(end)

    def result(self):
        n = tf.cast(self.count, tf.int32)  # ensure int32 for indexing

        def no_samples():
            return (
                tf.constant(-2.0, dtype=self.dtype),
                tf.constant(-2.0, dtype=self.dtype),
                tf.constant(-2.0, dtype=self.dtype),
            )

        def compute():
            # Safe slicing useing tf.gather instead of tf.slice
            idx = tf.range(n)
            scores = tf.gather(self.scores, idx)
            labels = tf.gather(self.labels, idx)
            weights = tf.gather(self.weights_var, idx)

            sorted_idx = tf.argsort(scores, direction="DESCENDING")
            sorted_labels = tf.gather(labels, sorted_idx)
            sorted_weights = tf.gather(weights, sorted_idx)

            is_signal = tf.cast(tf.equal(sorted_labels, 1.0), tf.float32)
            is_bkg = 1.0 - is_signal

            sig_weights = sorted_weights * is_signal
            bkg_weights = sorted_weights * is_bkg

            cum_sig = tf.cumsum(sig_weights)
            cum_bkg = tf.cumsum(bkg_weights)
            cum_bkg_w2 = tf.cumsum(bkg_weights * bkg_weights)

            # Scale by nParity to 'fake' realistic values
            cum_sig = (
                self.nParity * 0.0264215349425664 * cum_sig
            )  # Scale signal to BR too
            cum_bkg = self.nParity * cum_bkg
            cum_bkg_w2 = self.nParity * self.nParity * cum_bkg_w2

            total_sig = cum_sig[-1]
            threshold = tf.minimum(self.threshold_yield, total_sig)

            def no_signal():
                return (
                    tf.constant(-1.0, dtype=self.dtype),
                    tf.constant(-1.0, dtype=self.dtype),
                    tf.constant(-1.0, dtype=self.dtype),
                )

            def with_signal():
                mask = cum_sig >= threshold
                idx = tf.argmax(tf.cast(mask, tf.int32))
                # tf.print("DNN score cut ", scores[idx])
                # tf.print("Cum bkg ", cum_bkg[idx])
                # tf.print("Cum bkg w2 ", cum_bkg_w2[idx])
                # tf.print("Cum sig ", cum_sig[idx])
                return (
                    tf.cast(cum_bkg[idx], self.dtype),
                    tf.keras.ops.power(tf.cast(cum_bkg_w2[idx], self.dtype), 0.5),
                    tf.cast(scores[idx], self.dtype),
                )

            return tf.cond(
                tf.logical_or(tf.equal(total_sig, 0.0), tf.equal(threshold, 0.0)),
                no_signal,
                with_signal,
            )

        value, error, score = tf.cond(tf.equal(n, 0), no_samples, compute)

        return value, error, score

    def reset_state(self):
        self.scores.assign(tf.zeros_like(self.scores))
        self.labels.assign(tf.zeros_like(self.labels))
        self.weights_var.assign(tf.zeros_like(self.weights_var))
        self.count.assign(0)

    def _compute(self):
        return self.result()


class WeightedBackgroundAtSignalYieldValue(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_value", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        # raise RuntimeError("Dummy but second one")
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        value, _, _ = self.parent._compute()
        return tf.cast(value, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


class WeightedBackgroundAtSignalYieldError(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_error", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        # raise RuntimeError("Dummy")
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        _, error, _ = self.parent._compute()
        return tf.cast(error, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


class WeightedBackgroundAtSignalYieldScore(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_score", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        # raise RuntimeError("Dummy")
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        _, _, score = self.parent._compute()
        return tf.cast(score, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


@tf.function
def binary_entropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    x = tf.clip_by_value(output, epsilon, 1 - epsilon)
    return -target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)


@tf.function
def binary_focal_crossentropy(target, output, gamma1=2, gamma2=0.5):
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Un-nest the output (currently in shape [ [1], [2], [3], ...] and we want in shape [1, 2, 3])
    y_true = target[:, 0]
    y_pred = output[:, 0]

    bce = binary_entropy(y_true, y_pred)

    # Custom focal
    gamma_signal = 0  # DO NOT TOUCH
    gamma_bkg = gamma1
    # Target y_pred -> 1
    p_t = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    # Split into a 'signal' gamma and a 'bkg' gamma
    gamma = y_true * gamma_signal + (1 - y_true) * gamma_bkg
    # Calc factor of pT^gamma
    focal_factor = tf.keras.ops.power(p_t, gamma)

    # focal_factor = 1.0
    # focal_bce = focal_factor * bce

    gamma2_signal = gamma2
    gamma2_bkg = 1  # DO NOT TOUCH

    gamma2 = y_true * gamma2_signal + (1 - y_true) * gamma2_bkg

    focal_bce = focal_factor * tf.keras.ops.power(bce, gamma2)

    return focal_bce


class Model(tf.keras.Model):
    def __init__(self, setup, max_events, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = setup
        self.gamma1 = setup["gamma1"]
        self.gamma2 = setup["gamma2"]

        self.nClasses = setup["nClasses"]

        # self.class_loss = tf.keras.losses.categorical_crossentropy
        # self.class_loss = tf.keras.losses.categorical_focal_crossentropy
        # self.class_loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=5.0, reduction=None)
        self.class_loss = binary_focal_crossentropy

        self.class_accuracy = tf.keras.metrics.categorical_accuracy

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

        self.l2_loss_tracker = tf.keras.metrics.Mean(name="l2_loss")

        self.bkgAtSignal = WeightedBackgroundAtSignalYield(
            threshold_yield=5.0, max_events=max_events
        )
        self.bkgAtSignal_value = WeightedBackgroundAtSignalYieldValue(self.bkgAtSignal)
        self.bkgAtSignal_error = WeightedBackgroundAtSignalYieldError(self.bkgAtSignal)
        self.bkgAtSignal_score = WeightedBackgroundAtSignalYieldScore(self.bkgAtSignal)

        self.class_min_tracker = tf.keras.metrics.Mean(name="class_min")
        self.class_max_tracker = tf.keras.metrics.Mean(name="class_max")

        self.lr_tracker = tf.keras.metrics.Mean(name="learning_rate")

        self.other_class_min_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_min{n}")
            for n in range(self.nClasses)
            if n != 0
        ]
        self.other_class_max_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_max{n}")
            for n in range(self.nClasses)
            if n != 0
        ]

        self.class_layers = []

        def add_layer(layer_list, n_units, activation, name):
            if setup["use_batch_norm"]:
                batch_norm = tf.keras.layers.BatchNormalization(
                    name=name + "_batch_norm"
                )
                layer_list.append(batch_norm)

            layer = tf.keras.layers.Dense(
                n_units,
                activation=activation,
                name=name,
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
                kernel_regularizer=tf.keras.regularizers.l2(setup["l2_rate"]),
                # kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00000001, l2=0.00000001)
            )
            layer_list.append(layer)

            if setup["dropout"] > 0:
                dropout = tf.keras.layers.Dropout(
                    setup["dropout"], name=name + "_dropout"
                )
                layer_list.append(dropout)

        for n in range(setup["n_layers"]):
            add_layer(
                self.class_layers,
                setup["n_units"],
                setup["activation"],
                f"layer_{n}",
            )

        self.class_output = tf.keras.layers.Dense(
            setup["nClasses"], activation="softmax", name="class_output"
        )

        self.output_names = ["class_output"]

    def call(self, x):
        for layer in self.class_layers:
            x = layer(x)
        class_output = self.class_output(x)
        return class_output

    def _step(self, data, training):
        x, y = data

        y_class = tf.cast(y[0], dtype=tf.float32)

        class_weight = tf.cast(y[1], dtype=tf.float32)

        physics_weight = tf.cast(y[2], dtype=tf.float32)

        def compute_losses():
            y_pred_class = self(x, training=training)

            class_loss_vec = self.class_loss(
                y_class, y_pred_class, self.gamma1, self.gamma2
            )

            class_loss = tf.reduce_mean(class_loss_vec * class_weight)

            l2_loss = tf.add_n(self.losses)

            combined_loss = class_loss + l2_loss

            return y_pred_class, class_loss_vec, class_loss, l2_loss, combined_loss

        if training:
            with tf.GradientTape() as class_tape:
                y_pred_class, class_loss_vec, class_loss, l2_loss, combined_loss = (
                    compute_losses()
                )
        else:
            y_pred_class, class_loss_vec, class_loss, l2_loss, combined_loss = (
                compute_losses()
            )

        self.class_min_tracker.update_state(tf.reduce_min(y_pred_class[:, 0]))
        self.class_max_tracker.update_state(tf.reduce_max(y_pred_class[:, 0]))

        self.bkgAtSignal.update_state(
            y_class[:, 0], y_pred_class[:, 0], sample_weight=physics_weight
        )

        for n in range(self.nClasses):
            if n == 0:
                continue
            self.other_class_min_tracker[n - 1].update_state(
                tf.reduce_min(y_pred_class[:, n])
            )
            self.other_class_max_tracker[n - 1].update_state(
                tf.reduce_max(y_pred_class[:, n])
            )

        class_accuracy_vec = self.class_accuracy(y_class, y_pred_class)

        self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
        self.class_accuracy_tracker.update_state(
            class_accuracy_vec, sample_weight=class_weight
        )

        self.lr_tracker.update_state(self.optimizer.learning_rate)

        self.l2_loss_tracker.update_state(l2_loss)

        if training:
            grad = class_tape.gradient(combined_loss, self.trainable_variables)
            # grad = class_tape.gradient(class_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)

    @property
    def metrics(self):
        metric_list = [
            self.class_loss_tracker,
            self.class_accuracy_tracker,
            self.l2_loss_tracker,
            self.bkgAtSignal_value,
            self.bkgAtSignal_error,
            self.bkgAtSignal_score,
            self.class_min_tracker,
            self.class_max_tracker,
            self.lr_tracker,
        ]
        metric_list = (
            metric_list
            + [x for x in self.other_class_min_tracker]
            + [x for x in self.other_class_max_tracker]
        )

        return metric_list


def train_dnn(
    setup,
    training_file,
    weight_file,
    test_training_file,
    test_weight_file,
    output_folder,
):
    output_dnn_name = os.path.join(output_folder, f"best.onnx")

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = None

    dw.ReadFile(
        training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    dw.ReadWeightFile(weight_file, entry_start=entry_start, entry_stop=entry_stop)

    test_dw.ReadFile(
        test_training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    test_dw.ReadWeightFile(
        test_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )

    dw.physics_weight = np.where(
        (dw.class_target == 0) & (dw.X_mass != 600), 0.0, dw.physics_weight
    )
    test_dw.physics_weight = np.where(
        (test_dw.class_target == 0) & (test_dw.X_mass != 600),
        0.0,
        test_dw.physics_weight,
    )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    nClasses = setup["nClasses"]
    batch_size = setup["batch_size"]
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dw.features,
            (
                tf.one_hot(dw.class_target, nClasses),
                dw.class_weight,
                dw.physics_weight,
            ),
        )
    )
    train_tf_dataset = train_tf_dataset.shuffle(
        len(train_tf_dataset), reshuffle_each_iteration=True
    )
    batch_size_train = min(batch_size, train_tf_dataset.cardinality().numpy())
    train_tf_dataset = train_tf_dataset.batch(batch_size_train, drop_remainder=True)

    test_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            test_dw.features,
            (
                tf.one_hot(test_dw.class_target, nClasses),
                test_dw.class_weight,
                test_dw.physics_weight,
            ),
        )
    )
    test_tf_dataset = test_tf_dataset.shuffle(
        len(test_tf_dataset), reshuffle_each_iteration=True
    )
    batch_size_test = min(batch_size, test_tf_dataset.cardinality().numpy())
    test_tf_dataset = test_tf_dataset.batch(batch_size_test, drop_remainder=True)

    @tf.function
    def new_param_map(*x):
        dataset = x
        features = dataset[0]

        # Need to randomize the features parametric mass
        parametric_mass_probability = (
            np.ones(len(dw.param_list)) * 1.0 / len(dw.param_list)
        )
        random_param_mass = tf.random.categorical(
            tf.math.log([list(parametric_mass_probability)]),
            tf.shape(features)[0],
            dtype=tf.int64,
        )

        mass_values = tf.constant(dw.param_list)
        mass_keys = tf.constant(np.arange(len(dw.param_list)))
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(mass_keys, mass_values),
            default_value=-1,
        )

        actual_new_mass = table.lookup(random_param_mass)
        actual_new_mass = tf.cast(actual_new_mass, tf.float32)

        # Lastly we need to keep the signal events the correct mass
        class_targets = dataset[1][0]
        old_mass_mask = tf.cast(class_targets[:, 0], tf.float32)
        new_mass_mask = tf.cast((class_targets[:, 0] == 0), tf.float32)

        actual_mass = old_mass_mask * features[:, -1] + new_mass_mask * actual_new_mass
        actual_mass = tf.transpose(actual_mass)

        features = tf.concat([features[:, :-1], actual_mass], axis=-1)
        new_dataset = (features, dataset[1])
        return new_dataset

    if setup["UseParametric"]:
        train_tf_dataset = train_tf_dataset.map(new_param_map)
        test_tf_dataset = test_tf_dataset.map(new_param_map)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.double, name="x")]

    nBatches = max(
        train_tf_dataset.cardinality().numpy(), test_tf_dataset.cardinality().numpy()
    )
    max_events = nBatches * max(batch_size_train, batch_size_test)
    model = Model(setup, max_events)
    model.compile(
        loss=None,
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=setup["learning_rate"],
            weight_decay=setup["weight_decay"],
            clipnorm=1.0,
        ),
    )
    model(dw.features)
    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_class_loss",
        factor=setup["lr_decay"],
        patience=setup["lr_patience"],
        min_lr=0.0000000001,
    )

    callbacks = [
        ModelCheckpoint(
            output_folder,
            verbose=1,
            monitor="val_class_loss",
            mode="min",
            min_rel_delta=1e-3,
            patience=setup["patience"],
            save_callback=None,
            input_signature=input_signature,
        ),
        ModelCheckpoint(
            output_folder,
            verbose=1,
            monitor="val_weighted_bkg_at_sig_yield_value",
            mode="min",
            min_rel_delta=1e-3,
            patience=setup["patience"],
            save_callback=None,
            input_signature=input_signature,
        ),
        reduce_lr,
    ]

    verbose = setup["verbose"] if "verbose" in setup else 0
    # verbose = 1
    print("Fit model")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        verbose=verbose,
        epochs=setup["n_epochs"],
        shuffle=True,
        callbacks=callbacks,
    )

    os.makedirs(output_folder, exist_ok=True)

    def PlotMetric(history, metric, output_folder):
        if metric not in history.history:
            print(f"Metric {metric} not found in history")
            return
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"{metric}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.yscale("log")
        plt.ylim(bottom=0.0001)
        plt.savefig(os.path.join(output_folder, f"{metric}.pdf"), bbox_inches="tight")
        plt.clf()

    PlotMetric(history, "class_loss", output_folder)

    PlotMetric(history, "l2_loss", output_folder)

    PlotMetric(history, "class_min", output_folder)

    PlotMetric(history, "class_max", output_folder)

    PlotMetric(history, "weighted_bkg_at_sig_yield_value", output_folder)
    PlotMetric(history, "weighted_bkg_at_sig_yield_error", output_folder)
    PlotMetric(history, "weighted_bkg_at_sig_yield_score", output_folder)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.double, name="x")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, output_dnn_name)

    features_config = {
        "features": dw.feature_names,
        "use_parametric": dw.use_parametric,
        "parametric_list": dw.param_list,
        "model_setup": setup,
        "nClasses": setup["nClasses"],
        "nParity": 4,
    }

    with open(os.path.join(output_folder, "dnn_config.yaml"), "w") as file:
        yaml.dump(features_config, file)

    return


def validate_dnn(
    setup,
    validation_file,
    validation_weight_file,
    output_folder,
    model_name,
    model_config,
):
    print(f"Model load {model_name}")
    sess = ort.InferenceSession(model_name)

    dnnConfig = {}
    with open(model_config, "r") as file:
        dnnConfig = yaml.safe_load(file)

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = None

    dw.ReadFile(
        validation_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    dw.ReadWeightFile(
        validation_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    nClasses = setup["nClasses"]

    os.makedirs(output_folder, exist_ok=True)

    for cat in ["res2b", "boosted", "res1b"]:
        ROOTOut = ROOT.TFile(
            os.path.join(output_folder, f"validation_{cat}.root"), "RECREATE"
        )
        output_file = os.path.join(output_folder, f"validation_{cat}.pdf")
        fig, ax = plt.subplots()

        para_masspoint_list = [
            300,
            400,
            500,
            550,
            600,
            650,
            700,
            800,
            900,
            1000,
            2000,
            3000,
        ]  # [300, 450, 800]
        para_masspoint_list = [300, 600, 1000]
        canvases = []
        for para_masspoint in para_masspoint_list:
            print(f"Validating mass {para_masspoint}")
            if dw.use_parametric:
                dw.SetPredictParamValue(para_masspoint)
            features = (
                dw.features_paramSet if dw.use_parametric else dw.features_no_param
            )

            # print("Predicting")
            # print("Using features")
            # print(features)
            pred = sess.run(None, {"x": features})
            pred_class = pred[0]
            pred_signal = pred_class[:, 0]

            class_weight = dw.class_weight
            physics_weight = dw.physics_weight

            # Scale signal to BR
            physics_weight = np.where(
                dw.class_target == 0,
                dw.physics_weight * 0.0264215349425664,
                dw.physics_weight,
            )

            # Only keep res2b for now and scale by 4 for parity
            # physics_weight = np.where(dw.res2b == 1, 4*physics_weight, 0.0)
            # physics_weight = np.where(dw.boosted == 1, 4*physics_weight, 0.0)

            if cat == "res2b":
                physics_weight = np.where(dw.res2b == 1, physics_weight, 0.0)
            if cat == "boosted":
                physics_weight = np.where(dw.boosted == 1, physics_weight, 0.0)
            if cat == "res1b":
                physics_weight = np.where(dw.recovery == 1, physics_weight, 0.0)

            # Class Plots
            # Lets build Masks
            Sig_This_Mass = dw.X_mass == para_masspoint
            Sig_mask = (Sig_This_Mass) & (dw.class_target == 0)

            Background_mask = dw.class_target == 1

            TT_mask = dw.class_value == 1

            DY_mask = dw.class_value == 2

            Other_mask = dw.class_value == 3

            # Set class quantiles based on signal
            nQuantBins = 50
            quant_binning_class = np.zeros(
                nQuantBins + 1
            )  # Need +1 because 10 bins actually have 11 edges
            if len(pred_signal[Sig_mask]) == 0:
                print("No signal events in this mass point! Fake Quant Bins!")
                quant_binning_class = np.linspace(0, 1, nQuantBins + 1)
            else:
                quant_binning_class = np.quantile(
                    pred_signal[Sig_mask], np.linspace(0, 1, nQuantBins + 1)
                )
            quant_binning_class[0] = 0.0
            quant_binning_class[-1] = 1.0
            print("We found quant binning class")
            print(quant_binning_class)
            # print("From the signal prediction")
            # print(pred_signal[Sig_mask])

            mask_dict = {
                "Signal": Sig_mask,
                "TT": TT_mask,
                "DY": DY_mask,
                "Other": Other_mask,
            }

            canvases.append(ROOT.TCanvas("c1", "c1", 1200, 600 * len(mask_dict.keys())))
            canvas = canvases[-1]
            canvas.Divide(1, len(mask_dict.keys()))
            Class_list = []
            legend_list = []
            pads_list = []
            for i, process_name in enumerate(mask_dict.keys()):
                canvas.cd(i + 1)
                mask = mask_dict[process_name]

                class_out_hist, bins = np.histogram(
                    pred_signal[mask],
                    bins=quant_binning_class,
                    range=(0.0, 1.0),
                    weights=physics_weight[mask],
                )
                class_out_hist_w2, bins = np.histogram(
                    pred_signal[mask],
                    bins=quant_binning_class,
                    range=(0.0, 1.0),
                    weights=physics_weight[mask] ** 2,
                )

                Class_list.append(
                    ROOT.TH1D(
                        f"ClassOutput_{process_name}",
                        f"ClassOutput_{process_name}",
                        nQuantBins,
                        0.0,
                        1.0,
                    )
                )

                ROOT_ClassOutput = Class_list[-1]

                for binnum in range(nQuantBins):
                    ROOT_ClassOutput.SetBinContent(binnum + 1, class_out_hist[binnum])
                    ROOT_ClassOutput.SetBinError(
                        binnum + 1, class_out_hist_w2[binnum] ** (0.5)
                    )

                ROOTOut.WriteObject(
                    ROOT_ClassOutput, f"m{para_masspoint}_{process_name}"
                )

                if ROOT_ClassOutput.Integral() == 0:
                    print(
                        f"Process {process_name} has no class entries, maybe the background doesn't exist?"
                    )
                    continue

                # ROOT_ClassOutput.Scale(1.0 / ROOT_ClassOutput.Integral())

                pads_list.append(ROOT.TPad("p1", "p1", 0.0, 0.3, 1.0, 0.9, 0, 0, 0))
                p1 = pads_list[-1]
                p1.SetTopMargin(0)
                p1.Draw()

                p1.cd()

                plotlabel = f"Class Output for {process_name} ParaMass {para_masspoint} GeV {cat}"
                ROOT_ClassOutput.Draw()
                ROOT_ClassOutput.SetTitle(plotlabel)
                ROOT_ClassOutput.SetStats(0)
                min_val = max(
                    0.0001,
                    ROOT_ClassOutput.GetMinimum(),
                )
                max_val = ROOT_ClassOutput.GetMaximum()

                ROOT_ClassOutput.GetYaxis().SetRangeUser(
                    0.001 * min_val, 1000 * max_val
                )

                legend_list.append(ROOT.TLegend(0.5, 0.8, 0.9, 0.9))
                legend = legend_list[-1]
                legend.AddEntry(ROOT_ClassOutput, f"{process_name}")
                legend.Draw()

                print(f"Setting canvas to log scale with range {min_val}, {max_val}")
                p1.SetLogy()
                p1.SetGrid()

            # if para_masspoint == para_masspoint_list[0]:
            #     canvas.Print(f"{output_file}(", f"Title:Mass {para_masspoint} GeV")
            #     print("Saved [")
            # elif para_masspoint == para_masspoint_list[-1]:
            #     canvas.Print(f"{output_file})", f"Title:Mass {para_masspoint} GeV")
            #     print("Saved ]")
            # else:
            #     canvas.Print(f"{output_file}", f"Title:Mass {para_masspoint} GeV")
            # print(f"Saved mass {para_masspoint}")

            canvas.Close()

            canvas = ROOT.TCanvas("c1", "c1", 800, 600)
            legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
            pad_class = ROOT.TPad("class", "class", 0.0, 0.3, 1.0, 0.9, 0, 0, 0)
            pad_soverb = ROOT.TPad("soverb", "soverb", 0.0, 0.1, 1.0, 0.3, 0, 0, 0)
            pad_class.SetTopMargin(0)
            pad_class.Draw()

            pad_soverb.SetTopMargin(0)
            pad_soverb.SetBottomMargin(0)
            pad_soverb.Draw()

            pad_class.cd()

            soverb = Class_list[0].Clone()
            background = Class_list[1].Clone()

            color_list = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta]
            marker_list = [105, 107, 108, 109]
            for i in range(len(Class_list)):
                if i == 0:
                    Class_list[i].Draw()
                else:
                    Class_list[i].Draw("same")
                if i > 1:
                    background.Add(Class_list[i].Clone())
                Class_list[i].SetLineColor(color_list[i])
                Class_list[i].SetMarkerStyle(marker_list[i])
                Class_list[i].SetMarkerColor(color_list[i])
                Class_list[i].SetMarkerSize(0.1)
                legend.AddEntry(Class_list[i], Class_list[i].GetTitle())

            Class_list[0].GetYaxis().SetRangeUser(0.0001, 10000)

            legend.Draw()
            # canvas.SetLogy()
            # canvas.SetGrid()

            pad_class.SetLogy()
            pad_class.SetGrid()

            pad_soverb.cd()

            soverb.SetTitle("S over B")
            soverb.Divide(background)
            soverb.Draw()

            pad_soverb.SetLogy()
            pad_soverb.SetGrid()

            if para_masspoint == para_masspoint_list[0]:
                canvas.Print(
                    f"{output_file}(", f"Title:Mass {para_masspoint} {cat} GeV"
                )
            elif para_masspoint == para_masspoint_list[-1]:
                canvas.Print(
                    f"{output_file})", f"Title:Mass {para_masspoint} {cat} GeV"
                )
            else:
                canvas.Print(f"{output_file}", f"Title:Mass {para_masspoint} {cat} GeV")
            canvas.Close()

            if np.sum(physics_weight[dw.class_target == 1]) > 0:
                print("sum weights:", np.sum(physics_weight))
                print("min weight:", np.min(physics_weight))
                print("max weight:", np.max(physics_weight))
                display = sklearn.metrics.RocCurveDisplay.from_predictions(
                    dw.class_target == 0,
                    pred_signal,
                    sample_weight=np.clip(physics_weight, 0, None),
                    ax=ax,
                    name=f"Signal vs rest m{para_masspoint}",
                )
                _ = display.ax_.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title="Signal-vs-Background ROC curves",
                )

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curves {cat}")
        ax.grid()
        plt.savefig(os.path.join(output_folder, f"ROC_{cat}.pdf"))

        data_obs = ROOT.TH1D(
            f"data_obs",
            f"data_obs",
            nQuantBins,
            0.0,
            1.0,
        )
        ROOTOut.WriteObject(data_obs, f"data_obs")
        ROOTOut.Close()
