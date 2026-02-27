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

        features_to_load = self.feature_names

        features_to_load.append("X_mass")
        features_to_load.append("weight_Central")

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


class Model(tf.keras.Model):
    def __init__(self, setup, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = setup

        self.nClasses = setup["nClasses"]

        self.class_loss = tf.keras.losses.categorical_crossentropy
        self.class_accuracy = tf.keras.metrics.categorical_accuracy

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

        self.class_min_tracker = tf.keras.metrics.Mean(name="class_min")
        self.class_max_tracker = tf.keras.metrics.Mean(name="class_max")

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
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
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

        def compute_losses():
            y_pred_class = self(x, training=training)

            class_loss_vec = self.class_loss(y_class, y_pred_class)

            class_loss = tf.reduce_mean(class_loss_vec * class_weight)

            return y_pred_class, class_loss_vec, class_loss

        if training:
            with tf.GradientTape() as class_tape:
                y_pred_class, class_loss_vec, class_loss = compute_losses()
        else:
            y_pred_class, class_loss_vec, class_loss = compute_losses()

        self.class_min_tracker.update_state(tf.reduce_min(y_pred_class[:, 0]))
        self.class_max_tracker.update_state(tf.reduce_max(y_pred_class[:, 0]))

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

        if training:
            grad = class_tape.gradient(class_loss, self.trainable_variables)
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
            self.class_min_tracker,
            self.class_max_tracker,
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

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    nClasses = setup["nClasses"]
    batch_size = setup["batch_size"]
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dw.features,
            (tf.one_hot(dw.class_target, nClasses), dw.class_weight),
        )
    ).batch(batch_size, drop_remainder=True)
    train_tf_dataset = train_tf_dataset.shuffle(
        len(train_tf_dataset), reshuffle_each_iteration=True
    )

    test_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            test_dw.features,
            (
                tf.one_hot(test_dw.class_target, nClasses),
                test_dw.class_weight,
            ),
        )
    ).batch(batch_size, drop_remainder=True)
    test_tf_dataset = test_tf_dataset.shuffle(
        len(test_tf_dataset), reshuffle_each_iteration=True
    )

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

    model = Model(setup)
    model.compile(
        loss=None,
        optimizer=tf.keras.optimizers.Nadam(
            learning_rate=setup["learning_rate"], weight_decay=setup["weight_decay"]
        ),
    )
    model(dw.features)
    model.summary()

    callbacks = []

    verbose = setup["verbose"] if "verbose" in setup else 0
    verbose = 1
    print("Fit model")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        verbose=verbose,
        epochs=setup["n_epochs"],
        shuffle=False,
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
        plt.savefig(os.path.join(output_folder, f"{metric}.pdf"), bbox_inches="tight")
        plt.clf()

    PlotMetric(history, "class_loss", output_folder)

    PlotMetric(history, "class_min", output_folder)

    PlotMetric(history, "class_max", output_folder)

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
    output_file,
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
    batch_size = setup["batch_size"]
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dw.features,
            (tf.one_hot(dw.class_target, nClasses), dw.class_weight),
        )
    ).batch(batch_size, drop_remainder=True)
    train_tf_dataset = train_tf_dataset.shuffle(
        len(train_tf_dataset), reshuffle_each_iteration=True
    )

    para_masspoint_list = [300, 400, 600, 800, 1000, 3000, 4000]  # [300, 450, 800]
    canvases = []
    for para_masspoint in para_masspoint_list:
        print(f"Validating mass {para_masspoint}")
        if dw.use_parametric:
            dw.SetPredictParamValue(para_masspoint)
        features = dw.features_paramSet if dw.use_parametric else dw.features_no_param

        pred = sess.run(None, {"x": features})
        pred_class = pred[0]
        pred_signal = pred_class[:, 0]

        class_weight = dw.class_weight
        physics_weight = dw.physics_weight

        # Class Plots
        # Lets build Masks
        Sig_This_Mass = dw.X_mass == para_masspoint
        Sig_mask = (Sig_This_Mass) & (dw.class_target == 0)

        Background_mask = dw.class_target == 1

        TT_mask = dw.class_target == 1

        DY_mask = dw.class_target == 2

        Other_mask = dw.class_target == 3

        # Set class quantiles based on signal
        nQuantBins = 10
        quant_binning_class = np.zeros(
            nQuantBins + 1
        )  # Need +1 because 10 bins actually have 11 edges
        if len(pred_signal[Sig_mask]) == 0:
            print("No signal events in this mass point! Skip!")
            continue
        quant_binning_class[1:nQuantBins] = np.quantile(
            pred_signal[Sig_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )  # Change list to something dynamic with nQuantBins
        quant_binning_class[-1] = 1.0
        print("We found quant binning class")
        print(quant_binning_class)
        print("From the signal prediction")
        print(pred_signal[Sig_mask])

        mask_dict = {
            "Signal": Sig_mask,
            "TT": TT_mask,
            "DY": DY_mask,
            "Other": Other_mask,
        }

        mask_dict = {
            "Signal": Sig_mask,
            "Background": Background_mask,
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

            plotlabel = f"Class Output for {process_name} ParaMass {para_masspoint} GeV"
            ROOT_ClassOutput.Draw()
            ROOT_ClassOutput.SetTitle(plotlabel)
            ROOT_ClassOutput.SetStats(0)
            min_val = max(
                0.0001,
                ROOT_ClassOutput.GetMinimum(),
            )
            max_val = ROOT_ClassOutput.GetMaximum()

            ROOT_ClassOutput.GetYaxis().SetRangeUser(0.001 * min_val, 1000 * max_val)

            legend_list.append(ROOT.TLegend(0.5, 0.8, 0.9, 0.9))
            legend = legend_list[-1]
            legend.AddEntry(ROOT_ClassOutput, f"{process_name}")
            legend.Draw()

            print(f"Setting canvas to log scale with range {min_val}, {max_val}")
            p1.SetLogy()
            p1.SetGrid()

        if para_masspoint == para_masspoint_list[0]:
            canvas.Print(f"{output_file}(", f"Title:Mass {para_masspoint} GeV")
            print("Saved [")
        elif para_masspoint == para_masspoint_list[-1]:
            canvas.Print(f"{output_file})", f"Title:Mass {para_masspoint} GeV")
            print("Saved ]")
        else:
            canvas.Print(f"{output_file}", f"Title:Mass {para_masspoint} GeV")
        print(f"Saved mass {para_masspoint}")

        canvas.Close()
