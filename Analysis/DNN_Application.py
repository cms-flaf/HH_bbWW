# python3 /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/AnaProd/NNInterface.py --inModelDir /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/nn_models --inFile /tmp/prsolank/luigi-tmp-416131263.root --outFileName /tmp/prsolank/luigi-tmp-862152055.root --uncConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/Run2_2018/weights.yaml --globalConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/global.yaml --EraName e2018 --Mass 400 --Spin 2 --PairType 2

from __future__ import annotations
import os, sys
import gc
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import FLAF.Common.Utilities as Utilities
import Analysis.hh_bbww as analysis
from contextlib import contextmanager


class DNNProducer:
    def __init__(self, cfg, payload_name, period):

        self.cfg = cfg
        self.payload_name = payload_name
        self.period = period

        sys.path.append(os.environ["ANALYSIS_PATH"])

        load_features = set()
        columns_to_save = set()

        self.cfg_dict = {
            "DL": self.cfg.get("DL", None),
            "SL": self.cfg.get("SL", None),
        }

        self.models = {}
        self.masses = self.cfg.get("masses")
        self.dnnConfig = {}

        for channel, cfg in self.cfg_dict.items():
            if cfg == None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue
            self.models[channel] = {}
            self.dnnConfig[channel] = {}
            parametric = cfg.get("parametric", False)

            dnnFolder = os.path.join(
                os.environ["ANALYSIS_PATH"], "config", "DNN", cfg["version"]
            )

            for mass in self.masses:
                if parametric:
                    this_mass_folder = dnnFolder
                    mass = 0
                    # Set mass to 0 since parametric should have the same config each time
                    # This helps the mass loop happening in the ApplyDNN func

                else:
                    this_mass_folder = os.path.join(dnnFolder, f"m{mass}")
                file_name = os.path.join(this_mass_folder, "dnn_config.yaml")
                with open(file_name, "r") as file:
                    self.dnnConfig[channel][f"m{mass}"] = yaml.safe_load(file)

                load_features.update(self.dnnConfig[channel][f"m{mass}"]["features"])

                modelname_parity = self.dnnConfig[channel][f"m{mass}"][
                    "modelname_parity"
                ]
                self.dnnConfig[channel][f"m{mass}"]["model_paths"] = [
                    [f"{os.path.join(this_mass_folder, x)}.onnx", y]
                    for x, y in modelname_parity
                ]

        columns_to_save.update(
            [f"{self.payload_name}_{col}" for col in self.cfg["columns"]]
        )

        # What to save in tmp file
        load_features.update(["FullEventId", "event", "SL", "DL"])
        self.vars_to_save = load_features

    def run(self, array):
        print("Running DNN producer")

        array = self.ApplyDNN(array)
        array = self.selectDNN(array)

        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    del array[col]

        # Rename the branches
        for col in self.cfg["columns"]:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")
                print(f"Available columns were {array.fields}")

        return array

    def ApplyDNN(self, branches):
        output_fields = {}

        for channel, all_dnnConfig in self.dnnConfig.items():
            if all_dnnConfig == None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            masses = self.masses if ("m0" not in all_dnnConfig.keys()) else [0]
            for mass in masses:
                dnnConfig = all_dnnConfig[f"m{mass}"]
                models = dnnConfig["model_paths"]

                features = dnnConfig["features"]

                nClasses = (
                    dnnConfig["nClasses"] if "nClasses" in dnnConfig.keys() else 3
                )
                nParity = dnnConfig["nParity"] if "nParity" in dnnConfig.keys() else 4

                use_parametric = dnnConfig["use_parametric"]
                param_mass_list = dnnConfig["parametric_list"]

                class_names_list = (
                    dnnConfig["class_names"]
                    if "class_names" in dnnConfig.keys()
                    else ["Signal", "TT", "DY"]
                )

                nEvents = len(branches)
                print(f"Running DNN Over {nEvents} events")

                event_number = np.asarray(branches.event)
                if nParity != 1:
                    event_mod = event_number % nParity

                array = np.stack(
                    [
                        np.asarray(getattr(branches, feature_name), dtype=np.float32)
                        for feature_name in features
                    ],
                    axis=1,
                )

                if use_parametric:
                    final_array = np.empty(
                        (nEvents, array.shape[1] + 1), dtype=np.float32
                    )
                    final_array[:, :-1] = array

                all_predictions = np.zeros(
                    (len(param_mass_list), nEvents, nClasses), dtype=np.float32
                )

                for parityIdx, [onnx_name, parityfunc] in enumerate(models):
                    sess = ort.InferenceSession(onnx_name)
                    for param_idx, param_mass in enumerate(param_mass_list):
                        if use_parametric:
                            final_array[:, -1] = param_mass
                            input_array = final_array
                        else:
                            input_array = array

                        prediction = sess.run(None, {"x": input_array})
                        class_prediction = np.asarray(prediction[0], dtype=np.float32)

                        if nParity != 1:
                            mask = event_mod != parityIdx
                            class_prediction[~mask, :] = 0.0

                        all_predictions[param_idx] += class_prediction
                        del prediction, class_prediction

                if nParity != 1:
                    all_predictions /= nParity - 1

                # Last save the branches
                for param_idx, param_mass in enumerate(param_mass_list):
                    this_param_prediction = all_predictions[param_idx, :, :]
                    this_param_prediction_logit = np.clip(
                        this_param_prediction, 1e-7, 1 - 1e-7
                    )
                    this_param_prediction_logit = np.log(
                        this_param_prediction_logit / (1 - this_param_prediction_logit)
                    )

                    for class_idx, class_name in enumerate(class_names_list):
                        field_name = f"{channel}_M{param_mass}_{class_name}"
                        output_fields[field_name] = this_param_prediction_logit[
                            :, class_idx
                        ].copy()

                if use_parametric:
                    del final_array
                del array
                if nParity != 1:
                    del event_mod
                del all_predictions
                gc.collect()
                print("Finishing call, memory?")
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                print(f"Current memory usage: {mem_mb:.2f} MB")

        for field_name, values in output_fields.items():
            branches[field_name] = values

        del output_fields

        return branches

    def selectDNN(self, branches):
        # Here we will take SL and DL and choose which branch to save as final column
        output_fields = {}

        classes_to_save = ["Signal", "TT", "DY", "ST"]

        for mass in self.masses:
            for class_name in classes_to_save:
                field_name = f"M{mass}_{class_name}"
                # Build the empty branches with ones
                if f"SL_{field_name}" not in branches.fields:
                    branches[f"SL_{field_name}"] = np.zeros_like(branches.event)
                if f"DL_{field_name}" not in branches.fields:
                    branches[f"DL_{field_name}"] = np.zeros_like(branches.event)
                output_fields[field_name] = np.where(
                    branches.SL,
                    branches[f"SL_{field_name}"],
                    branches[f"DL_{field_name}"],
                )

        for field_name, values in output_fields.items():
            branches[field_name] = values
        del output_fields
        return branches


def make_session(path):
    """
    Helper function to create onnx sessions with options
    """
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


@contextmanager
def onnx_session(path):
    """
    RAII session helper
    """
    sess = make_session(path)
    try:
        yield sess
    finally:
        del sess
        gc.collect()


def compute_probas(logits, eps=1e-7):    
    max_logits = np.max(logits, axis=axis, keepdims=True)
    exp_values = np.exp(logits - max_logits)
    probas = exp_values / np.sum(exp_values, axis=axis, keepdims=True)
    probas = np.clip(probas, eps, 1.0 - eps)
    probas /= np.sum(probas, axis=axis, keepdims=True)
    return probas


def compute_logits(probas, eps=1e-7):
    probas = np.clip(probas, eps, 1 - eps)
    return np.log(probas/(1 - probas))


class TwoStageDNNProducer:
    def __init__(self, cfg, payload_name, period):
        self.cfg = cfg
        self.payload_name = payload_name
        self.period = period
        self.masses = self.cfg.get("masses")

        sys.path.append(os.environ["ANALYSIS_PATH"])

        load_features = set()
        columns_to_save = set()

        # categories with dedicated networks/configs, each in its own subdir
        self.categories = ["boosted", "resolved"]

        # maps channel -> dnn specs
        self.channel_dnn_specs = {
            "DL": self.cfg.get("DL", None),
            "SL": self.cfg.get("SL", None),
        }

        self.dnn_configs = {}
        # channel -> category -> models directory
        self.models_folders = {}

        for channel, specs in self.channel_dnn_specs.items():
            if specs is None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            parametric = specs.get("parametric", False)
            if parametric:
                raise NotImplementedError("Parametric two-stage DNN inference has not been implemented.")

            base_folder = os.path.join(
                os.environ["ANALYSIS_PATH"], "config", "DNN", specs["version"]
            )

            self.dnn_configs[channel] = {}
            self.models_folders[channel] = {}

            for category in self.categories:
                # boosted/resolved live in their own subdirectories
                category_folder = os.path.join(base_folder, category)
                self.models_folders[channel][category] = category_folder

                self.dnn_configs[channel][category] = {}

                binary_cfg_path = os.path.join(
                    category_folder, f"binary_{category}_dnn_config.yaml"
                )
                multiclass_cfg_path = os.path.join(
                    category_folder, f"multiclass_{category}_dnn_config.yaml"
                )

                with open(binary_cfg_path, "r") as f:
                    self.dnn_configs[channel][category]["binary"] = yaml.safe_load(f)
                with open(multiclass_cfg_path, "r") as f:
                    self.dnn_configs[channel][category]["multiclass"] = yaml.safe_load(f)

                load_features.update(
                    self.dnn_configs[channel][category]["binary"]["features"]
                )
                load_features.update(
                    self.dnn_configs[channel][category]["multiclass"]["features"]
                )

        columns_to_save.update(
            [f"{self.payload_name}_{col}" for col in self.cfg["columns"]]
        )

        load_features.update(["FullEventId", "event", "SL", "DL", "boosted"])
        self.vars_to_save = load_features


    def run(self, array):
        print("Running TwoStageDNNProducer producer")

        array = self.ApplyDNN(array)
        array = self.SelectDNN(array)

        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    del array[col]

        # Rename the branches
        for col in self.cfg["columns"]:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")
                print(f"Available columns were {array.fields}")

        return array


    def ApplyDNN(self, branches):
        output_fields = {}
        class_names_list = self.cfg["class_names"]

        for channel, specs in self.channel_dnn_specs.items():
            if specs is None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            num_parities = specs["num_parities"]
            num_classes = specs["num_classes"]
            num_events = len(branches)
            event_id = np.asarray(branches.event)

            bin_name_pattern = specs["binary_name_pattern"]
            multiclass_name_pattern = specs["multiclass_name_pattern"]

            for category in self.categories:
                models_folder = self.models_folders[channel][category]

                # 0..num_classes - multiclass scores
                # -1 in the last axis - binary score
                predictions = np.full(
                    (num_events, num_classes + 1), -1.0, dtype=np.float32
                )

                binary_feature_list = self.dnn_configs[channel][category]["binary"]["features"]
                multiclass_feature_list = self.dnn_configs[channel][category]["multiclass"]["features"]

                reuse_inputs = binary_feature_list == multiclass_feature_list

                for train_parity in range(num_parities):
                    application_parity = (train_parity + 3) % num_parities
                    application_mask = event_id % num_parities == application_parity

                    if not np.any(application_mask):
                        continue

                    selected = branches[application_mask]

                    def build(feature_list):
                        return np.stack(
                            [np.asarray(getattr(selected, fn), dtype=np.float32)
                             for fn in feature_list],
                            axis=1,
                        )

                    if reuse_inputs:
                        inputs = build(binary_feature_list)
                    else:
                        binary_inputs = build(binary_feature_list)
                        multiclass_inputs = build(multiclass_feature_list)

                    for mp in self.masses:
                        binary_model_name = bin_name_pattern.format(
                            train_parity=train_parity, mass=mp
                        )
                        multiclass_model_name = multiclass_name_pattern.format(
                            train_parity=train_parity, mass=mp
                        )

                        binary_model_path = os.path.join(models_folder, binary_model_name)
                        multiclass_model_path = os.path.join(models_folder, multiclass_model_name)

                        with onnx_session(binary_model_path) as bs, \
                             onnx_session(multiclass_model_path) as ms:
                            bs_in = bs.get_inputs()[0].name
                            ms_in = ms.get_inputs()[0].name

                            multiclass_scores = ms.run(
                                None,
                                {ms_in: inputs if reuse_inputs else multiclass_inputs},
                            )[0]
                            binary_scores = bs.run(
                                None,
                                {bs_in: inputs if reuse_inputs else binary_inputs},
                            )[0]

                            predictions[application_mask, :num_classes] = multiclass_scores
                            predictions[application_mask, -1] = binary_scores.ravel()

                assert np.all(predictions >= 0), \
                    f"All predictions must be filled/positive for {channel}/{category}"

                for mp in self.masses:
                    for class_idx, class_name in enumerate(class_names_list):
                        mc_field_name = f"multiclass_{channel}_{category}_M{mp}_{class_name}"
                        output_fields[mc_field_name] = predictions[:, class_idx].copy()

                    bin_field_name = f"binary_{channel}_{category}_M{mp}"
                    output_fields[bin_field_name] = predictions[:, -1].copy()

                del predictions

        for field_name, values in output_fields.items():
            branches[field_name] = values

        del output_fields
        gc.collect()
        return branches

    
    def SelectDNN(self, branches):
        nEvents = len(branches)
        output_fields = {}
        class_names_list = self.cfg["class_names"]

        def get_field(name):
            """Fetch a branch as float32, or zeros if it's missing."""
            if name in branches.fields:
                return np.asarray(branches[name], dtype=np.float32)
            return np.zeros(nEvents, dtype=np.float32)

        def get_mask(name):
            if name in branches.fields:
                return np.asarray(branches[name], dtype=bool)
            return np.zeros(nEvents, dtype=bool)

        sl_mask = get_mask("SL")
        boosted_mask = get_mask("boosted")

        def select(field_builder):
            """
            field_builder(channel, category) -> branch field name.
            Picks boosted/resolved per event, then SL/DL per event.
            """
            sl_sel = np.where(
                boosted_mask,
                get_field(field_builder("SL", "boosted")),
                get_field(field_builder("SL", "resolved")),
            )
            dl_sel = np.where(
                boosted_mask,
                get_field(field_builder("DL", "boosted")),
                get_field(field_builder("DL", "resolved")),
            )
            return np.where(sl_mask, sl_sel, dl_sel)

        for mass in self.masses:
            m = int(mass)

            # multiclass scores
            for class_name in class_names_list:
                target_field = f"M{m}_{class_name}"
                output_fields[target_field] = select(
                    lambda ch, cat: f"multiclass_{ch}_{cat}_{target_field}"
                )

            # binary score
            target_field = f"M{m}_binary"
            output_fields[target_field] = select(
                lambda ch, cat: f"binary_{ch}_{cat}_M{m}"
            )

        for field_name, values in output_fields.items():
            branches[field_name] = values

        return branches