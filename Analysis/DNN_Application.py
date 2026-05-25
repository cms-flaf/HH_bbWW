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


class DNNProducer:
    def __init__(self, cfg, payload_name, period):

        self.cfg = cfg
        self.payload_name = payload_name
        self.period = period

        sys.path.append(os.environ["ANALYSIS_PATH"])

        load_features = set()
        self.models = {}
        self.parametric = (
            self.cfg["parametric"] if "parametric" in self.cfg.keys() else True
        )
        self.masses = "parametric" if self.parametric else self.cfg["masses"]
        self.dnnConfig = {}
        dnnFolder = os.path.join(
            os.environ["ANALYSIS_PATH"], "config", "DNN", self.cfg["version"]
        )

        for mass in self.masses:
            if mass == "parametric":
                this_mass_folder = dnnFolder
            else:
                this_mass_folder = os.path.join(dnnFolder, f"m{mass}")
            file_name = os.path.join(this_mass_folder, "dnn_config.yaml")
            with open(file_name, "r") as file:
                self.dnnConfig[f"m{mass}"] = yaml.safe_load(file)

            load_features.update(self.dnnConfig[f"m{mass}"]["features"])

            load_features.update(["FullEventId"])
            load_features.update(["event"])

            modelname_parity = self.dnnConfig[f"m{mass}"]["modelname_parity"]
            self.dnnConfig[f"m{mass}"]["model_paths"] = [
                [f"{os.path.join(this_mass_folder, x)}.onnx", y]
                for x, y in modelname_parity
            ]

        # What to save in tmp file
        self.vars_to_save = load_features
        # What to save for final output
        self.cols_to_save = [
            f"{self.payload_name}_{col}" for col in self.cfg["columns"]
        ]

    def run(self, array):
        print("Running DNN producer")

        array = self.ApplyDNN(array)

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

        for mass in self.masses:

            dnnConfig = self.dnnConfig[f"m{mass}"]
            models = dnnConfig["model_paths"]

            features = dnnConfig["features"]

            nClasses = dnnConfig["nClasses"] if "nClasses" in dnnConfig.keys() else 3
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
                final_array = np.empty((nEvents, array.shape[1] + 1), dtype=np.float32)
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

                for class_idx, class_name in enumerate(class_names_list):
                    field_name = f"M{param_mass}_{class_name}"
                    output_fields[field_name] = this_param_prediction[
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
