import os
import uproot
import numpy as np
import yaml
from tqdm import tqdm

import ROOT
import FLAF.RunKit.grid_tools as grid_tools
from FLAF.RunKit.run_tools import ps_call

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()
ROOT.EnableImplicitMT(4)


def measure_cut_datasets(config_dict, output_folder, remote=False):
    storage_folder = os.path.join(config_dict["storage_folder"])

    process_dict = {}

    iterate_cut = config_dict["iterate_cut"]
    parity_cut = config_dict["parity_cut"]

    signal_list = config_dict["signal"]

    background_list = config_dict["background"]

    for nParity in range(config_dict["nParity"]):
        output_nParity = os.path.join(output_folder, f"nParity{nParity}_Merged")
        os.makedirs(output_nParity, exist_ok=True)

        nParity_string = f"nParity_{nParity}"

        process_dict[nParity_string] = {}

        for signal_name in config_dict["signal"]:
            process_dict[nParity_string][signal_name] = {}
            signal_dict = config_dict["signal"][signal_name]
            mass_points = signal_dict["mass_points"]
            for mass_point in mass_points:
                process_dict[nParity_string][signal_name][mass_point] = {
                    "total": 0,
                    "total_cut": 0,
                    "total_cut_weighted": 0.0,
                }

        for background_name in config_dict["background"]:
            process_dict[nParity_string][background_name] = {}
            background_dict = config_dict["background"][background_name]
            dataset_names = background_dict["background_datasets"]
            for dataset_name in dataset_names:
                process_dict[nParity_string][background_name][dataset_name] = {
                    "total": 0,
                    "total_cut": 0,
                    "total_cut_weighted": 0.0,
                }

    print("Looping signal datasets")
    for signal_name in signal_list:
        signal_dict = config_dict["signal"][signal_name]
        mass_points = signal_dict["mass_points"]
        dataset_name_format = signal_dict["dataset_name_format"]
        class_value = signal_dict["class_value"]

        for mass_point in tqdm(mass_points):
            X_mass = mass_point
            dataset_name = dataset_name_format.format(mass_point)

            process_dir = os.path.join(storage_folder, dataset_name)

            if remote:
                input_files = f"root://cmseos.fnal.gov/{process_dir}/*.root"
            else:
                input_files = f"{process_dir}/*.root"
            treeName = "Events"
            rdf = ROOT.RDataFrame(treeName, input_files)

            total = rdf.Count().GetValue()
            rdf = rdf.Filter(iterate_cut)

            for nParity in range(config_dict["nParity"]):
                nParity_string = f"nParity_{nParity}"
                parity_cut_formatted = parity_cut.format(
                    nParity=config_dict["nParity"], parity_scan=nParity
                )
                output_nParity = os.path.join(output_folder, f"nParity{nParity}_Merged")
                output_file = os.path.join(output_nParity, f"{dataset_name}_merge.root")

                rdf_tmp = rdf.Filter(parity_cut_formatted)
                cut = rdf_tmp.Count().GetValue()
                weighted_cut = rdf_tmp.Sum("weight_Central").GetValue()

                process_dict[nParity_string][signal_name][mass_point]["total"] = total
                process_dict[nParity_string][signal_name][mass_point]["total_cut"] = cut
                process_dict[nParity_string][signal_name][mass_point][
                    "total_cut_weighted"
                ] = weighted_cut
                rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
                rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
                rdf_tmp.Snapshot(treeName, output_file)

    for background_name in background_list:
        background_dict = config_dict["background"][background_name]
        dataset_names = background_dict["background_datasets"]
        class_value = background_dict["class_value"]
        X_mass = 0

        print(f"Looping background {background_name}")
        for dataset_name in tqdm(dataset_names):
            process_dir = os.path.join(storage_folder, dataset_name)

            if remote:
                input_files = f"root://cmseos.fnal.gov/{process_dir}/*.root"
            else:
                input_files = f"{process_dir}/*.root"
            treeName = "Events"
            rdf = ROOT.RDataFrame(treeName, input_files)

            total = rdf.Count().GetValue()
            rdf = rdf.Filter(iterate_cut)

            for nParity in range(config_dict["nParity"]):
                nParity_string = f"nParity_{nParity}"
                parity_cut_formatted = parity_cut.format(
                    nParity=config_dict["nParity"], parity_scan=nParity
                )
                output_nParity = os.path.join(output_folder, f"nParity{nParity}_Merged")
                output_file = os.path.join(output_nParity, f"{dataset_name}_merge.root")

                rdf_tmp = rdf.Filter(parity_cut_formatted)
                cut = rdf_tmp.Count().GetValue()
                weighted_cut = rdf_tmp.Sum("weight_Central").GetValue()

                process_dict[nParity_string][background_name][dataset_name][
                    "total"
                ] = total
                process_dict[nParity_string][background_name][dataset_name][
                    "total_cut"
                ] = cut
                process_dict[nParity_string][background_name][dataset_name][
                    "total_cut_weighted"
                ] = weighted_cut
                rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
                rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
                rdf_tmp.Snapshot(treeName, output_file)

    for nParity in range(config_dict["nParity"]):
        nParity_string = f"nParity_{nParity}"
        out_yaml = f"dataset_distribution_parity{nParity}.yaml"
        with open(os.path.join(output_folder, out_yaml), "w") as outfile:
            yaml.dump(process_dict[nParity_string], outfile)


def hadd_files(config_dict, output_folder):
    for nParity in range(config_dict["nParity"]):
        # hadd the files together to make a final merged.root
        hadd_out = os.path.join(output_folder, f"nParity{nParity}_Merged.root")
        hadd_in = os.path.join(output_folder, f"nParity{nParity}_Merged/*.root")
        # ps_call("hadd", hadd_out, hadd_in)
        os.system(f"hadd {hadd_out} {hadd_in}")


def add_weight_file(output_folder):
    inNames = [
        os.path.join(output_folder, x)
        for x in os.listdir(output_folder)
        if x.endswith(".root")
    ]
    for inName in inNames:
        if "weight" in inName:
            continue
        print(f"On file {inName}")
        in_file = uproot.open(inName)
        # outName = f"{inName[:-5]}_weight.root"
        outName = f"{inName[:-5]}_weight_m600.root"
        out_file = uproot.recreate(outName)

        tree = in_file["Events"]
        branches_to_load = [
            "class_value",
            "X_mass",
            "weight_Central",
        ]
        branches = tree.arrays(branches_to_load)

        X_mass = branches["X_mass"]
        class_targets = branches["class_value"]
        class_weight = branches["weight_Central"]

        # Set to binary for now actually
        class_targets = np.where(class_targets > 0, 1, class_targets)

        # Set any negative weight events to 0
        # class_weight = np.where(class_weight <= 0, 0.0, class_weight)

        # Clip weights to be within +- 3 std of mean
        mean_weight = np.mean(np.abs(class_weight))
        std = np.std(np.abs(class_weight))
        print(f"Normalizing from {mean_weight} +- {std}")
        class_weight = np.clip(
            class_weight, -(mean_weight + (3 * std)), (mean_weight + (3 * std))
        )

        # Set specific masses if you want
        class_weight = np.where(
            (class_targets == 0) & (X_mass != 600), 0.0, class_weight
        )

        # Total_Signal == Total_Background
        total_signal = np.sum(np.where(class_targets == 0, class_weight, 0.0))
        total_background = np.sum(np.where(class_targets != 0, class_weight, 0.0))

        print(f"Total signal: {total_signal}")
        print(f"Total background: {total_background}")
        norm_factor = total_background / total_signal
        class_weight = np.where(
            class_targets != 0, class_weight, class_weight * norm_factor
        )

        # norm_factor = total_signal / total_background
        # class_weight = np.where(
        #     class_targets == 0, class_weight, class_weight * norm_factor
        # )
        print(f"After reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, class_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, class_weight, 0.0))}"
        )

        # Total_Background1 == Total_Background2 == Total_Background3
        # Scale each background to total, then reduce all to total
        ### Do not scale backgrounds to each other in binary classifier ###
        # total_background = np.sum(np.where(class_targets != 0, class_weight, 0.0))
        # for class_value in np.unique(class_targets):
        #     if class_value == 0:
        #         continue # Don't do anything with signal here
        #     this_total = np.sum(np.where(class_targets == class_value, class_weight, 0.0))
        #     rescale_factor = total_background / this_total
        #     class_weight = np.where(
        #         class_targets == class_value, class_weight*rescale_factor, class_weight
        #     )
        # current_total = np.sum(np.where(class_targets != 0, class_weight, 0.0))
        # rescale_factor = total_background / current_total
        # class_weight = np.where(
        #     class_targets != 0, class_weight*rescale_factor, class_weight
        # )

        print(f"Final reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, class_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, class_weight, 0.0))}"
        )

        out_dict = {
            "class_weight": class_weight,
            "class_target": class_targets,
        }

        print("Finished with dict")
        print(out_dict)

        out_file["weight_tree"] = out_dict
        out_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        default="default_dataset.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        type=str,
        default="/eos/user/d/daebi/DNN_Training_Datasets",
        help="Output folder to store dataset",
    )

    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    output_base = args.output_folder
    output_folder = os.path.join(output_base, f"Dataset")
    if os.path.exists(output_folder):
        print(f"Output folder {output_folder} exists!!!")
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {config_file} {output_folder}/.")

    measure_cut_datasets(config_dict, output_folder)
    hadd_files(config_dict, output_folder)
    add_weight_file(output_folder)
