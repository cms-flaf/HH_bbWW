import os
import yaml
import awkward as ak

# Resolved
template = "config/training_setup_doubleLep_resolved.yaml"
output_folder = "CondorConfigs_19Mar_DoubleLepton_Resolved_Full_HME_m600"
input_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW_v2601a/Studies/DNN/ResolvedDataset_Mar14/Dataset/nParity{}_Merged.root"
weight_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW_v2601a/Studies/DNN/ResolvedDataset_Mar14/Dataset/nParity{}_Merged_weight_m600.root"
training_name = "DNN_DoubleLepton_Resolved_Training{i}_par{j}"
var_parse_dict = {
    "learning_rate": [0.0005],
    "n_epochs": [100, 200],
    "dropout": [0.2, 0.5],
    "parametric_list": [[600]],
    "l2_rate": [0.001, 0.0001, 0.01],
    "gamma1": [2.0, 1.5, 2.5],
    "gamma2": [0.5, 0.2, 0.7],
}

# Boosted
# template = "config/training_setup_doubleLep_boosted.yaml"
# output_folder = "CondorConfigs_19Mar_DoubleLepton_Boosted_Full_HME_m600"
# input_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW_v2601a/Studies/DNN/BoostedDataset_Mar14/Dataset/nParity{}_Merged.root"
# weight_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW_v2601a/Studies/DNN/BoostedDataset_Mar14/Dataset/nParity{}_Merged_weight_m600.root"
# training_name = "DNN_DoubleLepton_Boosted_Training{i}_par{j}"
# var_parse_dict = {
#     "learning_rate": [0.0005 ],
#     "n_epochs": [ 100, 200 ],
#     "dropout": [ 0.2, 0.5 ],
#     'parametric_list': [ [ 600 ] ],
#     'l2_rate': [ 0.001, 0.01, 0.005 ],
#     'gamma1': [ 2.0, 1.5, 2.5 ],
#     'gamma2': [ 0.5, 0.2, 0.7 ],
# }

os.makedirs(output_folder, exist_ok=True)

with open(template, "r") as f:
    default_config = yaml.safe_load(f)


var_names = var_parse_dict.keys()
var_combinations_list = [x for x in var_parse_dict.values()]
var_combinations = ak.cartesian(var_combinations_list, axis=0)

for i, varset in enumerate(var_combinations):
    config = default_config.copy()
    for name, var in zip(var_names, varset.tolist()):
        config[name] = var
    for j in range(4):
        # Set up each parity
        config["training_file"] = input_file_template.format(j)
        config["weight_file"] = weight_file_template.format(j)
        config["test_training_file"] = input_file_template.format((j + 1) % 4)
        config["test_weight_file"] = weight_file_template.format((j + 1) % 4)
        config["validation_file"] = input_file_template.format((j + 2) % 4)
        config["validation_weight_file"] = weight_file_template.format((j + 2) % 4)

        config["training_name"] = training_name.format(i=i, j=j)

        outFileName = f"{training_name.format(i = i, j = j)}.yaml"
        outFilePath = os.path.join(output_folder, outFileName)
        with open(outFilePath, "w") as f:
            yaml.dump(config, f)
