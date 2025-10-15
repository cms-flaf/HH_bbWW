import os
import yaml
import awkward as ak


template = "config/default_training_setup_singleLep.yaml"
output_folder = "CondorConfigs_Oct12_SingleLepton"
input_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/SingleLepton_v4/Dataset_Run3_2022/batchfile{}.root"
weight_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/SingleLepton_v4/Dataset_Run3_2022/weightfile{}.root"
hme_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/SingleLepton_v4/Dataset_Run3_2022/batchfile{}_HME_Friend.root"
batch_config_template = "/eos/user/d/daebi/DNN_Training_Datasets/SingleLepton_v4/Dataset_Run3_2022/batch_config_parity{}.yaml"
training_name = "DNN_SingleLep_Training{i}_par{j}"
var_parse_dict = {
    "adv_grad_factor": [0.0],
    "adv_learning_rate": [0.0001],
    "adv_submodule_steps": [0],
    "class_grad_factor": [1.0],
    "learning_rate": [ 0.00001 ],
    "n_epochs": [ 100 ],
    "dropout": [ 0.3 ],
    "weight_decay": [None],
    "adv_weight_decay": [None],
    "disco_lambda_factor": [ 0 ],
    "n_disco_layers": [ 5 ],
    "n_disco_units": [ 256 ],
    "disco_activation": [ "relu" ],
    'hmefeatures': [ ['SingleLep_DeepHME_mass', 'SingleLep_DeepHME_mass_error'], None ]
}


# template = "config/default_training_setup_doubleLep.yaml"
# output_folder = "CondorConfigs_Oct10_DoubleLepton"
# input_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v5/Dataset_Run3_2022/batchfile{}.root"
# weight_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v5/Dataset_Run3_2022/weightfile{}.root"
# hme_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v5/Dataset_Run3_2022/batchfile{}_HME_Friend.root"
# batch_config_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v5/Dataset_Run3_2022/batch_config_parity{}.yaml"
# training_name = "DNN_DoubleLep_Training{i}_par{j}"
# var_parse_dict = {
#     'adv_grad_factor': [ 0.0 ],
#     'adv_learning_rate': [ 0.0001 ],
#     'adv_submodule_steps': [ 0,],
#     'class_grad_factor': [ 1.0 ],
#     'learning_rate': [ 0.00001 ],
#     'n_epochs': [ 50 ],
#     'dropout': [ 0.1 ],
#     'weight_decay': [ None ],
#     'adv_weight_decay': [ None ],
#     'disco_lambda_factor': [ 0 ],
#     'n_disco_layers': [ 10 ],
#     'hmefeatures': [ ['DoubleLep_DeepHME_mass', 'DoubleLep_DeepHME_mass_error'], None ]
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
    for j in range(1):
        # Set up each parity
        config["training_file"] = input_file_template.format(j)
        config["weight_file"] = weight_file_template.format(j)
        config["hme_friend_file"] = hme_file_template.format(j)
        config["batch_config"] = batch_config_template.format(j)
        config["test_training_file"] = input_file_template.format((j + 1) % 4)
        config["test_weight_file"] = weight_file_template.format((j + 1) % 4)
        config["test_hme_friend_file"] = hme_file_template.format((j + 1) % 4)
        config["test_batch_config"] = batch_config_template.format((j + 1) % 4)
        config["validation_file"] = input_file_template.format((j + 2) % 4)
        config["validation_weight_file"] = weight_file_template.format((j + 2) % 4)
        config["validation_hme_friend_file"] = hme_file_template.format((j + 2) % 4)
        config["validation_batch_config"] = batch_config_template.format((j + 2) % 4)

        config["training_name"] = training_name.format(i=i, j=j)

        outFileName = f"{training_name.format(i = i, j = j)}.yaml"
        outFilePath = os.path.join(output_folder, outFileName)
        with open(outFilePath, "w") as f:
            yaml.dump(config, f)
