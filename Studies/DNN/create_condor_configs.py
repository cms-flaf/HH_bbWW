import os
import yaml
import awkward as ak

template = "config/default_training_setup_doubleLep.yaml"
output_folder = "CondorConfigs_Sep16"
input_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v3/Dataset_Run3_2022/batchfile{}.root"
weight_file_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v3/Dataset_Run3_2022/weightfile{}.root"
batch_config_template = "/eos/user/d/daebi/DNN_Training_Datasets/DoubleLepton_v3/Dataset_Run3_2022/batch_config_parity{}.yaml"
os.makedirs(output_folder, exist_ok=True)

with open(template, 'r') as f:
    default_config = yaml.safe_load(f)

var_parse_dict = {
    'adv_grad_factor': [ 1.0, 0.9, 0.7, 0.5 ],
    'adv_learning_rate': [ 0.001, 0.0005, 0.0001 ],
    'adv_submodule_steps': [ 20, 50 ],
    'class_grad_factor': [ 0.2, 0.1 ],
    'learning_rate': [ 0.001, 0.0005, 0.0001 ],
    'n_epochs': [ 50 ],
    'dropout': [ 0.0, 0.1 ],
}

var_names = var_parse_dict.keys()
var_combinations_list = [ x for x in var_parse_dict.values()]
var_combinations = ak.cartesian(var_combinations_list, axis=0)

for i, varset in enumerate(var_combinations):
    config = default_config.copy()
    for name, var in zip(var_names, varset.tolist()):
        config[name] = var
    for j in range(1):
        # Set up each parity
        config["training_file"] = input_file_template.format(j)
        config["weight_file"] = weight_file_template.format(j)
        config["batch_config"] = batch_config_template.format(j)
        config["test_training_file"] = input_file_template.format((j+1)%4)
        config["test_weight_file"] = weight_file_template.format((j+1)%4)
        config["test_batch_config"] = batch_config_template.format((j+1)%4)

        training_name = f"DNN_DoubleLep_Training{i}_par{j}"
        config['training_name'] = training_name

        outFileName = f"{training_name}.yaml"
        outFilePath = os.path.join(output_folder, outFileName)
        with open(outFilePath, 'w') as f:
            yaml.dump(config, f)
