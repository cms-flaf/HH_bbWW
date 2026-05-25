import os

storage_dir = "/eos/user/d/daebi/HH_bbWW/DNNTraining/23May_Resolved_v1/Run3_2022EE"

output_dir = "DoubleLepton_IndependentMasses_v1"

parametric = False

masslist = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000]

paritylist = [0, 1, 2, 3]

os.makedirs(output_dir, exist_ok=True)


if not parametric:
    for mass in masslist:
        mass_output = os.path.join(output_dir, f"m{mass}")
        os.makedirs(mass_output, exist_ok=True)
        folder_name = f"DNN_DoubleLepton_Resolved_Training0_par0_m{mass}"

        os.system(
            f"cp {os.path.join(storage_dir, folder_name, folder_name, 'dnn_config.yaml')} {mass_output}/."
        )

        for parity in paritylist:
            folder_name = f"DNN_DoubleLepton_Resolved_Training0_par{parity}_m{mass}"
            onnx_file = os.path.join(
                storage_dir, folder_name, folder_name, "stage1.onnx"
            )
            os.system(
                f"cp {onnx_file} {mass_output}/ResHH_Classifier_parity{parity}.onnx"
            )
