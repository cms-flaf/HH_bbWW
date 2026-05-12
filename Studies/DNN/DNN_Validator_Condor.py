import argparse
import Studies.DNN.DNN_Class_HistTuples as DNNClass
import threading
import yaml
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--validation_file", required=True, type=str, help="Training file"
    )
    parser.add_argument(
        "--validation_weight_file", required=True, type=str, help="Weight file"
    )
    parser.add_argument(
        "--output_folder", required=True, type=str, help="Output Folder"
    )
    parser.add_argument(
        "--setup-config", required=True, type=str, help="Setup config for training"
    )
    parser.add_argument(
        "--model-name", required=True, type=str, help="Model file for validation"
    )
    parser.add_argument(
        "--model-name-stage2",
        required=False,
        type=str,
        help="Model file for validation",
    )
    parser.add_argument(
        "--model-config", required=True, type=str, help="Config file for model"
    )

    args = parser.parse_args()

    setup = {}
    with open(args.setup_config, "r") as file:
        setup = yaml.safe_load(file)

    modelname_parity = []

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        # model = DNNClass.validate_dnn(
        model = DNNClass.validate_dnn_pipeline(
            setup,
            args.validation_file,
            args.validation_weight_file,
            args.output_folder,
            args.model_name,
            model_stage2=args.model_name_stage2,
            model_config=args.model_config,
        )

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
