import argparse
import Studies.DNN.DNN_Class as DNNClass
import threading
import yaml
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--training_file", required=True, type=str, help="Training file"
    )
    parser.add_argument("--weight_file", required=True, type=str, help="Weight file")
    parser.add_argument(
        "--hme_friend_file",
        required=False,
        type=str,
        default=None,
        help="HME friend file",
    )
    parser.add_argument(
        "--batch_config", required=True, type=str, help="Batch config file"
    )
    parser.add_argument(
        "--test_training_file", required=True, type=str, help="Test file"
    )
    parser.add_argument(
        "--test_weight_file", required=True, type=str, help="Test weight file"
    )
    parser.add_argument(
        "--test_hme_friend_file",
        required=False,
        type=str,
        default=None,
        help="Test HME friend file",
    )
    parser.add_argument(
        "--test_batch_config", required=True, type=str, help="Test batch config file"
    )
    parser.add_argument("--output_folder", required=True, type=str, help="Output model")
    parser.add_argument(
        "--setup-config", required=True, type=str, help="Setup config for training"
    )

    args = parser.parse_args()

    setup = {}
    with open(args.setup_config, "r") as file:
        setup = yaml.safe_load(file)

    modelname_parity = []

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        config_dict = {}
        with open(args.batch_config, "r") as file:
            config_dict = yaml.safe_load(file)
        test_config_dict = {}
        with open(args.test_batch_config, "r") as file:
            test_config_dict = yaml.safe_load(file)

        model = DNNClass.train_dnn(
            setup,
            args.training_file,
            args.weight_file,
            config_dict,
            args.test_training_file,
            args.test_weight_file,
            test_config_dict,
            args.output_folder,
            args.hme_friend_file,
            args.test_hme_friend_file,
        )

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
