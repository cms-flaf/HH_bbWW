import argparse
import threading
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
import os
import shutil
import json
import yaml
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--output_folder", required=True, type=str, help="Output Folder"
    )
    parser.add_argument(
        "--limit_paths", required=True, nargs="+", type=str, default=None
    )

    args = parser.parse_args()

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        os.makedirs(args.output_folder, exist_ok=True)

        plot_vars = ["dropout", "l2_rate", "learning_rate", "n_epochs"]

        res2b_dict = {}
        res1b_dict = {}
        boosted_dict = {}

        best_limits = {
            "res2b": {
                "value": 99999,
                "training": "",
            },
            "res1b": {
                "value": 99999,
                "training": "",
            },
            "boosted": {
                "value": 99999,
                "training": "",
            },
        }

        for var in plot_vars:
            res2b_dict[var] = {
                "x": [],
                "y": [],
            }
            res1b_dict[var] = {
                "x": [],
                "y": [],
            }
            boosted_dict[var] = {
                "x": [],
                "y": [],
            }

        for training in args.limit_paths:
            training_yamls_resolved = [
                yam
                for yam in os.listdir(training)
                if yam.endswith(".yaml") and "Resolved" in yam
            ]
            training_yamls_boosted = [
                yam
                for yam in os.listdir(training)
                if yam.endswith(".yaml") and "Boosted" in yam
            ]

            training_config_resolved = None
            with open(os.path.join(training, training_yamls_resolved[0]), "r") as f:
                training_config_resolved = yaml.safe_load(f)

            training_config_boosted = None
            with open(os.path.join(training, training_yamls_boosted[0]), "r") as f:
                training_config_boosted = yaml.safe_load(f)

            res2b_limits = None
            with open(os.path.join(training, "m600_res2b.json")) as f:
                d = json.load(f)
                res2b_limits = d["exp"]
            res1b_limits = None
            with open(os.path.join(training, "m600_recovery.json")) as f:
                d = json.load(f)
                res1b_limits = d["exp"]
            boosted_limits = None
            with open(os.path.join(training, "m600_boosted.json")) as f:
                d = json.load(f)
                boosted_limits = d["exp"]

            for var in plot_vars:
                res2b_dict[var]["x"].append(training_config_resolved[var])
                res2b_dict[var]["y"].append(res2b_limits)

                res1b_dict[var]["x"].append(training_config_resolved[var])
                res1b_dict[var]["y"].append(res1b_limits)

                if training_config_boosted["learning_rate"] > 0.001:
                    continue
                boosted_dict[var]["x"].append(training_config_boosted[var])
                boosted_dict[var]["y"].append(boosted_limits)

            if res2b_limits < best_limits["res2b"]["value"]:
                best_limits["res2b"]["value"] = res2b_limits
                best_limits["res2b"]["training"] = training_yamls_resolved[0]
            if res1b_limits < best_limits["res1b"]["value"]:
                best_limits["res1b"]["value"] = res1b_limits
                best_limits["res1b"]["training"] = training_yamls_resolved[0]
            if boosted_limits < best_limits["boosted"]["value"]:
                best_limits["boosted"]["value"] = boosted_limits
                best_limits["boosted"]["training"] = training_yamls_boosted[0]

        # print(f"Finished the loop")
        # print(best_limits)
        # print(res2b_dict)
        # print(res1b_dict)
        # print(boosted_dict)

        for var in plot_vars:
            x = res2b_dict[var]["x"]
            y = res2b_dict[var]["y"]

            fig, ax = plt.subplots()
            ax.scatter(x, y)

            plt.title(f"Res2b")
            plt.xlabel(f"{var}")
            plt.ylabel(f"Limits")
            # plt.show()
            plt.savefig(os.path.join(args.output_folder, f"res2b_{var}.pdf"))

            x = res1b_dict[var]["x"]
            y = res1b_dict[var]["y"]

            fig, ax = plt.subplots()
            ax.scatter(x, y)

            plt.title(f"Res1b")
            plt.xlabel(f"{var}")
            plt.ylabel(f"Limits")
            # plt.show()
            plt.savefig(os.path.join(args.output_folder, f"res1b_{var}.pdf"))

            x = boosted_dict[var]["x"]
            y = boosted_dict[var]["y"]

            fig, ax = plt.subplots()
            ax.scatter(x, y)

            plt.title(f"Boosted")
            plt.xlabel(f"{var}")
            plt.ylabel(f"Limits")
            # plt.show()
            plt.savefig(os.path.join(args.output_folder, f"boosted_{var}.pdf"))

        with open(os.path.join(args.output_folder, f"best_limits.json"), "w") as fp:
            json.dump(best_limits, fp)

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
