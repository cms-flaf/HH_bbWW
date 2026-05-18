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
    parser.add_argument("--resolved", required=True, type=int)
    parser.add_argument(
        "--limit_paths", required=True, nargs="+", type=str, default=None
    )

    args = parser.parse_args()

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        os.makedirs(args.output_folder, exist_ok=True)

        plot_vars = [
            # "dropout",
            # "l2_rate",
            # "learning_rate",
            # "n_epochs",
            # "gamma1",
            # "gamma2",
            # "n_units_reduction_factor",
            # "n_layers",
            # "loss_scale",
        ]
        best_x = []
        best_y = []

        res2b_dict = {}
        res1b_dict = {}
        boosted_dict = {}

        best_limits = {}

        masslist = [
            300,
            400,
            500,
            550,
            600,
            650,
            700,
            800,
            900,
            1000,
            # 1200,
            # 1400,
            # 1600,
            # 1800,
            # 2000,
        ]

        for mass in masslist:

            res2b_dict[f"m{mass}"] = {}
            res1b_dict[f"m{mass}"] = {}
            boosted_dict[f"m{mass}"] = {}

            for var in plot_vars:
                res2b_dict[f"m{mass}"][var] = {
                    "x": [],
                    "y": [],
                }
                res1b_dict[f"m{mass}"][var] = {
                    "x": [],
                    "y": [],
                }
                boosted_dict[f"m{mass}"][var] = {
                    "x": [],
                    "y": [],
                }

            best_limits[f"m{mass}"] = {
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

        for training in args.limit_paths:
            print(f"Checking training {training}")

            if args.resolved:
                training_yamls_resolved = [
                    yam
                    for yam in os.listdir(training)
                    if yam.endswith(".yaml") and "Resolved" in yam
                ]

                training_config_resolved = None
                with open(os.path.join(training, training_yamls_resolved[0]), "r") as f:
                    training_config_resolved = yaml.safe_load(f)

                for mass in masslist:

                    res2b_limits = None
                    with open(os.path.join(training, f"m{mass}_res2b.json")) as f:
                        d = json.load(f)
                        res2b_limits = d["exp"]
                    res1b_limits = None
                    with open(os.path.join(training, f"m{mass}_recovery.json")) as f:
                        d = json.load(f)
                        res1b_limits = d["exp"]

                    for var in plot_vars:
                        res2b_dict[f"m{mass}"][var]["x"].append(
                            training_config_resolved[var]
                        )
                        res2b_dict[f"m{mass}"][var]["y"].append(res2b_limits)

                        res1b_dict[f"m{mass}"][var]["x"].append(
                            training_config_resolved[var]
                        )
                        res1b_dict[f"m{mass}"][var]["y"].append(res1b_limits)

                    if res2b_limits < best_limits[f"m{mass}"]["res2b"]["value"]:
                        best_limits[f"m{mass}"]["res2b"]["value"] = res2b_limits
                        best_limits[f"m{mass}"]["res2b"]["training"] = (
                            training_yamls_resolved[0]
                        )
                    if res1b_limits < best_limits[f"m{mass}"]["res1b"]["value"]:
                        best_limits[f"m{mass}"]["res1b"]["value"] = res1b_limits
                        best_limits[f"m{mass}"]["res1b"]["training"] = (
                            training_yamls_resolved[0]
                        )

            else:
                training_yamls_boosted = [
                    yam
                    for yam in os.listdir(training)
                    if yam.endswith(".yaml") and "Boosted" in yam
                ]

                training_config_boosted = None
                with open(os.path.join(training, training_yamls_boosted[0]), "r") as f:
                    training_config_boosted = yaml.safe_load(f)

                for mass in masslist:

                    boosted_limits = None
                    with open(os.path.join(training, f"m{mass}_boosted.json")) as f:
                        d = json.load(f)
                        boosted_limits = d["exp"]

                    for var in plot_vars:
                        boosted_dict[f"m{mass}"][var]["x"].append(
                            training_config_boosted[var]
                        )
                        boosted_dict[f"m{mass}"][var]["y"].append(boosted_limits)

                    if boosted_limits < best_limits[f"m{mass}"]["boosted"]["value"]:
                        best_limits[f"m{mass}"]["boosted"]["value"] = boosted_limits
                        best_limits[f"m{mass}"]["boosted"]["training"] = (
                            training_yamls_boosted[0]
                        )

        # Example data
        for mass in masslist:
            best_x.append(mass)
            if args.resolved:
                best_y.append(best_limits[f"m{mass}"]["res2b"]["value"])
            else:
                best_y.append(best_limits[f"m{mass}"]["boosted"]["value"])

            for var in plot_vars:
                x = res2b_dict[f"m{mass}"][var]["x"]
                y = res2b_dict[f"m{mass}"][var]["y"]

                fig, ax = plt.subplots()
                ax.scatter(x, y)

                plt.title(f"Res2b")
                plt.xlabel(f"{var}")
                plt.ylabel(f"Limits")
                plt.savefig(
                    os.path.join(args.output_folder, f"res2b_{var}_m{mass}.pdf")
                )

                plt.close()

                x = res1b_dict[f"m{mass}"][var]["x"]
                y = res1b_dict[f"m{mass}"][var]["y"]

                fig, ax = plt.subplots()
                ax.scatter(x, y)

                plt.title(f"Res1b")
                plt.xlabel(f"{var}")
                plt.ylabel(f"Limits")
                plt.savefig(
                    os.path.join(args.output_folder, f"res1b_{var}_m{mass}.pdf")
                )

                plt.close()

                x = boosted_dict[f"m{mass}"][var]["x"]
                y = boosted_dict[f"m{mass}"][var]["y"]

                fig, ax = plt.subplots()
                ax.scatter(x, y)

                plt.title(f"Boosted")
                plt.xlabel(f"{var}")
                plt.ylabel(f"Limits")
                plt.savefig(
                    os.path.join(args.output_folder, f"boosted_{var}_m{mass}.pdf")
                )

                plt.close()

        # Create scatter plot

        if args.resolved:
            plt.scatter(best_x, best_y, label="Run3 Res2b", color="blue")
            x_2018_res2b = [300, 400, 500, 550, 600, 650, 700, 800, 900]
            y_2018_res2b = [
                5.4275,
                2.0375,
                0.4641,
                0.3125,
                0.2570,
                0.2289,
                0.2266,
                0.2242,
                0.2773,
            ]
            plt.scatter(x_2018_res2b, y_2018_res2b, label="2018 Res2b", color="red")
            x_2018_res1b = [300, 400, 500, 550, 600, 650, 700, 800, 900]
            y_2018_res1b = [
                9.6375,
                3.85,
                1.4062,
                1.2313,
                0.8750,
                0.7906,
                0.7344,
                0.6438,
                0.5781,
            ]
            plt.scatter(x_2018_res1b, y_2018_res1b, label="2018 Res1b", color="green")
        else:
            plt.scatter(best_x, best_y, label="Run3 Boosted", color="blue")
            x_2018 = [300, 400, 500, 550, 600, 650, 700, 800, 900]
            y_2018 = [62.7, 9.975, 2.125, 1.2969, 0.6844, 0.3422, 0.193, 0.1152, 0.0785]
            plt.scatter(x_2018, y_2018, label="2018 Boosted", color="red")

        plt.legend()

        # Add labels and title
        plt.xlabel("Mass")
        plt.ylabel("Limits")
        plt.title("Limits")

        plt.ylim(0.001, 100)
        plt.yscale("log")
        plt.grid(True)

        # Show the plot
        plt.savefig(os.path.join(args.output_folder, f"boosted_limits.pdf"))
        plt.close()

        with open(os.path.join(args.output_folder, f"best_limits.json"), "w") as fp:
            json.dump(best_limits, fp)

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
