import argparse
import threading
import yaml
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
import os
import shutil
from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.envToJson import get_cmsenv
import uproot
import json

cmssw_env = get_cmsenv(cmssw_path=os.getenv("FLAF_CMSSW_BASE"))


def limit_to_json(input_root, output_json):
    t = uproot.open(f"{input_root}:limit", branches=["limit"])
    array = t.arrays()
    limits = {
        "exp-2": array.limit[0],
        "exp-1": array.limit[1],
        "exp": array.limit[2],
        "exp+1": array.limit[3],
        "exp+2": array.limit[4],
        "observed": array.limit[5],
    }
    with open(output_json, "w") as fp:
        json.dump(limits, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--output_folder", required=True, type=str, help="Output Folder"
    )
    parser.add_argument(
        "--validation_paths", required=True, nargs="+", type=str, default=None
    )

    args = parser.parse_args()

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        for cat in ["res1b", "res2b", "boosted"]:
            hadd_list = []
            for val_path in args.validation_paths:
                hadd_list.append(os.path.join(val_path, f"validation_{cat}.root"))
            ps_call(["hadd", f"run3_{cat}.root", *hadd_list])

        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "call_combine_benchmark.sh",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_boosted.txt",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_res2b.txt",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_recovery.txt",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_combined.txt",
            ),
            ".",
        )

        os.makedirs(args.output_folder, exist_ok=True)

        for mass in [600]:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_boosted.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_boosted",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                "higgsCombine_m600_boosted.AsymptoticLimits.mH600.root",
                os.path.join(args.output_folder, f"m{mass}_boosted.json"),
            )

            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_res2b.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_res2b",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                "higgsCombine_m600_res2b.AsymptoticLimits.mH600.root",
                os.path.join(args.output_folder, f"m{mass}_res2b.json"),
            )

            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_recovery.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_recovery",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                "higgsCombine_m600_recovery.AsymptoticLimits.mH600.root",
                os.path.join(args.output_folder, f"m{mass}_recovery.json"),
            )

            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_combined.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_combined",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                "higgsCombine_m600_combined.AsymptoticLimits.mH600.root",
                os.path.join(args.output_folder, f"m{mass}_combined.json"),
            )

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()
