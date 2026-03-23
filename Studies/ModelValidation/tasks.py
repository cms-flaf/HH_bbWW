import law
import os
import yaml
import shutil
import luigi
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
    copy_param,
)
from FLAF.RunKit.run_tools import ps_call
from Studies.DNN.tasks import DNNValidationTask
import re
import contextlib


class DNNLimitTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir_resolved = luigi.Parameter()
    training_configuration_dir_boosted = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(DNNLimitTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        unique_trainings = {}
        branches = {}
        DNNValidation_Resolved_map = DNNValidationTask.req(
            self,
            branch=-1,
            branches=(),
            training_configuration_dir=self.training_configuration_dir_resolved,
        ).create_branch_map()
        for n_branch, (
            config,
            config_name,
            n_branch,
        ) in DNNValidation_Resolved_map.items():
            training_id = re.search(r"Training\d+", config_name).group(0)
            if training_id not in unique_trainings.keys():
                unique_trainings[training_id] = {
                    "config_names_resolved": [],
                    "branches_resolved": [],
                    "config_names_boosted": [],
                    "branches_boosted": [],
                }
            unique_trainings[training_id]["config_names_resolved"].append(config_name)
            unique_trainings[training_id]["branches_resolved"].append(n_branch)

        DNNValidation_Boosted_map = DNNValidationTask.req(
            self,
            branch=-1,
            branches=(),
            training_configuration_dir=self.training_configuration_dir_boosted,
        ).create_branch_map()
        for n_branch, (
            config,
            config_name,
            n_branch,
        ) in DNNValidation_Boosted_map.items():
            training_id = re.search(r"Training\d+", config_name).group(0)
            if training_id not in unique_trainings.keys():
                unique_trainings[training_id] = {
                    "config_names_resolved": [],
                    "branches_resolved": [],
                    "config_names_boosted": [],
                    "branches_boosted": [],
                }
            unique_trainings[training_id]["config_names_boosted"].append(config_name)
            unique_trainings[training_id]["branches_boosted"].append(n_branch)

        k = 0
        for training_id in unique_trainings.keys():
            branches[k] = (
                training_id,
                unique_trainings[training_id]["config_names_resolved"],
                unique_trainings[training_id]["config_names_boosted"],
                unique_trainings[training_id]["branches_resolved"],
                unique_trainings[training_id]["branches_boosted"],
            )
            k += 1
        return branches

    def workflow_requires(self):
        branch_set_resolved = set()
        branch_set_boosted = set()
        for idx, (
            training_id,
            config_names_resolved,
            config_names_boosted,
            branches_resolved,
            branches_boosted,
        ) in self.branch_map.items():
            branch_set_resolved.update(branches_resolved)
            branch_set_boosted.update(branches_boosted)
        reqs = {}

        if len(branch_set_resolved) > 0:
            reqs["DNNValidation_Resolved"] = DNNValidationTask.req(
                self,
                training_configuration_dir=self.training_configuration_dir_resolved,
                branches=tuple(branch_set_resolved),
                customisations=self.customisations,
                max_runtime=DNNValidationTask.max_runtime._default,
                n_cpus=DNNValidationTask.n_cpus._default,
            )

        if len(branch_set_boosted) > 0:
            reqs["DNNValidation_Boosted"] = DNNValidationTask.req(
                self,
                training_configuration_dir=self.training_configuration_dir_boosted,
                branches=tuple(branch_set_boosted),
                customisations=self.customisations,
                max_runtime=DNNValidationTask.max_runtime._default,
                n_cpus=DNNValidationTask.n_cpus._default,
            )

        return reqs

    def requires(self):
        (
            training_id,
            config_names_resolved,
            config_names_boosted,
            branches_resolved,
            branches_boosted,
        ) = self.branch_data
        return [
            [
                DNNValidationTask.req(
                    self,
                    training_configuration_dir=self.training_configuration_dir_resolved,
                    branch=br,
                    max_runtime=DNNValidationTask.max_runtime._default,
                    branches=(br,),
                )
                for br in branches_resolved
            ],
            [
                DNNValidationTask.req(
                    self,
                    training_configuration_dir=self.training_configuration_dir_boosted,
                    branch=br,
                    max_runtime=DNNValidationTask.max_runtime._default,
                    branches=(br,),
                )
                for br in branches_boosted
            ],
        ]

    def output(self):
        (
            training_id,
            config_names_resolved,
            config_names_boosted,
            branches_resolved,
            branches_boosted,
        ) = self.branch_data
        training_name = training_id
        outFileName = f"limits"
        output_path = os.path.join(
            "DNNLimits", self.version, self.period, training_name, outFileName
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        (
            training_id,
            config_names_resolved,
            config_names_boosted,
            branches_resolved,
            branches_boosted,
        ) = self.branch_data
        training_name = training_id
        dnn_limits = os.path.join(
            self.ana_path(), "Studies", "ModelValidation", "DNN_Limit_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        print("Inputs?")
        print(self.input())

        with contextlib.ExitStack() as stack:
            resolved_inputs = self.input()[0]
            boosted_inputs = self.input()[1]
            combined_inputs = resolved_inputs + boosted_inputs
            print(f"Starting localize of {len(combined_inputs)} inputs")
            local_inputs = [
                stack.enter_context(inp[0].localize("r")).path
                for inp in combined_inputs
            ]

            dnn_limits_cmd = [
                "python3",
                "-u",
                dnn_limits,
                "--output_folder",
                training_name,
                "--validation_paths",
                *local_inputs,
            ]
            ps_call(dnn_limits_cmd, verbose=1, cwd=job_home)

        for fname in config_names_resolved + config_names_boosted:
            shutil.copy(fname, tmpFolder)

        limit_outputs = self.output()
        with limit_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)


class DNNComparisonTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir_resolved = luigi.Parameter()
    training_configuration_dir_boosted = luigi.Parameter()

    # fancy_string = "gamma1<5 && gamma2>3"

    def __init__(self, *args, **kwargs):
        super(DNNComparisonTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        unique_trainings = {}
        branches = {}
        DNNLimit_map = DNNLimitTask.req(
            self,
            branch=-1,
            branches=(),
        ).create_branch_map()
        branch_list = []
        for n_branch, (
            training_id,
            config_names_resolved,
            config_names_boosted,
            branches_resolved,
            branches_boosted,
        ) in DNNLimit_map.items():
            unique_trainings[training_id] = (
                n_branch,
                config_names_resolved,
                config_names_boosted,
            )
            branch_list.append(n_branch)

        k = 0
        branches[0] = (unique_trainings, branch_list)
        return branches

    def workflow_requires(self):
        reqs = {}
        reqs["DNNLimit"] = DNNLimitTask.req(
            self,
            branches=tuple(),
            customisations=self.customisations,
            max_runtime=DNNLimitTask.max_runtime._default,
            n_cpus=DNNLimitTask.n_cpus._default,
        )
        return reqs

    def requires(self):
        unique_trainings, branch_list = self.branch_data
        return [
            DNNLimitTask.req(
                self,
                branch=(br),
                max_runtime=DNNLimitTask.max_runtime._default,
                branches=(br,),
            )
            for br in branch_list
        ]

    def output(self):
        unique_trainings, branch_list = self.branch_data
        outFileName = f"DNNComparisonPlots"
        output_path = os.path.join(
            "DNNComparison", self.version, self.period, outFileName
        )
        return [
            self.remote_target(output_path, fs=self.fs_histograms),
        ]

    def run(self):
        unique_trainings, branch_list = self.branch_data

        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"comparison_plots")

        dnn_comparison = os.path.join(
            self.ana_path(), "Studies", "ModelValidation", "DNN_Comparison_Condor.py"
        )

        with contextlib.ExitStack() as stack:
            local_inputs = [
                stack.enter_context(inp[0].localize("r")).path for inp in self.input()
            ]

            dnn_comparison_cmd = [
                "python3",
                "-u",
                dnn_comparison,
                "--output_folder",
                tmpFolder,
                "--limit_paths",
                *local_inputs,
            ]
            ps_call(dnn_comparison_cmd, verbose=1, cwd=job_home)

        comparison_outputs = self.output()
        with comparison_outputs[0].localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)

            # eval(my_str, env={"gamma1": point["gamma1"]})

            # Here we have localized the limit folders, now we need code to look over the configs to build a dict of training params and their limits

            # Make a dict of just a list of all limits for a given variable point (gamma = 0.5, all limits)
            # ALSO other part that finds and returns training with best limit for fun

            # Then we would make plots of that dict with error bars for up/down to show impact

            # Then copy that final output folder to the transferred law output

            # And given the best training point, do the inverse -- freeze one value and scan the other vars
