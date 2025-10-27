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


class DNNTrainingTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()
    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 48.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 4)

    def __init__(self, *args, **kwargs):
        super(DNNTrainingTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNN_Configurations = [
            os.path.join(self.training_configuration_dir, x)
            for x in os.listdir(self.training_configuration_dir)
            if x.endswith(".yaml")
        ]
        DNN_Configurations.sort()
        # open yaml training_configuration
        for DNN_Configuration in DNN_Configurations:
            with open(DNN_Configuration, "r") as f:
                config = yaml.safe_load(f)
            br_idx = len(branches)
            branches[br_idx] = (config, DNN_Configuration)
        return branches

    def output(self):
        config, config_name = self.branch_data
        training_name = config["training_name"]
        outFolderName = f"{training_name}"
        output_path = os.path.join(
            "DNNTraining", self.version, self.period, training_name, outFolderName
        )
        config_path = os.path.join(
            "DNNTraining",
            self.version,
            self.period,
            training_name,
            os.path.basename(config_name),
        )
        return [
            self.remote_target(output_path, fs=self.fs_anaTuple),
            self.remote_target(config_path, fs=self.fs_anaTuple),
        ]

    def run(self):
        config, config_name = self.branch_data
        training_name = config["training_name"]
        dnn_trainer = os.path.join(
            self.ana_path(), "Studies", "DNN", "DNN_Trainer_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFolder = os.path.join(job_home, f"{training_name}")

        training_file = config["training_file"]
        weight_file = config["weight_file"]
        batch_config = config["batch_config"]
        test_training_file = config["test_training_file"]
        test_weight_file = config["test_weight_file"]
        test_batch_config = config["test_batch_config"]

        hme_friend_file = config["hme_friend_file"]
        test_hme_friend_file = config["test_hme_friend_file"]

        # with config["training_file"].localize("r") as training_file, config["weight_file"].localize("r") as weight_file, config["batch_config"].localize("r") as batch_config, config["test_training_file"].localize("r") as test_training_file, config["test_weight_file"].localize("r") as test_weight_file, config["test_batch_config"].localize("r") as test_batch_config:
        dnn_trainer_cmd = [
            "python3",
            "-u",
            dnn_trainer,
            "--training_file",
            training_file,
            "--weight_file",
            weight_file,
            "--batch_config",
            batch_config,
            "--test_training_file",
            test_training_file,
            "--test_weight_file",
            test_weight_file,
            "--test_batch_config",
            test_batch_config,
            "--output_folder",
            tmpFolder,
            "--setup-config",
            config_name,
            "--hme_friend_file",
            hme_friend_file,
            "--test_hme_friend_file",
            test_hme_friend_file,
        ]
        ps_call(dnn_trainer_cmd, verbose=1)

        model_output = self.output()[0]
        with model_output.localize("w") as tmp_local_folder:
            out_local_path = tmp_local_folder.path
            shutil.move(tmpFolder, out_local_path)
        config_output = self.output()[1]
        with config_output.localize("w") as tmp_local_file:
            out_local_path = tmp_local_file.path
            shutil.copy(config_name, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)


class DNNValidationTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(DNNValidationTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNNTraining_map = DNNTrainingTask.req(
            self, branch=-1, branches=()
        ).create_branch_map()
        k = 0
        for n_branch, (config, config_name) in DNNTraining_map.items():
            branches[k] = (config, config_name, n_branch)
            k += 1
        return branches

    def workflow_requires(self):
        return {"DNNTrainer": DNNTrainingTask.req(self)}

    def requires(self):
        config, config_name, n_branch = self.branch_data
        return DNNTrainingTask.req(
            self,
            branch=n_branch,
            max_runtime=DNNTrainingTask.max_runtime._default,
            branches=(),
        )

    def output(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        outFileName = f"validation.pdf"
        output_path = os.path.join(
            "DNNTraining", self.version, self.period, training_name, outFileName
        )
        return [self.remote_target(output_path, fs=self.fs_anaTuple)]

    def run(self):
        config, config_name, n_branch = self.branch_data
        training_name = config["training_name"]
        dnn_validator = os.path.join(
            self.ana_path(), "Studies", "DNN", "DNN_Validator_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFile = os.path.join(job_home, f"{training_name}.pdf")

        validation_file = config["validation_file"]
        valitation_weight_file = config["validation_weight_file"]
        valitation_batch_config = config["validation_batch_config"]

        validation_hme_friend_file = config["validation_hme_friend_file"]

        tmp_local = os.path.join(self.input()[0].path, "best.onnx")
        # with self.input()[0].localize("r") as model_file, self.input()[1].localize("r") as model_config:
        with self.remote_target(tmp_local, fs=self.fs_anaTuple).localize(
            "r"
        ) as model_file, self.input()[1].localize("r") as model_config:
            print(os.listdir())
            dnn_validator_cmd = [
                "python3",
                "-u",
                dnn_validator,
                "--validation_file",
                validation_file,
                "--validation_weight_file",
                valitation_weight_file,
                "--validation_batch_config",
                valitation_batch_config,
                "--output_file",
                tmpFile,
                "--setup-config",
                config_name,
                "--model-name",
                model_file.path,
                "--model-config",
                model_config.path,
                "--validation_hme_friend_file",
                validation_hme_friend_file,
            ]
            ps_call(dnn_validator_cmd, verbose=1)

        validation_output = self.output()[0]
        with validation_output.localize("w") as tmp_local_file:
            out_local_path = tmp_local_file.path
            shutil.move(tmpFile, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)
