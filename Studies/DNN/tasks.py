import law
import os
import yaml
import shutil
import luigi
from FLAF.run_tools.law_customizations import (
    Task,
    HTCondorWorkflow,
)
from FLAF.RunKit.run_tools import ps_call


class DNNTrainingTask(Task, HTCondorWorkflow, law.LocalWorkflow):
    training_configuration_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(DNNTrainingTask, self).__init__(*args, **kwargs)

    def create_branch_map(self):
        branches = {}
        DNN_Configurations = [ os.path.join(self.training_configuration_dir, x) for x in os.listdir(self.training_configuration_dir) if x.endswith('.yaml') ]
        # open yaml training_configuration
        for DNN_Configuration in DNN_Configurations:
            with open(DNN_Configuration, 'r') as f:
                config = yaml.safe_load(f)
            br_idx = len(branches)
            branches[br_idx] = (config, DNN_Configuration)
        return branches

    def output(self):
        config, config_name = self.branch_data
        training_name = config["training_name"]
        outFileName = f"{training_name}.onnx"
        output_path = os.path.join(
            "DNNTraining",
            self.version,
            self.period,
            training_name,
            outFileName)
        return [ self.remote_target(output_path, fs=self.fs_anaTuple), self.remote_target(config_name, fs=self.fs_anaTuple) ]
    
    def run(self):
        config, config_name = self.branch_data
        training_name = config["training_name"]
        dnn_trainer = os.path.join(
            self.ana_path(), "Studies", "DNN", "DNN_Trainer_Condor.py"
        )
        job_home, remove_job_home = self.law_job_home()
        print(f"At job_home {job_home}")

        tmpFile = os.path.join(job_home, f"{training_name}.onnx")


        training_file = config["training_file"]
        weight_file = config["weight_file"]
        batch_config = config["batch_config"]
        test_training_file = config["test_training_file"]
        test_weight_file = config["test_weight_file"]
        test_batch_config = config["test_batch_config"]

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
            "--output_file",
            tmpFile,
            "--setup-config",
            config_name
        ]
        ps_call(dnn_trainer_cmd, verbose=1)

        model_output = self.output()[0]
        with model_output.localize("w") as tmp_local_file:
            out_local_path = tmp_local_file.path
            shutil.move(tmpFile, out_local_path)
        config_output = self.output()[1]
        with config_output.localize("w") as tmp_local_file:
            out_local_path = tmp_local_file.path
            shutil.copy(config_name, out_local_path)

        if remove_job_home:
            shutil.rmtree(job_home)