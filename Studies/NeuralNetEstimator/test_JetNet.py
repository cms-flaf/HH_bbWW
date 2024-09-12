from Common.JetNet import JetNet
from Common.DataWrapper import DataWrapper
from Common.JetNet_utils import PlotPrediction
import yaml 
import argparse


def main():
    parser = argparse.ArgumentParser(prog='train_net', description='Trains Neural Net Estimator')
    parser.add_argument('config_file', type=str, help="File with neural net configuration")
    parser.add_argument('files', type=str, help="File with list of input files separated by newline character")
    parser.add_argument('model_path', type=str, help="Path where to save trained model")

    args = parser.parse_args()
    config = args.config_file
    path_to_model = args.model_path
    test_files = []
    with open(arg.files, 'r') as file:
        test_files = [line[:-1] for line in file.readlines()]

    if not test_files:
        raise RuntimeError(f"file {arg.files} contained empty list of input files")
    
    with open(config, 'r') as stream:
        cfg = yaml.safe_load(stream)

        dw = DataWrapper(cfg)
        dw.ReadFiles(test_files)

        net = JetNet(cfg)
        net.LoadModel(path_to_model)

        pred_df = net.Predict(dw.test_features)
        PlotPrediction(dw.test_labels, pred_df)
    
    
if __name__ == '__main__':
    main()
