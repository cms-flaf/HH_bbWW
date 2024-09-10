from common.JetNet import JetNet
from common.DataWrapper import DataWrapper
from common.JetNet_utils import PlotLoss, PlotPrediction
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
    train_files = []
    with open(arg.files, 'r') as file:
        train_files = [line[:-1] for line in file.readlines()]

    if not train_files:
        raise RuntimeError(f"file {arg.files} contained empty list of input files")

    with open(config, 'r') as stream:
        cfg = yaml.safe_load(stream)

        data = DataWrapper(cfg)
        data.ReadFiles(train_files)
        
        data.Shuffle()
        data.TrainTestSplit()        

        net = JetNet(cfg)
        net.ConfigureModel(data.train_features.shape)
        history = net.Fit(data.train_features, data.train_labels)
        net.SaveModel(path_to_model)
        PlotLoss(history)

    
if __name__ == '__main__':
    main()