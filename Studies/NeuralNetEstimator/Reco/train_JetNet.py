from JetNet import JetNet
from Common.JetNetData import JetNetData
from Common.JetNet_utils import PlotLoss, PlotPrediction


def main():
    jet_obs = ['px', 'py', 'pz', 'E', 'btagPNetB', 'PNetRegPtRawCorr', 'PNetRegPtRawCorrNeutrino', 'PNetRegPtRawRes']
    features = [f'centralJet{i}_{var}' for i in range(10) for var in jet_obs]
    labels = ['H_VV_px', 'H_VV_py', 'H_VV_pz', 'H_VV_E', 'X_mass']

    input_files = ["JetNetTrain_reco.root"]

    data = JetNetData(features, labels)
    data.ReadFiles(input_files)
    
    data.Shuffle()
    data.TrainTestSplit()        

    net = JetNet(features, labels)
    net.ConfigureModel(data.train_features.shape)
    history = net.Fit(data.train_features, data.train_labels)
    net.SaveModel("./models/")
    PlotLoss(history)

    
if __name__ == '__main__':
    main()
