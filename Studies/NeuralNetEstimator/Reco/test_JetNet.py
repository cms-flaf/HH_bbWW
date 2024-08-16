from JetNet import JetNet
from Common.JetNetData import JetNetData
from Common.JetNet_utils import PlotPrediction


def main():
    jet_obs = ['px', 'py', 'pz', 'E', 'btagPNetB', 'PNetRegPtRawCorr', 'PNetRegPtRawCorrNeutrino', 'PNetRegPtRawRes']
    features = [f'centralJet{i}_{var}' for i in range(10) for var in jet_obs]
    labels = ['H_VV_px', 'H_VV_py', 'H_VV_pz', 'H_VV_E', 'X_mass']

    input_files = ["test_GluGlutotoRadiontoHHto2B2Vto2B2JLNu_450.root"]

    data = JetNetData(features, labels)
    data.ReadFiles(input_files)

    net = JetNet(features, labels)
    net.LoadModel("./models/JetNet_v2.keras")

    pred = net.Predict(data.data[features])
    mass = PlotPrediction(pred, data.data)
    
    
if __name__ == '__main__':
    main()
