import ROOT
import argparse


def main():
    parser = argparse.ArgumentParser(prog='train_net', description='Runs Random Sampling Heavy Mass Estimator')
    parser.add_argument('file', type=str, help="Input file")

    args = parser.parse_args()
    input_file = args.file

    ROOT.gROOT.SetBatch(True)
    ROOT.EnableImplicitMT(8)
    ROOT.gROOT.ProcessLine('#include "include/EstimatorTools.hpp"')

    df = ROOT.RDataFrame("Events", input_file)
    df = df.Filter("ncentralJet > 4", "At least 4 jets for resolved topology")
    df = df.Filter("(genb1_vis_pt > 0.0) && (genb2_vis_pt > 0.0) && (genV2prod1_vis_pt > 0.0) && (genV2prod2_vis_pt > 0.0)", "All quarks have gen match")
    print(f"dataset contains {df.Count().GetValue()} events")


if __name__ == '__main__':
    main()