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
    ROOT.gROOT.ProcessLine('#include "include/MiscTools.hpp"')

    df = ROOT.RDataFrame("Events", input_file)
    df = df.Filter("ncentralJet > 4", "At least 4 jets for resolved topology and SL channel")
    df = df.Filter("(genb1_vis_pt > 0.0) && (genb2_vis_pt > 0.0) && (genV2prod1_vis_pt > 0.0) && (genV2prod2_vis_pt > 0.0)", "All quarks have gen match")
    
    df = df.Define("bq1_p4", "CreateP4(genb1_pt, genb1_eta, genb1_phi, genb1_mass)")
    df = df.Define("bq2_p4", "CreateP4(genb2_pt, genb2_eta, genb2_phi, genb2_mass)")
    df = df.Define("b_quarks_dR", "bq1_p4.DeltaR(bq2_p4)")
    df = df.Filter("b_quarks_dR > 0.4", "Resolved b quarks")

    df = df.Define("lq1_p4", "CreateP4(genV2prod1_pt, genV2prod1_eta, genV2prod1_phi, genV2prod1_mass)")
    df = df.Define("lq2_p4", "CreateP4(genV2prod2_pt, genV2prod2_eta, genV2prod2_phi, genV2prod2_mass)")
    df = df.Define("light_quarks_dR", "lq1_p4.DeltaR(lq2_p4)")
    df = df.Filter("light_quarks_dR > 0.4", "Resolved light quarks")

    print(f"dataset contains {df.Count().GetValue()} events")


if __name__ == '__main__':
    main()