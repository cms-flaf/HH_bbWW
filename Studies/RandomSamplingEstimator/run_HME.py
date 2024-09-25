import ROOT
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(prog='train_net', description='Runs Random Sampling Heavy Mass Estimator')
    parser.add_argument('file', type=str, help="Input file")
    parser.add_argument('mod', type=int, help="Value modulo which event should be selected")
    parser.add_argument('val', type=int, help="Value for event selection")

    args = parser.parse_args()
    input_file = args.file
    mod = args.mod
    val = args.val

    ROOT.gROOT.SetBatch(True)
    ROOT.EnableImplicitMT(8)

    ROOT.gROOT.ProcessLine('#include "include/EstimatorTools.hpp"')
    ROOT.gROOT.ProcessLine('#include "include/KinTools.hpp"')
    ROOT.gROOT.ProcessLine('#include "include/CombTools.hpp"')
    ROOT.gROOT.ProcessLine('#include "include/Definitions.hpp"')

    ROOT.gInterpreter.Declare("""auto file_pdf = std::make_unique<TFile>("pdf.root", "READ");""")
    ROOT.gInterpreter.Declare("""auto pdf = std::unique_ptr<TH1F>(static_cast<TH1F*>(file_pdf->Get("pdf")));""")
    ROOT.gInterpreter.Declare("""TRandom3 rg; rg.SetSeed(42);""")

    df = ROOT.RDataFrame("Events", input_file)
    print(f"Total events: {df.Count().GetValue()}")
    
    df = df.Filter(f"event % {mod} == {val}", "Evaluation selection")
    df = df.Filter("ncentralJet >= 4", "At least 4 jets for resolved topology and SL channel")
    
    df = df.Define("bq1_p4", "CreateP4(genb1_pt, genb1_eta, genb1_phi, genb1_mass)")
    df = df.Define("bq2_p4", "CreateP4(genb2_pt, genb2_eta, genb2_phi, genb2_mass)")
    df = df.Define("b_quarks_dR", "bq1_p4.DeltaR(bq2_p4)")
    df = df.Filter("b_quarks_dR > 0.4", "Resolved b quarks")

    df = df.Define("lq1_p4", "CreateP4(genV2prod1_pt, genV2prod1_eta, genV2prod1_phi, genV2prod1_mass)")
    df = df.Define("lq2_p4", "CreateP4(genV2prod2_pt, genV2prod2_eta, genV2prod2_phi, genV2prod2_mass)")
    df = df.Define("light_quarks_dR", "lq1_p4.DeltaR(lq2_p4)")
    df = df.Filter("light_quarks_dR > 0.4", "Resolved light quarks")

    df = df.Define("reco_lep_p4", "CreateP4(lep1_pt, lep1_eta, lep1_phi, lep1_mass)")
    df = df.Define("reco_met_p4", "CreateP4(PuppiMET_pt, 0.0, PuppiMET_phi, 0.0)")

    df = df.Define("reco_bj1_p4", "GetLeadBJetP4(centralJet_pt, centralJet_eta, centralJet_phi, centralJet_mass)")
    df = df.Define("reco_bj2_p4", "GetSublBJetP4(centralJet_pt, centralJet_eta, centralJet_phi, centralJet_mass)")

    df = df.Define("light_jet_indices", "CreateIndices(ncentralJet, 2)")
    df = df.Define("light_jets", "CreateP4(centralJet_pt, centralJet_eta, centralJet_phi, centralJet_mass, light_jet_indices)")
    df = df.Define("best_onshell_pair", "ChooseBestPair(light_jets, Onshell())")
    df = df.Define("best_offshell_pair", "ChooseBestPair(light_jets, Offshell())")

    print(f"HME events: {df.Count().GetValue()}")

    df = df.Define("input_onshell", """  RVecLV res;
                                         res.push_back(reco_bj1_p4);
                                         res.push_back(reco_bj2_p4);
                                         res.push_back(light_jets[best_onshell_pair.first]);
                                         res.push_back(light_jets[best_onshell_pair.second]);
                                         res.push_back(reco_lep_p4);
                                         res.push_back(reco_met_p4);
                                         return res; """)

    df = df.Define("input_offshell", """ RVecLV res;
                                         res.push_back(reco_bj1_p4);
                                         res.push_back(reco_bj2_p4);
                                         res.push_back(light_jets[best_offshell_pair.first]);
                                         res.push_back(light_jets[best_offshell_pair.second]);
                                         res.push_back(reco_lep_p4);
                                         res.push_back(reco_met_p4);
                                         return res; """)

    df = df.Define("hme_onshell", """ auto hme = EstimateMass(input_onshell, pdf, rg, event);
                                      if (hme)
                                      {{
                                          auto [mass, succ_rate] = hme.value();
                                          return mass;
                                      }}
                                      return -1.0; """)

    df = df.Define("hme_offshell", """ auto hme = EstimateMass(input_offshell, pdf, rg, event);
                                       if (hme)
                                       {{
                                           auto [mass, succ_rate] = hme.value();
                                           return mass;
                                       }}
                                       return -1.0; """)

    result = df.AsNumpy(["hme_onshell", "hme_offshell"])
    output = np.column_stack((np.array(result["hme_onshell"]), np.array(result["hme_offshell"])))
    np.savetxt("hme_output.txt", output, delimiter=' ', fmt='%10.5f')
    
    print("\nCutflow report:")
    df.Report().Print()

if __name__ == '__main__':
    main()
