import ROOT
import argparse
import numpy as np
import time
from hmeVariables import GetHMEVariables


def main():
    parser = argparse.ArgumentParser(prog="HME", description="Runs HME")
    parser.add_argument("file", type=str, help="Input file")
    parser.add_argument("channel", type=str, help="Channel (DL or SL)")
    parser.add_argument(
        "mod", type=int, help="Value modulo which event should be selected"
    )
    parser.add_argument("val", type=int, help="Value for event selection")

    args = parser.parse_args()
    input_file = args.file
    mod = args.mod
    val = args.val
    channel = args.channel

    if channel not in ["SL", "DL"]:
        raise RuntimeError(
            f"Attempting to evaluate HME for wrong channel {channel}, allowed options are SL or DL"
        )

    ROOT.gROOT.SetBatch(True)
    ROOT.EnableImplicitMT(8)

    start = time.perf_counter()
    df = ROOT.RDataFrame("Events", input_file)
    print(f"Total events: {df.Count().GetValue()}")
    df = df.Filter(f"event % {mod} == {val}", "Evaluation selection")
    if channel == "DL":
        df = df.Define(
            "has_necessary_inputs",
            "return (ncentralJet >= 2 && lep1_pt > 0.0 && lep2_pt > 0.0);",
        )
    elif channel == "SL":
        df = df.Define(
            "has_necessary_inputs", "return (ncentralJet >= 4 && lep1_pt > 0.0);"
        )

    df = GetHMEVariables(df, channel)
    df = df.Define(
        "hme_mass", "return hme_output[static_cast<size_t>(HME::EstimOut::mass)];"
    )

    c1 = ROOT.TCanvas("c1", "c1")
    c1.SetGrid()
    hist = df.Histo1D(("hme_mass", "HME X->HH mass", 100, -10, 2000), "hme_mass")
    hist.GetXaxis().SetTitle("mass, [GeV]")
    hist.GetXaxis().SetTitle("Count")
    hist.Draw()
    c1.SaveAs(f"hme_{channel}.png")

    end = time.perf_counter()
    print(f"Execution time: {(end - start):.2f}s")


if __name__ == "__main__":
    main()
