import ROOT
import argparse
import numpy as np
import time
from hmeVariables import GetHMEVariables


def main():
	parser = argparse.ArgumentParser(prog='HME', description='Runs HME')
	parser.add_argument('file', type=str, help="Input file")
	parser.add_argument('channel', type=str, help="Channel (DL or SL)")
	parser.add_argument('mod', type=int, help="Value modulo which event should be selected")
	parser.add_argument('val', type=int, help="Value for event selection")

	args = parser.parse_args()
	input_file = args.file
	mod = args.mod
	val = args.val
	channel = args.channel

	if channel not in ["SL", "DL"]:
		raise RuntimeError(f"Attempting to evaluate HME for wrong channel {channel}, allowed options are SL or DL")

	ROOT.gROOT.SetBatch(True)
	ROOT.EnableImplicitMT(8)

	start = time.perf_counter()
	df = ROOT.RDataFrame("Events", input_file)
	print(f"Total events: {df.Count().GetValue()}")
	df = df.Filter(f"event % {mod} == {val}", "Evaluation selection")
	if channel == "DL":
		df = df.Filter(f"ncentralJet >= 2", "jets")
		df = df.Filter(f"lep1_pt > 0.0 && lep2_pt > 0.0", "leptons")
	elif channel == "SL":
		df = df.Filter(f"ncentralJet >= 4", "jets")
		df = df.Filter(f"lep1_pt > 0.0", "leptons")
	
	hme_events = df.Count().GetValue()
	print(f"HME events: {hme_events}")
	df = GetHMEVariables(df, channel)
	end = time.perf_counter()
	print(f"Execution time: {(end - start):.2f}s")


if __name__ == '__main__':
    main()
