import uproot
import numpy as np
import vector
import awkward as ak
import argparse


parser = argparse.ArgumentParser(prog='check_purity', description='Checks purity of the given sample')
parser.add_argument('file_name', type=str, help="File with input data")
parser.add_argument('sort_by', type=str, help="Name of the branch used to sort and select jets")

args = parser.parse_args()

file = uproot.open(args.file_name)
tree = file['Events']
branches = tree.arrays()

jet_p4 = vector.zip({'pt': branches['centralJet_pt'],\
                     'eta': branches['centralJet_eta'],\
                     'phi': branches['centralJet_phi'],\
                     'mass': branches['centralJet_mass']})

jet_hadrFlav = branches['centralJet_hadronFlavour']
jet_trueBjetTag = branches['centralJet_TrueBjetTag']

btag_name = args.sort_by
if 'centralJet_' not in btag_name:
    btag_name = 'centralJet_' + btag_name
jet_btag = branches[btag_name]

has_at_least_2_reco_bjets = (ak.count_nonzero(jet_hadrFlav == 5, axis=1) >= 2)

reco_lep1_type = branches['lep1_type']
reco_lep2_type = branches['lep2_type']

gen_lep1_type = branches['lep1_genLep_kind']
gen_lep2_type = branches['lep2_genLep_kind']

reco_lep1_mu = reco_lep1_type == 1
reco_lep1_ele = reco_lep1_type == 0

reco_lep2_mu = reco_lep2_type == 1
reco_lep2_ele = reco_lep2_type == 0

# see https://github.com/abolshov/Framework/blob/main/include/GenLepton.h#L125 for explanation of values on rhs
gen_lep1_mu = (gen_lep1_type == 2) | (gen_lep1_type == 4)
gen_lep1_ele = (gen_lep1_type == 1) | (gen_lep1_type == 3)

gen_lep2_mu = (gen_lep2_type == 2) | (gen_lep2_type == 4)
gen_lep2_ele = (gen_lep2_type == 1) | (gen_lep2_type == 3)

lep1_correct = (reco_lep1_mu & gen_lep1_mu) | (reco_lep1_ele & gen_lep1_ele)
lep2_correct = (reco_lep2_mu & gen_lep2_mu) | (reco_lep2_ele & gen_lep2_ele)
both_lep_correct = lep1_correct & lep2_correct

presel = has_at_least_2_reco_bjets & both_lep_correct

jet_btag = jet_btag[presel]
jet_trueBjetTag = jet_trueBjetTag[presel]

sort_by_btag = ak.argsort(jet_btag)[::, ::-1]
jet_btag = jet_btag[sort_by_btag]
jet_trueBjetTag = jet_trueBjetTag[sort_by_btag]

both_are_b_jets = (jet_trueBjetTag[:, 0]) & (jet_trueBjetTag[:, 1])
purity = ak.count_nonzero(both_are_b_jets)/ak.count(both_are_b_jets)

print(f"Sorted by: {args.sort_by}")
print(f"Purity: {purity:.3f}")