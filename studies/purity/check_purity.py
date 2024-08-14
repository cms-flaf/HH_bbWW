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

# vis_b1_p4 = vector.zip({'pt': branches['genb1_vis_pt'],\
#                         'eta': branches['genb1_vis_eta'],\
#                         'phi': branches['genb1_vis_phi'],\
#                         'mass': branches['genb1_vis_mass']})

# vis_b2_p4 = vector.zip({'pt': branches['genb2_vis_pt'],\
#                         'eta': branches['genb2_vis_eta'],\
#                         'phi': branches['genb2_vis_phi'],\
#                         'mass': branches['genb2_vis_mass']})

has_at_least_2_reco_bjets = (ak.count_nonzero(jet_hadrFlav == 5, axis=1) >= 2)

# jets_in_accep = ((vis_b1_p4.pt > 20) & (np.abs(vis_b1_p4.eta) < 2.5))\
#               & ((vis_b2_p4.pt > 20) & (np.abs(vis_b2_p4.eta) < 2.5)) & has_at_least_2_b_reco_jets

jets_in_accep = has_at_least_2_reco_bjets

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

presel = jets_in_accep & both_lep_correct

jet_hadrFlav = jet_hadrFlav[presel]
jet_btag = jet_btag[presel]
jet_trueBjetTag = jet_trueBjetTag[presel]

sort_by_btag = ak.argsort(jet_btag)[::, ::-1]
jet_btag = jet_btag[sort_by_btag]
jet_hadrFlav = jet_hadrFlav[sort_by_btag]
jet_trueBjetTag = jet_trueBjetTag[sort_by_btag]

# f1 = jet_hadrFlav[:, 0]
# f2 = jet_hadrFlav[:, 1]
# both_are_b_jets = (f1 == 5) & (f2 == 5)
both_are_b_jets = (jet_trueBjetTag[:, 0]) & (jet_trueBjetTag[:, 1])

purity = ak.count_nonzero(both_are_b_jets)/ak.count(both_are_b_jets)

print(f"Sorted by: {args.sort_by}")
print(f"Purity: {purity:.3f}")