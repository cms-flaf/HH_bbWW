import numpy as np
import awkward as ak
import uproot
import vector

fname = "DNN_dataset_2025-03-19-13-41-00/batchfile1.root"
# fname = "Radionto2L.root"
# fname = "TTto2L2Nu.root"

tree = uproot.open(f"{fname}:Events")
branches = tree.arrays(
    [
        "sample_type",
        "X_mass",
        "centralJet_hadronFlavour",
        "bb_mass",
        "bb_mass_PNetRegPtRawCorr",
        "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
        "ll_mass",
        "lep1_type",
        "lep2_type",
        "lep1_gen_kind",
        "lep2_gen_kind",
    ]
)


labels = branches["sample_type"]
X_mass = branches["X_mass"]

mbb = branches["bb_mass"]
mbb_PNetCorr = branches["bb_mass_PNetRegPtRawCorr"]
mbb_PNetCorrNeutrino = branches["bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino"]

lep1_type = branches["lep1_type"]
lep2_type = branches["lep2_type"]

lep1_gen_kind = branches["lep1_gen_kind"]
lep2_gen_kind = branches["lep2_gen_kind"]

mll = branches["ll_mass"]

hadronFlavour = branches["centralJet_hadronFlavour"]

lep1_gen_match_e = (lep1_type == 1) & ((lep1_gen_kind == 1) | (lep1_gen_kind == 3))
lep1_gen_match_mu = (lep1_type == 2) & ((lep1_gen_kind == 2) | (lep1_gen_kind == 4))
lep1_gen_match = (lep1_gen_match_e) | (lep1_gen_match_mu)

lep2_gen_match_e = (lep2_type == 1) & ((lep2_gen_kind == 1) | (lep2_gen_kind == 3))
lep2_gen_match_mu = (lep2_type == 2) & ((lep2_gen_kind == 2) | (lep2_gen_kind == 4))
lep2_gen_match = (lep2_gen_match_e) | (lep2_gen_match_mu)


sig_mask = (
    (labels == 1)
    & (hadronFlavour[:, 0] == 5)
    & (hadronFlavour[:, 1] == 5)
    & (lep1_gen_match)
    & (lep2_gen_match)
)
mbb_sig = mbb[sig_mask]
mbb_PNetCorr_sig = mbb_PNetCorr[sig_mask]
mbb_PNetCorrNeutrino_sig = mbb_PNetCorrNeutrino[sig_mask]
mll_sig = mll[sig_mask]

mbb_sig_m450 = mbb[(sig_mask) & (X_mass == 450)]

import matplotlib.pyplot as plt

plt.hist(mbb_sig, bins=100, range=(0.0, 300.0))
plt.savefig("sig_mbb.pdf")
plt.clf()
plt.hist(mbb_PNetCorr_sig, bins=100, range=(0.0, 300.0))
plt.savefig("sig_mbb_PNetCorr.pdf")
plt.clf()
plt.hist(mbb_PNetCorrNeutrino_sig, bins=100, range=(0.0, 300.0))
plt.savefig("sig_mbb_PNetCorrNeutrino.pdf")
plt.clf()
plt.hist(mll_sig, bins=100, range=(0.0, 200.0))
plt.savefig("sig_mll.pdf")
plt.clf()
plt.hist(mbb_sig_m450, bins=100, range=(0.0, 300.0))
plt.savefig("sig_mbb_m450.pdf")
plt.clf()

import os

# os.system("imgcat sig_mbb.pdf")
# os.system("imgcat sig_mbb_PNetCorr.pdf")
os.system("imgcat sig_mbb_PNetCorrNeutrino.pdf")
os.system("imgcat sig_mll.pdf")
os.system("imgcat sig_mbb_m450.pdf")


TT_mask = labels == 8

mbb_TT = mbb[TT_mask]
mbb_PNetCorr_TT = mbb_PNetCorr[TT_mask]
mbb_PNetCorrNeutrino_TT = mbb_PNetCorrNeutrino[TT_mask]
mll_TT = mll[TT_mask]

import matplotlib.pyplot as plt

plt.hist(mbb_TT, bins=100, range=(0.0, 300.0))
plt.savefig("TT_mbb.pdf")
plt.clf()
plt.hist(mbb_PNetCorr_TT, bins=100, range=(0.0, 300.0))
plt.savefig("TT_mbb_PNetCorr.pdf")
plt.clf()
plt.hist(mbb_PNetCorrNeutrino_TT, bins=100, range=(0.0, 300.0))
plt.savefig("TT_mbb_PNetCorrNeutrino.pdf")
plt.clf()
plt.hist(mll_TT, bins=100, range=(0.0, 200.0))
plt.savefig("TT_mll.pdf")
plt.clf()

import os

# os.system("imgcat TT_mbb.pdf")
# os.system("imgcat TT_mbb_PNetCorr.pdf")
os.system("imgcat TT_mbb_PNetCorrNeutrino.pdf")
os.system("imgcat TT_mll.pdf")


# Goal to have inside be 90% (then 80% for comparison) and have smallest width


def quant_bin(mbb_signal, window_size):

    low_arange = np.arange(0.0, 1.0 - window_size, 0.001)
    high_arange = low_arange + window_size

    low_quantile = np.quantile(mbb_signal, low_arange)
    high_quantile = np.quantile(mbb_signal, high_arange)

    width_array = high_quantile - low_quantile

    min_width_index = np.argmin(width_array)
    print(f"Low {low_quantile[min_width_index]}")
    print(f"High {high_quantile[min_width_index]}")

    print("Low quant")
    print(low_arange[min_width_index])

    print("High quant")
    print(high_arange[min_width_index])
