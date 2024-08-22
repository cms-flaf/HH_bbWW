import uproot
import numpy as np
import vector
import awkward as ak
from statsmodels.stats.proportion import proportion_confint


class PurityChecker:
    def __init__(self, sort_by):
        self.sort_by = sort_by if 'centralJet_' in sort_by else 'centralJet_' + sort_by


    def ComputePurity(self, file_name, channel):
        file = uproot.open(file_name)
        tree = file['Events']
        branches = tree.arrays()

        jet_hadrFlav = branches['centralJet_hadronFlavour']
        jet_trueBjetTag = branches['centralJet_TrueBjetTag']
        jet_btag = branches[self.sort_by]

        has_at_least_2_reco_bjets = (ak.count_nonzero(jet_hadrFlav == 5, axis=1) >= 2)

        reco_lep1_type = branches['lep1_type']
        reco_lep2_type = branches['lep2_type']

        gen_lep1_type = branches['lep1_genLep_kind']
        gen_lep2_type = branches['lep2_genLep_kind']

        reco_lep1_mu = reco_lep1_type == 1
        reco_lep1_ele = reco_lep1_type == 0

        reco_lep2_mu = reco_lep2_type == 1
        reco_lep2_ele = reco_lep2_type == 0

        gen_lep1_mu = (gen_lep1_type == 2) | (gen_lep1_type == 4)
        gen_lep1_ele = (gen_lep1_type == 1) | (gen_lep1_type == 3)

        gen_lep2_mu = (gen_lep2_type == 2) | (gen_lep2_type == 4)
        gen_lep2_ele = (gen_lep2_type == 1) | (gen_lep2_type == 3)

        lep1_correct = (reco_lep1_mu & gen_lep1_mu) | (reco_lep1_ele & gen_lep1_ele)
        lep2_correct = (reco_lep2_mu & gen_lep2_mu) | (reco_lep2_ele & gen_lep2_ele)

        corr_lep_reco = lep1_correct
        if channel == 'dl':
            corr_lep_reco = corr_lep_reco & lep2_correct

        presel = has_at_least_2_reco_bjets & corr_lep_reco

        jet_btag = jet_btag[presel]
        jet_trueBjetTag = jet_trueBjetTag[presel]

        sort_by_btag = ak.argsort(jet_btag)[::, ::-1]
        jet_btag = jet_btag[sort_by_btag]
        jet_trueBjetTag = jet_trueBjetTag[sort_by_btag]

        both_are_b_jets = (jet_trueBjetTag[:, 0]) & (jet_trueBjetTag[:, 1])
        success_trials = ak.count_nonzero(both_are_b_jets)
        total_trials = ak.count(both_are_b_jets)
        purity = success_trials/total_trials
        low, high = proportion_confint(success_trials, total_trials, alpha=0.32, method='beta')

        masspoint = branches['X_mass'][0]

        return masspoint, purity, low, high
