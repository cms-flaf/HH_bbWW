import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import *

WorkingPointsParticleNet = {
    "Run3_2022": {"Loose": 0.047, "Medium": 0.245, "Tight": 0.6734},
    "Run3_2022EE": {"Loose": 0.0499, "Medium": 0.2605, "Tight": 0.6915},
    "Run3_2023": {"Loose": 0.0358, "Medium": 0.1917, "Tight": 0.6172},
    "Run3_2023BPix": {"Loose": 0.0359, "Medium": 0.1919, "Tight": 0.6133},
}


def createKeyFilterDict(global_params, period):
    filter_dict = {}
    filter_str = ""
    channels_to_consider = global_params["channels_to_consider"]
    categories = global_params["categories"]
    ### add custom categories eventually:
    custom_categories = []
    custom_categories_name = global_params.get(
        "custom_categories", None
    )  # can be extended to list of names
    if custom_categories_name:
        custom_categories = list(global_params.get(custom_categories_name, []))
        if not custom_categories:
            print("No custom categories found")
    ### regions
    custom_regions = []
    custom_regions_name = global_params.get(
        "custom_regions", None
    )  # can be extended to list of names, if for example adding QCD regions + other control regions
    if custom_regions_name:
        custom_regions = list(global_params.get(custom_regions_name, []))
        if not custom_regions:
            print("No custom regions found")
    all_categories = categories + custom_categories
    custom_subcategories = list(global_params.get("custom_subcategories", []))
    triggers_dict = global_params["triggers"]
    for ch in channels_to_consider:
        triggers_list = triggers_dict[ch]
        triggers_list_complete = [f"HLT_{trg}" for trg in triggers_list]
        triggers_str = "(" + " || ".join(triggers_list_complete) + ")"
        print(triggers_str)
        # if period in triggers_dict[ch].keys():
        #     triggers = triggers_dict[ch][period]
        for reg in custom_regions:
            for cat in all_categories:

                filter_base = f" ( (channelId == {global_params['channelDefinition'][ch]})  && {triggers_str} && {reg} && {cat} ) "
                if custom_subcategories:
                    for subcat in custom_subcategories:
                        # filter_base += f"&& {custom_subcat}"
                        filter_str = f"(" + filter_base + f" && {subcat}"
                        filter_str += ")"
                        key = (ch, reg, cat, subcat)
                        filter_dict[key] = filter_str
                else:
                    filter_str = f"(" + filter_base
                    filter_str += ")"
                    key = (ch, reg, cat)
                    filter_dict[key] = filter_str
    return filter_dict


def GetBTagWeight(global_cfg_dict, cat, applyBtag=False):
    btag_weight = "1"
    btagshape_weight = "1"
    if applyBtag:
        if global_cfg_dict["btag_wps"][cat] != "":
            btag_weight = f"weight_bTagSF_{btag_wps[cat]}_Central"
    else:
        if cat != "btag_shape" and cat != "boosted":
            btagshape_weight = "weight_bTagShape_Central"
    return f"{btag_weight}*{btagshape_weight}"


def GetWeight(channel, cat, boosted_categories):  # do you need all these args?
    # weights_to_apply = ["weight_base", "ExtraDYWeight"]
    weights_to_apply = ["weight_base"]
    total_weight = "*".join(weights_to_apply)
    for lep_index in [1, 2]:
        total_weight = f"{total_weight} * {GetLepWeight(lep_index)}"
    total_weight = f"{total_weight} * {GetTriggerWeight()}"
    total_weight = f"{total_weight} * weight_bTagShape_Central"
    return total_weight


def GetLepWeight(lep_index):
    weight_Ele = f"(lep{lep_index}_legType == static_cast<int>(Leg::e) ? weight_lep{lep_index}_EleSF_wp80iso_EleIDCentral : 1.0)"

    # Medium pT Muon SF
    weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? weight_lep{lep_index}_MuonID_SF_TightID_TrkCentral * weight_lep{lep_index}_MuonID_SF_LoosePFIso_TightIDCentral : 1.0)"

    # High pT Muon SF
    # weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? weight_lep{lep_index}_HighPt_MuonID_SF_HighPtIDCentral * weight_lep{lep_index}_HighPt_MuonID_SF_RecoCentral * weight_lep{lep_index}_HighPt_MuonID_SF_TightIDCentral : 1.0)"

    # No Muon SF
    # weight_Mu = f"(lep{lep_index}_legType == static_cast<int>(Leg::mu) ? 1.0 : 1.0)"

    return f"{weight_Mu} * {weight_Ele}"


def GetTriggerWeight():
    weight_MuTrg = f"(lep1_legType == static_cast<int>(Leg::mu) ? weight_lep1_TrgSF_singleIsoMu_Central : 1.0)"
    weight_EleTrg = f"(lep1_legType == static_cast<int>(Leg::e) ? weight_lep1_TrgSF_singleEleWpTight_Central : 1.0)"

    return f"{weight_MuTrg} * {weight_EleTrg}"


class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    def defineCutFlow(self):
        self.df = self.df.Define("cutflow", "int(0)")
        cutflow_cuts = ["event_selection", "OS_Iso", "Zveto || OppFlavor", "mbb_SR"]
        for i, cut in enumerate(cutflow_cuts):
            self.df = self.df.Redefine(
                "cutflow", f"{cut} && cutflow >= {i} ? cutflow+1 : cutflow"
            )

    def defineLeptonChannel(self):
        self.DefineAndAppend("SL", "channelId == 1 || channelId == 2")
        self.DefineAndAppend(
            "DL", "channelId == 11 || channelId == 12 || channelId == 22"
        )

    def defineCategories(self):
        self.DefineAndAppend("baseline", f"return true;")

        # Test boosted -> res2b -> recovery
        self.DefineAndAppend(
            "HbbCand_isValid", "(bjet1_isValid && bjet2_isValid) || fatbjet_isValid"
        )
        self.DefineAndAppend(
            "WhadCand_isValid", "(wjet1_isValid && wjet2_isValid) || fatwjet_isValid"
        )
        self.DefineAndAppend("inclusive", "HbbCand_isValid && (DL || WhadCand_isValid)")
        self.DefineAndAppend(
            "boosted", "inclusive && (fatbjet_isValid || fatwjet_isValid)"
        )
        self.DefineAndAppend("resolved", "inclusive && !boosted")
        self.DefineAndAppend("res2b", "resolved && bjet1_isBTagged && bjet2_isBTagged")
        self.DefineAndAppend("recovery", "resolved && !res2b && bjet1_isBTagged")

    def defineLeptonPreselection(self):
        self.df = self.df.Define(
            "leadingleppT_ele",
            "((lep1_legType == 1 && lep1_pt  > 32 ) || (lep2_legType == 1 && lep2_pt  > 32))",
        )
        self.df = self.df.Define(
            "leadingleppT_Mu",
            "((lep1_legType == 2 && lep1_pt  > 25 ) || (lep2_legType == 2 && lep2_pt  > 25))",
        )
        self.df = self.df.Define(
            "leadingleppT", "(leadingleppT_ele || leadingleppT_Mu)"
        )  # 32 need to be changed to 25 for DL channel once Double lepton trigger SF are integrated
        self.df = self.df.Define(
            "subleadleppT", "(lep2_legType < 1 || (lep1_pt > 10 && lep2_pt > 10))"
        )
        self.df = self.df.Define(
            "tightlep",
            "((lep1_legType == 2 && lep1_Muon_tightId == 1) || (lep1_legType == 1 && lep1_Electron_mvaIso_WP80 == 1)) && (lep2_legType < 1 || ((lep2_legType == 2 && lep2_Muon_tightId == 1 ) || (lep2_legType == 1 && lep2_Electron_mvaIso_WP80 == 1)))",
        )
        self.df = self.df.Define(
            "tightlep_Iso",
            # " (((lep1_legType == 1 && lep1_Electron_pfRelIso03_all < 0.15) || (lep1_legType == 2 && lep1_Muon_pfRelIso04_all < 0.15)) || ((lep2_legType < 1 ) || ((lep2_legType == 1 && lep2_Electron_pfRelIso03_all < 0.15) || (lep2_legType == 2 && lep2_Muon_pfRelIso04_all < 0.15)) ) )",
            " (((lep1_legType == 1) || (lep1_legType == 2 && lep1_Muon_pfRelIso04_all < 0.15)) || ((lep2_legType < 1 ) || ((lep2_legType == 1) || (lep2_legType == 2 && lep2_Muon_pfRelIso04_all < 0.15)) ) )",  # Remove any electron Iso since we use iso in the ID
        )
        self.df = self.df.Define(
            "Single_lep_trg",
            """
            (HLT_singleIsoMu && lep1_legType == 2 && lep1_HasMatching_singleIsoMu) || (HLT_singleEleWpTight && lep1_legType == 1 && lep1_HasMatching_singleEleWpTight) ||
            (HLT_singleIsoMu && lep2_legType == 2 && lep2_HasMatching_singleIsoMu) || (HLT_singleEleWpTight && lep2_legType == 1 && lep2_HasMatching_singleEleWpTight)
            """,
        )
        self.df = self.df.Define(
            "event_selection",
            "leadingleppT &&  subleadleppT && Single_lep_trg && tightlep && ( lep2_legType < 1 ||  diLep_mass > 12 )",
        )

    def defineQCDRegions(self):
        self.DefineAndAppend(
            "OS", "(lep2_legType < 1) || (lep1_charge*lep2_charge < 0)"
        )
        self.DefineAndAppend("SS", "!OS")
        self.DefineAndAppend("Iso", "tightlep_Iso")
        self.DefineAndAppend("AntiIso", f"!Iso")
        self.DefineAndAppend("OS_Iso", f"OS && Iso && event_selection")
        self.DefineAndAppend("SS_Iso", f"SS && Iso && event_selection")
        self.DefineAndAppend("OS_AntiIso", f"OS && AntiIso && event_selection")
        self.DefineAndAppend("SS_AntiIso", f"SS && AntiIso && event_selection")
        # MR
        self.DefineAndAppend(
            "mbbCR_Tight",
            "Single_lep_trg && "
            "tightlep && "
            "!(bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)",
        )
        self.DefineAndAppend(
            "mbbCR_AntiTight",
            "Single_lep_trg && "
            "!tightlep && "
            "!(bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150)",
        )

    def defineControlRegions(self):
        # Define Single Muon Control Region (W Region) -- Require Muon + High MT (>50)
        # Define Double Muon Control Region (Z Region) -- Require lep1 lep2 are opposite sign muons, and combined mass is within 10GeV of 91
        self.DefineAndAppend(
            "Zpeak",
            f"(lep1_legType == lep2_legType ) && (abs(diLep_mass - 91.1876) < 10)",
        )
        self.DefineAndAppend(
            "Zveto",
            # f"(lep1_legType == lep2_legType ) && (abs(diLep_mass - 91.1876) > 10)",
            f"(lep1_legType == lep2_legType ) && (diLep_mass < 70)",
        )

        self.DefineAndAppend("OppFlavor", f"(lep1_legType != lep2_legType)")

        self.DefineAndAppend("ZVeto_OS_Iso", f"(Zveto || OppFlavor) && OS_Iso")

        self.DefineAndAppend("ZVeto_SS_Iso", f"(Zveto || OppFlavor) && SS_Iso")

        self.DefineAndAppend("ZPeak_OS_Iso", f"(Zpeak || OppFlavor) && OS_Iso")

        self.DefineAndAppend(
            "TTbar_CR", f"OS_Iso && lep1_legType == lep2_legType && diLep_mass > 100 "
        )
        self.DefineAndAppend(
            "mbb_SR",
            f"bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino > 70 && bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino < 150",
        )
        self.DefineAndAppend(
            "Lep1Lep2Jet1Jet2_mass",
            f"(lep1_legType > 0 && lep2_legType > 0) ? Lep1Lep2Jet1Jet2_p4.mass() : 0.0",
        )
        self.DefineAndAppend(
            "Lep1Jet1Jet2_mass", f"(lep1_legType > 0) ? Lep1Jet1Jet2_p4.mass() : 0.0"
        )

    def addDYReweighting(self):
        self.DefineAndAppend(
            "ExtraDYWeight_ee_res2b", f"channelId == 11  && res2b ? 1.4 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight_ee_recovery", f"channelId == 11 && recovery ? 1.13 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight_mumu_res2b", f"channelId == 22 && res2b ? 1.39 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight_mumu_recovery", f"channelId == 22 && recovery ? 1.12 : 1.0"
        )
        self.DefineAndAppend(
            "ExtraDYWeight",
            f"ExtraDYWeight_ee_res2b * ExtraDYWeight_ee_recovery * ExtraDYWeight_mumu_res2b * ExtraDYWeight_mumu_recovery",
        )

    def calculateMT(self):
        self.df = self.df.Define(
            "MT_lep1", f"(lep1_legType > 0) ? Calculate_MT(lep1_p4, PuppiMET_p4) : 0.0"
        )
        self.df = self.df.Define(
            "MT_lep2", f"(lep2_legType > 0) ? Calculate_MT(lep1_p4, PuppiMET_p4) : 0.0"
        )
        self.df = self.df.Define(
            "MT_tot",
            f"(lep1_legType > 0 && lep2_legType > 0) ? Calculate_TotalMT(lep1_p4, lep2_p4, PuppiMET_p4) : 0.0",
        )

    def selectTrigger(self, trigger):
        self.df = self.df.Filter(trigger)

    def addCut(self, cut=""):
        if cut != "":
            self.df = self.df.Filter(cut)

    def defineTriggers(self):
        for ch in self.config["channelSelection"]:
            for trg in self.config["triggers"][ch]:
                trg_name = "HLT_" + trg
                self.colToSave.append(trg_name)
                if trg_name not in self.df.GetColumnNames():
                    print(f"{trg_name} not present in colNames")
                    self.df = self.df.Define(trg_name, "1")
        # singleTau_th_dict = self.config['singleTau_th']
        # singleMu_th_dict = self.config['singleMu_th']
        # singleEle_th_dict = self.config['singleEle_th']
        for trg_name, trg_dict in self.config["application_regions"].items():
            for key in trg_dict.keys():
                region_name = trg_dict["region_name"]
                region_cut = trg_dict["region_cut"].format()
                self.colToSave.append(region_name)
                if region_name not in self.df.GetColumnNames():
                    self.df = self.df.Define(region_name, region_cut)

    def DefineAndAppend(self, varToDefine, var_expression):
        self.df = self.df.Define(varToDefine, var_expression)
        self.colToSave.append(varToDefine)

    def __init__(self, df, config, period, colToSave=[], **kwargs):
        super(DataFrameBuilderForHistograms, self).__init__(df, **kwargs)
        self.config = config
        self.period = period
        self.colToSave = colToSave + ["channelId"]
        self.bTagWP = WorkingPointsParticleNet[period]["Medium"]
        self.bTagWP_Loose = WorkingPointsParticleNet[period][
            "Loose"
        ]  # wp should go to global config.


def defineAllP4(df):
    df = df.Define(f"SelectedFatJet_idx", f"CreateIndexes(SelectedFatJet_pt.size())")
    df = df.Define(
        f"SelectedFatJet_p4",
        f"GetP4(SelectedFatJet_pt, SelectedFatJet_eta, SelectedFatJet_phi, SelectedFatJet_mass, SelectedFatJet_idx)",
    )
    for idx in [0, 1]:
        df = Utilities.defineP4(df, f"lep{idx+1}")
    df = df.Define(
        f"centralJet_p4",
        f"GetP4(centralJet_pt, centralJet_eta, centralJet_phi, centralJet_mass)",
    )
    for met_var in ["PuppiMET"]:
        df = df.Define(
            f"{met_var}_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>({met_var}_pt,0.,{met_var}_phi,0.)",
        )
    return df


def AddDNNVariables(df):
    df = df.Define("HT", f"Sum(centralJet_pt)")

    df = df.Define("dR_dilep", f"ROOT::Math::VectorUtil::DeltaR(lep1_p4, lep2_p4)")
    df = df.Define(
        "dR_dibjet",
        f"ROOT::Math::VectorUtil::DeltaR(bjet1_p4, bjet2_p4)",
    )
    df = df.Define(
        "dR_dilep_dibjet",
        f"ROOT::Math::VectorUtil::DeltaR((lep1_p4+lep2_p4), (bjet1_p4+bjet2_p4))",
    )
    df = df.Define(
        "dR_dilep_dijet",
        f"(wjet1_isValid && wjet2_isValid) ? ROOT::Math::VectorUtil::DeltaR((lep1_p4+lep2_p4), (wjet1_p4+wjet2_p4)) : -100.",
    )
    df = df.Define(
        "dPhi_lep1_lep2", f"ROOT::Math::VectorUtil::DeltaPhi(lep1_p4,lep2_p4)"
    )
    df = df.Define(
        "dPhi_jet1_jet2",
        f"ROOT::Math::VectorUtil::DeltaPhi(bjet1_p4,bjet2_p4)",
    )
    df = df.Define(
        "dPhi_MET_dilep",
        f"ROOT::Math::VectorUtil::DeltaPhi(PuppiMET_p4,(lep1_p4+lep2_p4))",
    )
    df = df.Define(
        "dPhi_MET_dibjet",
        f"ROOT::Math::VectorUtil::DeltaPhi(PuppiMET_p4,(bjet1_p4+bjet2_p4))",
    )
    df = df.Define("min_dR_lep0_jets", f"MinDeltaR(lep1_p4, centralJet_p4)")
    df = df.Define("min_dR_lep1_jets", f"MinDeltaR(lep2_p4, centralJet_p4)")

    df = df.Define(
        "MT",
        f"(lep1_legType > 0 && lep2_legType > 0) ? Calculate_TotalMT(lep1_p4, lep2_p4, PuppiMET_p4) : -100.",
    )
    df = df.Define(
        "MT2",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2(lep1_p4, lep2_p4, bjet1_p4, bjet2_p4, PuppiMET_p4)) : -100.",
    )

    # Functional form of MT2 claculation
    df = df.Define(
        "MT2_ll",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(lep1_p4, lep2_p4, bjet1_p4 + bjet2_p4 + PuppiMET_p4, bjet1_p4.mass(), bjet2_p4.mass())) : -100.",
    )
    df = df.Define(
        "MT2_bb",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(bjet1_p4, bjet2_p4, lep1_p4 + lep2_p4 + PuppiMET_p4, 80.4, 80.4)) : -100.",
    )
    df = df.Define(
        "MT2_blbl",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(lep1_p4 + bjet1_p4, lep2_p4 + bjet2_p4, PuppiMET_p4, 0.0, 0.0)) : -100.",
    )
    df = df.Define(
        "MT2_blbl2",
        f"(lep1_legType > 0 && lep2_legType > 0) ? float(analysis::Calculate_MT2_func(lep1_p4 + bjet2_p4, lep2_p4 + bjet1_p4, PuppiMET_p4, 0.0, 0.0)) : -100.",
    )

    df = df.Define(
        "CosTheta_bb",
        f"(centralJet_pt.size() > 1) ? analysis::Calculate_CosDTheta(bjet1_p4, bjet2_p4) : -100.",
    )

    df = df.Define("diLep_p4", "(lep1_p4+lep2_p4)")
    df = df.Define(
        f"ll_mass",
        f"(lep1_legType > 0 && lep2_legType > 0) ? (lep1_p4+lep2_p4).mass() : -1.0",
    )
    df = df.Define(
        f"diLep_mass",
        f"(lep1_legType > 0 && lep2_legType > 0) ? (lep1_p4+lep2_p4).mass() : -1.0",
    )
    df = df.Define(f"pt_ll", "(lep1_p4+lep2_p4).Pt()")
    df = df.Define(
        "Lep1Lep2Jet1Jet2_p4",
        "(bjet1_isValid && bjet2_isValid) ? (lep1_p4+lep2_p4+bjet1_p4+bjet2_p4) : LorentzVectorM()",
    )
    df = df.Define(
        "Lep1Jet1Jet2_p4",
        "(bjet1_isValid && bjet2_isValid) ? (lep1_p4+bjet1_p4+bjet2_p4) : LorentzVectorM()",
    )
    # fixed PT values for mT_fix (decorrelated from lepton pt)
    # 35 GeV for muons, 30 GeV for electrons
    df = df.Define(
        "pT_fix", "(lep1_legType == static_cast<int>(Leg::mu) ? 35.0 : 30.0)"
    )
    # dphi between lepton and MET using VectorUtil
    df = df.Define(
        "dphi_fix", "abs(ROOT::Math::VectorUtil::DeltaPhi(lep1_p4, PuppiMET_p4))"
    )
    # fixed transverse mass
    df = df.Define("mT_fix", "sqrt(2.0 * pT_fix * PuppiMET_pt * (1.0 - cos(dphi_fix)))")

    return df


def defineJetSelections(df, isData):
    # Define vars to save
    jet_vars = [
        "p4",
        "pt",
        "phi",
        "eta",
        "mass",
        "btagPNetB",
        "idbtagPNetB",
        "rawFactor",
        "PNetRegPtRawCorr",
        "PNetRegPtRawCorrNeutrino",
    ]
    jet_mc_vars = ["hadronFlavour", "partonFlavour"]
    fatjet_vars = [
        "p4",
        "pt",
        "phi",
        "eta",
        "mass",
        "particleNetWithMass_HbbvsQCD",
        "particleNet_massCorr",
        "msoftdrop",
        "nConstituents",
        "tau1",
        "tau2",
        "tau3",
        "tau4",
    ]
    fatjet_mc_vars = ["hadronFlavour"]  # SelectedFatJet does not have partonFlavour
    if not isData:
        fatjet_vars = fatjet_vars + fatjet_mc_vars
        jet_vars = jet_vars + jet_mc_vars

    # First step is to decide Hbb boosted
    # Take FatJets, mask by BTag and msoftdrop, sort by BTag Score
    # If one exists, category is Hbb_Boosted

    df = df.Define(
        "FatBJet_Sel",
        "SelectedFatJet_particleNetWithMass_HbbvsQCD > 0.92 && SelectedFatJet_msoftdrop > 30",
    )
    df = df.Define("FatBJet_idx", "CreateIndexes(Sum(FatBJet_Sel))")
    df = df.Define(
        "FatBJet_idxSorted",
        "Take(ReorderObjects(SelectedFatJet_particleNetWithMass_HbbvsQCD[FatBJet_Sel], FatBJet_idx), min((int)FatBJet_idx.size(), 1))",
    )
    for var in fatjet_vars:
        df = df.Define(
            f"FatBJet_{var}",
            f"Take(SelectedFatJet_{var}[FatBJet_Sel], FatBJet_idxSorted)",
        )

    # Do not need a selection, we should just take the top 2 score Jets whether they pass the cut
    # df = df.Define("BJet_Sel", "centralJet_idbtagPNetB >= 1")
    df = df.Define("BJet_Sel", "centralJet_idbtagPNetB >= -1")
    df = df.Define("BJet_idx", "CreateIndexes(Sum(BJet_Sel))")
    df = df.Define(
        "BJet_idxSorted",
        "Take(ReorderObjects(centralJet_btagPNetB[BJet_Sel], BJet_idx), min((int)BJet_idx.size(), 2))",
    )
    for var in jet_vars:
        df = df.Define(
            f"BJet_{var}", f"Take(centralJet_{var}[BJet_Sel], BJet_idxSorted)"
        )

    df = df.Define("Nbjets", "BJet_pt.size()")
    df = df.Define("bjet1_isValid", "(Nbjets > 0)")
    df = df.Define("bjet2_isValid", "(Nbjets > 1)")

    df = df.Define("bjet1_isBTagged", "bjet1_isValid ? BJet_idbtagPNetB[0] >= 1 : 0")
    df = df.Define("bjet2_isBTagged", "bjet2_isValid ? BJet_idbtagPNetB[1] >= 1 : 0")

    df = df.Define("Nfatbjets", "FatBJet_pt.size()")
    df = df.Define("fatbjet_isValid", "(Nfatbjets > 0)")

    df = df.Define("Hbb_Boosted", "fatbjet_isValid")

    for var in jet_vars:
        df = df.Define(
            f"bjet1_{var}",
            f"bjet1_isValid ? BJet_{var}[0] : std::decay_t<decltype(BJet_{var})>::value_type()",
        )
        df = df.Define(
            f"bjet2_{var}",
            f"bjet2_isValid ? BJet_{var}[1] : std::decay_t<decltype(BJet_{var})>::value_type()",
        )

    for var in fatjet_vars:
        df = df.Define(
            f"fatbjet_{var}",
            f"fatbjet_isValid ? FatBJet_{var}[0] : std::decay_t<decltype(FatBJet_{var})>::value_type()",
        )

    df = df.Define(
        f"fatbjet_mass_PNetCorr",
        "fatbjet_isValid ? FatBJet_mass[0] * FatBJet_particleNet_massCorr[0] : std::decay_t<decltype(FatBJet_mass)>::value_type()",
    )

    df = df.Define("Nfatjets", "SelectedFatJet_pt.size()")
    df = df.Define("AllTrue_FatJet", "SelectedFatJet_pt > 0.0")
    # Create a mask removing FatJets that are chosen as the FatBJet
    df = df.Define(
        "FatWJet_HbbBoostedSel",
        "RemoveOverlaps(SelectedFatJet_p4, AllTrue_FatJet, FatBJet_p4, 0.1)",
    )
    # Create a mask removing FatJets that overlap within 0.8 dR of the Jets chosen as BJets
    df = df.Define(
        "FatWJet_HbbResolvedSel",
        "RemoveOverlaps(SelectedFatJet_p4, AllTrue_FatJet, BJet_p4, 0.8)",
    )

    df = df.Define(
        "FatWJet_Sel",
        "Hbb_Boosted ? FatWJet_HbbBoostedSel : FatWJet_HbbResolvedSel",
    )

    df = df.Define(
        "FatWJet_idx",
        "CreateIndexes(Sum(FatWJet_Sel))",
    )
    df = df.Define(
        "FatWJet_idxSorted",
        "ReorderObjects(SelectedFatJet_pt[FatWJet_Sel], FatWJet_idx)",
    )
    for var in fatjet_vars:
        df = df.Define(
            f"FatWJet_{var}",
            f"Take(SelectedFatJet_{var}[FatWJet_Sel], FatWJet_idxSorted)",
        )

    df = df.Define("Nfatwjets", "FatWJet_pt.size()")
    df = df.Define("fatwjet_isValid", "(Nfatwjets > 0) && !DL")
    df = df.Define(
        "fatwjet",
        "fatwjet_isValid ? FatWJet_p4[0] : LorentzVectorM()",
    )

    df = df.Define("Njets", "centralJet_pt.size()")
    df = df.Define("AllTrue_Jet", "centralJet_pt > 0.0")
    # Create a mask removing Jets that are chosen as the BJets
    df = df.Define(
        "WJet_HbbResolvedSel",
        "RemoveOverlaps(centralJet_p4, AllTrue_Jet, BJet_p4, 0.1)",
    )

    df = df.Define(
        "WJet_HbbBoostedSel",
        "RemoveOverlaps(centralJet_p4, AllTrue_Jet, FatBJet_p4, 0.8)",
    )

    df = df.Define(
        "WJet_Sel",
        "Hbb_Boosted ? WJet_HbbBoostedSel : WJet_HbbResolvedSel",
    )

    df = df.Define(
        "WJet_idx",
        "CreateIndexes(Sum(WJet_Sel))",
    )
    df = df.Define(
        "WJet_idxSorted",
        "ReorderObjects(centralJet_pt[WJet_Sel], WJet_idx)",
    )
    for var in jet_vars:
        df = df.Define(
            f"WJet_{var}",
            f"Take(centralJet_{var}[WJet_Sel], WJet_idxSorted)",
        )

    df = df.Define("wjet1_isValid", "(WJet_p4.size() > 0) && !DL")
    df = df.Define("wjet2_isValid", "(WJet_p4.size() > 1) && !DL")
    df = df.Define(
        "wjet1",
        "wjet1_isValid ? WJet_p4[0] : LorentzVectorM()",
    )
    df = df.Define(
        "wjet2",
        "wjet2_isValid ? WJet_p4[1] : LorentzVectorM()",
    )
    df = df.Define(
        "resolvedW_pt",
        "(wjet1_isValid && wjet2_isValid) ? (wjet1 + wjet2).Pt() : -1.0",
    )
    df = df.Define(
        "WJets_Boosted",
        "fatwjet_isValid && (resolvedW_pt < 0.0 || fatwjet.Pt() > resolvedW_pt)",
    )

    # Finally save the simple-name variables
    # bjet1, bjet2, fatbjet, wjet1, wjet2, fatwjet

    for var in jet_vars:
        df = df.Define(
            f"wjet1_{var}",
            f"wjet1_isValid ? WJet_{var}[0] : std::decay_t<decltype(WJet_{var})>::value_type()",
        )
        df = df.Define(
            f"wjet2_{var}",
            f"wjet2_isValid ? WJet_{var}[1] : std::decay_t<decltype(WJet_{var})>::value_type()",
        )

    for var in fatjet_vars:
        df = df.Define(
            f"fatwjet_{var}",
            f"fatwjet_isValid ? FatWJet_{var}[0] : std::decay_t<decltype(FatWJet_{var})>::value_type()",
        )

    # Lastly set mbb
    # PNet Corrections are currently incorrect
    # Using 1.0-rawFactor doesn't work since Jet corrections are already applied
    for bjet_idx in [1, 2]:
        df = df.Define(
            f"bjet{bjet_idx}_PNetRegPtRawCorr_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet{bjet_idx}_pt*(1.0-bjet{bjet_idx}_rawFactor)*bjet{bjet_idx}_PNetRegPtRawCorr, bjet{bjet_idx}_eta, bjet{bjet_idx}_phi, bjet{bjet_idx}_mass)",
        )
        df = df.Define(
            f"bjet{bjet_idx}_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet{bjet_idx}_pt*(1.0-bjet{bjet_idx}_rawFactor)*bjet{bjet_idx}_PNetRegPtRawCorr*bjet{bjet_idx}_PNetRegPtRawCorrNeutrino, bjet{bjet_idx}_eta, bjet{bjet_idx}_phi, bjet{bjet_idx}_mass)",
        )

    df = df.Define(
        f"bb_mass",
        "bjet1_isValid && bjet2_isValid ? (bjet1_p4+bjet2_p4).mass() : std::decay_t<decltype(BJet_mass)>::value_type()",
    )
    df = df.Define(
        f"bb_mass_PNetRegPtRawCorr",
        "bjet1_isValid && bjet2_isValid ? (bjet1_PNetRegPtRawCorr_p4+bjet2_PNetRegPtRawCorr_p4).mass() : std::decay_t<decltype(BJet_mass)>::value_type()",
    )
    df = df.Define(
        f"bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
        "bjet1_isValid && bjet2_isValid ? (bjet1_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4+bjet2_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino_p4).mass() : std::decay_t<decltype(BJet_mass)>::value_type()",
    )

    return df


def PrepareDfForHistograms(dfForHistograms, isData):
    dfForHistograms.defineLeptonChannel()
    dfForHistograms.df = defineAllP4(dfForHistograms.df)
    dfForHistograms.df = defineJetSelections(dfForHistograms.df, isData)
    dfForHistograms.df = AddDNNVariables(dfForHistograms.df)
    dfForHistograms.defineTriggers()
    dfForHistograms.defineLeptonPreselection()
    dfForHistograms.defineQCDRegions()
    dfForHistograms.defineControlRegions()
    dfForHistograms.defineCategories()
    dfForHistograms.addDYReweighting()
    dfForHistograms.calculateMT()
    dfForHistograms.defineCutFlow()
    return dfForHistograms
