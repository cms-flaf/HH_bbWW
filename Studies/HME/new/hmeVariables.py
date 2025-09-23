import ROOT
import os

ROOT.gROOT.ProcessLine(
    f'#include "{os.environ["ANALYSIS_PATH"]}/Studies/HME/new/include/Estimator.hpp"'
)
ROOT.gROOT.ProcessLine(
    f'#include "{os.environ["ANALYSIS_PATH"]}/Studies/HME/new/include/Constants.hpp"'
)
ROOT.gROOT.ProcessLine("TH1::AddDirectory(false)")
ROOT.gROOT.ProcessLine("TH2::AddDirectory(false)")
ROOT.gROOT.ProcessLine(
    f'auto estimator = std::make_unique<HME::Estimator>("{os.environ["ANALYSIS_PATH"]}/Studies/HME/new/pdf_sl.root", "{os.environ["ANALYSIS_PATH"]}/Studies/HME/new/pdf_dl.root");'
)


def GetHMEVariables(df, channel):
    df = df.Define(
        "jets",
        """HME::VecLVF_t res;
                           for (size_t i = 0; i < ncentralJet; ++i)
                           {{
                                res.emplace_back(centralJet_pt[i], centralJet_eta[i], centralJet_phi[i], centralJet_mass[i]);  
                            }}
                            return res;""",
    )

    if channel == "DL":
        df = df.Define(
            "leptons",
            """HME::VecLVF_t res;
                                    res.emplace_back(lep1_pt, lep1_eta, lep1_phi, lep1_mass);
                                    res.emplace_back(lep2_pt, lep2_eta, lep2_phi, lep2_mass);
                                    return res;""",
        )
    elif channel == "SL":
        df = df.Define(
            "leptons",
            """HME::VecLVF_t res;
                                    res.emplace_back(lep1_pt, lep1_eta, lep1_phi, lep1_mass);
                                    return res;""",
        )

    df = df.Define(
        "met",
        """HME::LorentzVectorF_t res(PuppiMET_pt, 0.0, PuppiMET_phi, 0.0);    
                          return res;""",
    )

    df = df.Define(
        "btags",
        """std::vector<Float_t> res;
                           for (size_t i = 0; i < ncentralJet; ++i)
                           {{
                                res.emplace_back(centralJet_btagPNetB[i]);  
                            }}
                            return res;""",
    )
    df = df.Define(
        "light_tags",
        """std::vector<Float_t> res;
                           for (size_t i = 0; i < ncentralJet; ++i)
                           {{
                                res.emplace_back(centralJet_btagPNetQvG[i]);  
                            }}
                            return res;""",
    )

    df = df.Define(
        "hme_output",
        f"""if (has_necessary_inputs)
                                    {{
                                        auto const& hme = estimator->EstimateMass(jets, leptons, met, btags, light_tags, event, HME::Channel::{channel});
                                        if (hme.has_value())
                                        {{
                                            return hme.value();
                                        }}
                                    }}
                                    std::array<Float_t, HME::ESTIM_OUT_SZ> hme_res{{}};
                                    for (size_t i = 0; i < HME::ESTIM_OUT_SZ; ++i)
                                    {{
                                        hme_res[i] = -1.0f;
                                    }}
                                    return hme_res;""",
    )
    return df
