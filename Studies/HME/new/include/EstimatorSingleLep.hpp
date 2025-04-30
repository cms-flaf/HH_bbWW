#ifndef ESTIM_SL
#define ESTIM_SL

#include "EstimatorBase.hpp"
#include "Constants.hpp"
#include "Definitions.hpp"
#include "EstimatorUtils.hpp"
#include "EstimatorTools.hpp"

namespace HME 
{
    class EstimatorSingleLep final : public EstimatorBase
    {
        public:
        explicit EstimatorSingleLep(TString const& pdf_file_name);
        ~EstimatorSingleLep() override = default;

        ArrF_t<ESTIM_OUT_SZ> EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label) override;
        OptArrF_t<ESTIM_OUT_SZ> EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id) override;
    };

    
    EstimatorSingleLep::EstimatorSingleLep(TString const& pdf_file_name)
    {
        m_pdf_1d.resize(pdf1d_sl_names.size());
        m_pdf_2d.resize(pdf2d_sl_names.size());

        TFile* pf = TFile::Open(pdf_file_name);
        Get1dPDFs(pf, m_pdf_1d, Channel::SL);
        Get2dPDFs(pf, m_pdf_2d, Channel::SL);
        pf->Close();
    }


    ArrF_t<ESTIM_OUT_SZ> EstimatorSingleLep::EstimateCombination(VecLVF_t const& particles, ULong64_t evt_id, TString const& comb_label)
    {
        ArrF_t<ESTIM_OUT_SZ> res{};
        std::fill(res.begin(), res.end(), -1.0f);

        LorentzVectorF_t const& bj1 = particles[static_cast<size_t>(ObjSL::bj1)];
        LorentzVectorF_t const& bj2 = particles[static_cast<size_t>(ObjSL::bj2)];
        LorentzVectorF_t const& lj1 = particles[static_cast<size_t>(ObjSL::lj1)];
        LorentzVectorF_t const& lj2 = particles[static_cast<size_t>(ObjSL::lj2)];
        LorentzVectorF_t const& lep = particles[static_cast<size_t>(ObjSL::lep)];
        LorentzVectorF_t const& met = particles[static_cast<size_t>(ObjSL::met)];

        UHist_t<TH1F>& pdf_b1 = m_pdf_1d[static_cast<size_t>(PDF1_sl::b1)];
        UHist_t<TH1F>& pdf_q1 = m_pdf_1d[static_cast<size_t>(PDF1_sl::q1)];
        UHist_t<TH2F>& pdf_mw1mw2 = m_pdf_2d[static_cast<size_t>(PDF2_sl::mw1mw2)];

        Float_t mh = m_prg->Gaus(HIGGS_MASS, HIGGS_WIDTH);
        
        [[maybe_unused]] TString tree_name = Form("evt_%llu_%s", evt_id, comb_label.Data());
        [[maybe_unused]] int failed_iter = 0;
        for (int i = 0; i < N_ITER; ++i)
        {
            Float_t unclust_dpx = m_prg->Gaus(0.0, MET_SIGMA);
            Float_t unclust_dpy = m_prg->Gaus(0.0, MET_SIGMA);

            auto bresc = ComputeJetResc(bj1, bj2, pdf_b1, mh);
            if (!bresc.has_value())
            {
                ++failed_iter;
                continue;
            }
            auto [c1, c2] = bresc.value();

            LorentzVectorF_t b1 = bj1;
            LorentzVectorF_t b2 = bj2;
            b1 *= c1;
            b2 *= c2;

            LorentzVectorF_t Hbb = b1 + b2;

            Float_t bjet_resc_dpx = -1.0*(c1 - 1)*bj1.Px() - (c2 - 1)*bj2.Px();
            Float_t bjet_resc_dpy = -1.0*(c1 - 1)*bj1.Py() - (c2 - 1)*bj2.Py();

            Double_t mw1 = 1.0;
            Double_t mw2 = 1.0;
            pdf_mw1mw2->GetRandom2(mw1, mw2, m_prg.get());

            std::vector<Float_t> masses;
            for (int control = 0; control < CONTROL; ++control)
            {
                bool lepW_onshell = control / 2;
                bool add_deta = control % 2;

                Float_t mWlep = lepW_onshell ? mw1 : mw2;
                Float_t mWhad = lepW_onshell ? mw2 : mw1;

                LorentzVectorF_t j1 = lj1;
                LorentzVectorF_t j2 = lj2;
                auto lresc = ComputeJetResc(j1, j2, pdf_q1, mWhad);
                if (!lresc.has_value())
                {
                    ++failed_iter;
                    continue;
                }
                
                auto [c3, c4] = lresc.value();
                j1 *= c3;
                j2 *= c4;
                
                Float_t ljet_resc_dpx = -1.0*(c3 - 1)*lj1.Px() - (c4 - 1)*lj2.Px();
                Float_t ljet_resc_dpy = -1.0*(c3 - 1)*lj1.Py() - (c4 - 1)*lj2.Py();

                Float_t met_corr_px = met.Px() + bjet_resc_dpx + ljet_resc_dpx + unclust_dpx;
                Float_t met_corr_py = met.Py() + bjet_resc_dpy + ljet_resc_dpy + unclust_dpy;

                Float_t met_corr_pt = std::sqrt(met_corr_px*met_corr_px + met_corr_py*met_corr_py);
                Float_t met_corr_phi = std::atan2(met_corr_py, met_corr_px);
                LorentzVectorF_t met_corr = LorentzVectorF_t(met_corr_pt, 0.0, met_corr_phi, 0.0);
                
                auto opt_nu = NuFromW(lep, met_corr, add_deta, mWlep);
                if (opt_nu)
                {
                    LorentzVectorF_t nu = opt_nu.value();
                    LorentzVectorF_t lepW = nu + lep;
                    LorentzVectorF_t hadW = j1 + j2;
                    LorentzVectorF_t Hww = lepW + hadW;
                    LorentzVectorF_t Xhh = Hww + Hbb;
                    Float_t mass = Xhh.M();
                    bool correct_hww_mass = (std::abs(mh - Hww.M()) < 1.0);
                    if (!correct_hww_mass)
                    {
                        continue;
                    }
                    masses.push_back(mass);
                } 
                else 
                {
                    continue;
                }
            }

            if (masses.empty())
            {
                ++failed_iter;
                continue;
            }

            Int_t num_sol = masses.size();
            Float_t weight = 1.0/num_sol;
            for (auto mass: masses)
            {
                m_res_mass->Fill(mass, weight);
            }
        }

        // combination data is returned in any case for further analysis
        Float_t integral = m_res_mass->Integral();
        if (m_res_mass->GetEntries() && integral > 0.0)
        {
            int binmax = m_res_mass->GetMaximumBin(); 
            res[static_cast<size_t>(EstimOut::mass)] = m_res_mass->GetXaxis()->GetBinCenter(binmax);
            res[static_cast<size_t>(EstimOut::peak_value)] = m_res_mass->GetBinContent(binmax);
            res[static_cast<size_t>(EstimOut::width)] = ComputeWidth(m_res_mass, Q16, Q84);
            res[static_cast<size_t>(EstimOut::integral)] = integral;
            return res;
        }
        return res;
    }

    OptArrF_t<ESTIM_OUT_SZ> EstimatorSingleLep::EstimateMass(VecLVF_t const& jets, VecLVF_t const& leptons, LorentzVectorF_t const& met, ULong64_t evt_id)
    {
        VecLVF_t particles(static_cast<size_t>(ObjSL::count));
        particles[static_cast<size_t>(ObjSL::lep)] = leptons[static_cast<size_t>(Lep::lep1)];
        particles[static_cast<size_t>(ObjSL::met)] = met;

        std::vector<ArrF_t<ESTIM_OUT_SZ>> results;
        std::vector<Float_t> integrals;
        size_t num_bjets = jets.size() < NUM_BEST_BTAG ? jets.size() : NUM_BEST_BTAG;

        std::unordered_set<size_t> used;
        for (size_t bj1_idx = 0; bj1_idx < num_bjets; ++bj1_idx)
        {
            used.insert(bj1_idx);
            for (size_t bj2_idx = bj1_idx + 1; bj2_idx < num_bjets; ++bj2_idx)
            {
                used.insert(bj2_idx);
                if (jets[bj1_idx].Pt() > jets[bj2_idx].Pt())
                {
                    particles[static_cast<size_t>(ObjSL::bj1)] = jets[bj1_idx];
                    particles[static_cast<size_t>(ObjSL::bj2)] = jets[bj2_idx];
                }
                else 
                {
                    particles[static_cast<size_t>(ObjSL::bj1)] = jets[bj2_idx];
                    particles[static_cast<size_t>(ObjSL::bj2)] = jets[bj1_idx];
                }

                for (size_t lj1_idx = 0; lj1_idx < jets.size(); ++lj1_idx)
                {
                    if (used.count(lj1_idx))
                    {
                        continue;
                    }
                    used.insert(lj1_idx);

                    for (size_t lj2_idx = lj1_idx + 1; lj2_idx < jets.size(); ++lj2_idx)
                    {
                        if (used.count(lj2_idx))
                        {
                            continue;
                        }
                        used.insert(lj2_idx);

                        if (jets[lj1_idx].Pt() > jets[lj2_idx].Pt())
                        {
                            particles[static_cast<size_t>(ObjSL::lj1)] = jets[lj1_idx];
                            particles[static_cast<size_t>(ObjSL::lj2)] = jets[lj2_idx];
                        }
                        else 
                        {
                            particles[static_cast<size_t>(ObjSL::lj1)] = jets[lj2_idx];
                            particles[static_cast<size_t>(ObjSL::lj2)] = jets[lj1_idx];
                        }
                        
                        TString comb_label = Form("b%zub%zuq%zuq%zu", bj1_idx, bj2_idx, lj1_idx, lj2_idx);
                        ArrF_t<ESTIM_OUT_SZ> comb_result = EstimateCombination(particles, evt_id, comb_label);
                        if (comb_result[static_cast<size_t>(EstimOut::mass)] > 0.0)
                        {
                            results.push_back(comb_result);
                            integrals.push_back(comb_result[static_cast<size_t>(EstimOut::integral)]);
                        }

                        // reset m_res_mass and use "clean" hist to build distribution for each combination
                        // else keep filling histogram and only reset it when moving to another event
                        ResetHist(m_res_mass);
                        used.erase(lj2_idx);
                    }
                    used.erase(lj1_idx);
                }
                used.erase(bj2_idx);
            }
            used.erase(bj1_idx);
        }

        if (!results.empty())
        {
            auto it = std::max_element(integrals.begin(), integrals.end());
            size_t choice = it - integrals.begin();           
            ResetHist(m_res_mass);
            return std::make_optional<ArrF_t<ESTIM_OUT_SZ>>(results[choice]);
        }
        ResetHist(m_res_mass);
        return std::nullopt;
    }
}

#endif