#ifndef ESTIMATOR_TOOLS_HPP
#define ESTIMATOR_TOOLS_HPP

#include <optional>
#include <vector>
#include <memory>

#include "TLorentzVector.h"
#include "TRandom3.h"
#include "TH1.h"

#include "Definitions.hpp"
#include "RealQuadEqn.hpp"

inline constexpr int N_ITER = 100;
inline constexpr double TOL = 10e-9;

inline constexpr double HIGGS_MASS = 125.03;
inline constexpr double HIGGS_WIDTH = 0.004;

inline constexpr double MAX_MASS = 1500;
inline constexpr int N_BINS = 1500;

enum PhysObj { bj1, bj2, wj1, wj2, lep, met };

OptionalPair JetRescFact(TLorentzVector& j1, TLorentzVector& j2, std::unique_ptr<TH1F>& pdf, double mass, TRandom3& rg)
{
    if (j1.Pt() <= j2.Pt()) 
    {
        std::swap(j1, j2);
    }

    double c1 = pdf->GetRandom(&rg);

    double x1 = j2.M2();
    double x2 = 2*c1*(j1*j2);
    double x3 = c1*c1*j1.M2() - mass*mass;

    std::vector<double> solutions = QuadEqn<double>(x1, x2, x3).Solutions();

    double c2 = 0;
    if (solutions.empty())
    {
        return std::nullopt;
    }
    else if (solutions.size() == 1)
    {
        c2 = solutions[0];
    }
    else if (solutions.size() == 2)
    {
        c2 = solutions[0] > 0.0 ? solutions[0] : solutions[1];
    }

    if (std::abs(c2) <= TOL)
    {
        return std::nullopt;
    }
    
    return std::make_optional<std::pair<double, double>>(c1, c2);
}

std::optional<TLorentzVector> ComputeNu(TLorentzVector const& l, TLorentzVector const& j1, TLorentzVector const& j2, TLorentzVector const& met, double mh, double eta)
{
    TLorentzVector vis(l);
    vis += j1;
    vis += j2;

    double phi = met.Phi();
    //m_h^2 = (j + j + l + nu)^2
    double pt = (mh*mh - vis*vis)/(2*(vis.E()*std::cosh(eta) - vis.Px()*std::cos(phi) - vis.Py()*std::sin(phi) - vis.Pz()*std::sinh(eta)));

    if (std::isinf(pt) || std::isnan(pt) || pt < TOL)
    {
        return std::nullopt;
    }

    TLorentzVector v;
    v.SetPtEtaPhiM(pt, eta, phi, 0.0);
    return std::make_optional<TLorentzVector>(std::move(v));
}

OptionalPair EstimateMass(RVecLV const& particles, std::unique_ptr<TH1F>& pdf, TRandom3& rg, int evt)
{
    TLorentzVector b1 = particles[PhysObj::bj1];
    TLorentzVector b2 = particles[PhysObj::bj2];
    TLorentzVector j1 = particles[PhysObj::wj1];
    TLorentzVector j2 = particles[PhysObj::wj2];
    TLorentzVector l = particles[PhysObj::lep];
    TLorentzVector met = particles[PhysObj::met];

    rg.SetSeed(42);
    double mh = rg.Gaus(HIGGS_MASS, HIGGS_WIDTH);

    double W_dijet_mass = (j1 + j2).M();

    auto res_mass = std::make_unique<TH1F>("X_mass", Form("X->HH mass: event %d", evt), N_BINS, 0.0, MAX_MASS);
    
    int failed_iter = 0;
    for (int i = 0; i < N_ITER; ++i)
    {
        double eta = rg.Uniform(-6, 6);
        OptionalPair b_jet_resc = JetRescFact(b1, b2, pdf, mh, rg);
        if (b_jet_resc)
        {
            assert(b1.Pt() >= b2.Pt());
            auto [c1, c2] = b_jet_resc.value();
            TLorentzVector bb1 = c1*b1;
            TLorentzVector bb2 = c2*b2;
            double met_corr_px = met.Px() - (c1 - 1)*b1.Px() - (c2 - 1)*b2.Px();
            double met_corr_py = met.Py() - (c1 - 1)*b1.Py() - (c2 - 1)*b2.Py();

            TLorentzVector met_corr(met_corr_px, met_corr_py, 0.0, 0.0);
            std::optional<TLorentzVector> nu = ComputeNu(l, j1, j2, met_corr, mh, eta);

            if (nu)
            {
                TLorentzVector tmp(l);
                tmp += nu.value();

                double W_lep_mass = tmp.M();
                if (W_lep_mass + W_dijet_mass > mh)
                {
                    ++failed_iter;
                    continue;
                } 

                tmp += j1;
                tmp += j2;  
                tmp += bb1;
                tmp += bb2;

                double X_mass = tmp.M();
                res_mass->Fill(X_mass);
            }
            else
            {
                ++failed_iter;
                continue;
            }
        }
        else
        {
            ++failed_iter;
            continue;
        }
    }

    if (res_mass->GetEntries())
    {
        int binmax = res_mass->GetMaximumBin(); 
        double estimated_mass = res_mass->GetXaxis()->GetBinCenter(binmax);
        double success_rate = 1.0 - static_cast<double>(failed_iter)/N_ITER;
        return std::make_optional<std::pair<double, double>>(estimated_mass, success_rate);
    }

    return std::nullopt;
}

size_t Test(RVecLV const& input)
{
    return input.size();
}

double UseHist(std::unique_ptr<TH1F>& h)
{
    return h->GetRandom();
}

#endif