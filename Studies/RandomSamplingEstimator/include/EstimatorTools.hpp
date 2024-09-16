#ifndef ESTIMATOR_TOOLS_HPP
#define ESTIMATOR_TOOLS_HPP

#include <optional>
#include <vector>
#include <memory>

#include "TLorentzVector.h"
#include "TRandom3.h"
#include "TH1.h"

using OptionalPair = std::optional<std::pair<double, double>>;

OptionalPair JetRescFact(TLorentzVector& j1, TLorentzVector& j2, std::unique_ptr<TH1F>& pdf, double mass);
std::optional<TLorentzVector> ComputeNu(TLorentzVector const& l, TLorentzVector const& j1, TLorentzVector const& j2, TLorentzVector const& met, double mh, double eta);
OptionalPair EstimateMass(std::vector<TLorentzVector> const& particles, std::unique_ptr<TH1F>& pdf, TRandom3& rg, int evt);

#endif