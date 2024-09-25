#ifndef MISC_TOOLS_HPP
#define MISC_TOOLS_HPP

#include <numeric>

#include "TLorentzVector.h"

#include "Definitions.hpp"

RVecS CreateIndices(size_t sz, size_t start = 0)
{
    if (start != 0)
    {
        sz -= start;
    }
    RVecS indices(sz);
    std::iota(indices.begin(), indices.end(), start);
    return indices;
}

// use this function if branch is array and need to create an array of p4s for each event 
// (i.e. when input branch has "horizontal" structure)
RVecLV CreateP4(RVecD const& pt, RVecD const& eta, RVecD const& phi, RVecD const& mass, RVecS const& indices)
{
    RVecLV res;
    res.reserve(indices.size());
    for (auto idx: indices)
    {
        TLorentzVector p;
        p.SetPtEtaPhiM(pt[idx], eta[idx], phi[idx], mass[idx]);
        res.push_back(p);
    }
    return res;
}

// this function is used to create a new column: p4
// but branches from which p4 is constructed here are single numbers, not arrays 
// (i.e. when input branch has no "horizontal" arrays) 
inline TLorentzVector CreateP4(double pt, double eta, double phi, double mass)
{
    TLorentzVector res;
    res.SetPtEtaPhiM(pt, eta, phi, mass);
    return res;
}

inline TLorentzVector GetLeadBJetP4(RVecD const& pt, RVecD const& eta, RVecD const& phi, RVecD const& mass)
{
    if (pt[0] > pt[1]) 
    {
        return CreateP4(pt[0], eta[0], phi[0], mass[0]);
    }
    return CreateP4(pt[1], eta[1], phi[1], mass[1]);
}

inline TLorentzVector GetSublBJetP4(RVecD const& pt, RVecD const& eta, RVecD const& phi, RVecD const& mass)
{
    if (pt[1] > pt[0]) 
    {
        return CreateP4(pt[0], eta[0], phi[0], mass[0]);
    }
    return CreateP4(pt[1], eta[1], phi[1], mass[1]);
}

#endif