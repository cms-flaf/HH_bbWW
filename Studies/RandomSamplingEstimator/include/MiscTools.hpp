#ifndef MISC_TOOLS_HPP
#define MISC_TOOLS_HPP

#include "TLorentzVector.h"

// this function is used to create a new column: p4
// but branches from which p4 is constructed here are single numbers, not arrays (i.e. when input branch has no "horizontal")
inline TLorentzVector CreateP4(double pt, double eta, double phi, double mass)
{
    TLorentzVector res;
    res.SetPtEtaPhiM(pt, eta, phi, mass);
    return res;
}

#endif