#ifndef COMB_TOOLS_HPP
#define COMB_TOOLS_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

#include "Definitions.hpp"

template <typename T>
std::vector<int> sort_indices(T* v, int n) 
{
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] > v[i2];});
    return idx;
}

template <typename Func>
std::pair<int, int> ChooseBestPair(RVecLV const& jets, Func func)
{
    size_t sz = jets.size();
    if (sz == 2)
    {
        return {0, 1};
    }


    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(sz*(sz-1)/2);
    for (size_t i = 0; i < sz; ++i)
    {
        for (size_t j = 0; j < sz; ++j)
        {
            pairs.emplace_back(i, j);
        }
    }

    double min_metric = std::numeric_limits<double>::max();
    std::pair<int, int> res{-1, -1};
    for (auto const& p: pairs)
    {
        auto [i1, i2] = p;
        TLorentzVector const& jet1 = jets[i1];
        TLorentzVector const& jet2 = jets[i2];

        double metric = func(jet1, jet2);
        if (metric < min_metric)
        {
            min_metric = metric;
            res = p;
        }
    }

    return res;
}

struct Onshell
{
    inline double operator()(TLorentzVector const& p1, TLorentzVector const& p2) const 
    {
        return std::abs(80.0 - (p1 + p2).M());
    }
};

struct Offshell
{
    inline double operator()(TLorentzVector const& p1, TLorentzVector const& p2) const 
    {
        return std::abs(40.0 - (p1 + p2).M());
    }
};

#endif