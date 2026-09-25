#!/usr/bin/env python3
"""How much of a binning's apparent significance is an artefact of choosing it.

The edges are derived from the same MC that then defines the expected limit, so a search
free to put a boundary anywhere will put some of them on fluctuations, and the Z it
reports is optimistic by however much of that it found. autoMCStats does not undo this:
the MC-statistical nuisances widen the uncertainty on a bin's content, they do not know
that the bin's *edges* were chosen after looking at that content. Which means the bias
grows with how hard the search looks -- an exact partition DP over ~nx^2*ny^2/4 candidate
cells is structurally more exposed than a greedy scan over nx of them, and the comparison
between the two is not fair until this is measured.

This is the standard optimism bootstrap. Treat the observed MC as the population, draw
replicas by perturbing every bin by its own MC error, derive a binning on each replica,
and score that binning twice: on the replica it was derived from, and on the original.
The first is what the optimiser believes; the second is what the binning is worth on data
it did not get to fit. Their mean difference is the optimism, in Z^2.

    optimism  = <Z2(binning(replica), replica)> - <Z2(binning(replica), original)>
    corrected = Z2(binning(original), original) - optimism

Run it on every strategy being compared, because only the *difference* of the corrected
numbers is a reason to prefer one. A strategy that gains 3% of Z and carries 3% more
optimism has gained nothing.

    python3 Studies/BinningScan/bootstrap_bias.py --replicas 10
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.append(os.environ.get("ANALYSIS_PATH", "."))

from StatInference.common.tools import importROOT

ROOT = importROOT()

from StatInference.common.binning_core import get_hist, sum_hists
from StatInference.common import binning_dp as D
from StatInference.bin_opt_2d import rebin_2d as R


def perturb(hist, rng):
    """One replica of a histogram: every bin redrawn from N(content, MC error).

    Under/overflow included, because the outermost bins absorb them and a replica that
    left them fixed would be a replica of a different histogram.

    Bins with zero error are left alone rather than redrawn: an empty bin of a sample
    that simply has no events there is not a measurement with an uncertainty, and
    smearing it would invent background where the search would then find structure.
    Negative draws are kept -- a background that fluctuates below zero is exactly the
    case the positivity gate exists for, and clipping would hide how often the gates bind.
    """
    clone = hist.Clone(f"{hist.GetName()}_replica")
    clone.SetDirectory(0)
    values, variances = D._hist_arrays(hist)
    errors = np.sqrt(np.maximum(variances, 0.0))
    drawn = np.where(
        errors > 0, rng.normal(values, np.where(errors > 0, errors, 1.0)), values
    )
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    for bx in range(nx + 2):
        for by in range(ny + 2):
            clone.SetBinContent(bx, by, drawn[bx, by])
    return clone


def score_binning(cells, slices, mode):
    """Total Z^2 of a binning, evaluated against whichever cells are handed in.

    binning_objective() rather than a sum written out here, because it knows both
    layouts: DNN slices binned in HME, and an HME box binned in the DNN.
    """
    if slices is None:
        return 0.0
    return D.binning_objective(cells, slices, mode)[0]


BASE = "/eos/user/d/daebi/HH_bbWW/v2605a_DNNOutputs_v3/Hists_merged"
ERAS = ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
SIGNALS = ["XtoHHto2B2W_2L", "XtoHHto2Tau2B"]
BACKGROUNDS = ["TT", "DY", "ST", "VV"]
CHANNELS = ["muMu", "eMu", "eE"]
CATEGORIES = ["SR/res2b", "SR/boosted", "SR/recovery"]


def load(mass, channel, category):
    files = [
        ROOT.TFile.Open(f"{BASE}/{era}/DNN_m{mass}_vs_HME/DNN_m{mass}_vs_HME.root")
        for era in ERAS
    ]
    prefix = f"{channel}/{category}/"
    # Both decay modes, summed, as the production binning sees them.
    signal = sum_hists(
        [get_hist(f, prefix + f"{name}_{mass}") for f in files for name in SIGNALS]
    )
    backgrounds = {}
    for name in BACKGROUNDS:
        per_era = [get_hist(f, prefix + name) for f in files]
        if all(h is not None for h in per_era):
            backgrounds[name] = per_era
    return files, signal, backgrounds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--masses", type=str, default="300,400,500,600,700,800,1000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--binning",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="a strategy to compare, as a binning yaml, e.g. "
        "box=config/Datacards/binning_hmebox.yaml; repeatable, and the first two are "
        "the pair compared at the end. Without it, the built-in greedy and dp pair.",
    )
    parser.add_argument(
        "--channels", type=str, default=",".join(CHANNELS), help="comma-separated"
    )
    parser.add_argument(
        "--categories", type=str, default=",".join(CATEGORIES), help="comma-separated"
    )
    args = parser.parse_args()
    channels = args.channels.split(",")
    categories = args.categories.split(",")
    masses = [int(m) for m in args.masses.split(",")]

    defaults = dict(R.BINNING_DEFAULTS)
    common = dict(
        n_slices=4,
        bkg_per_bin=1.0,
        min_bin_bkg_each_neff=2.0,
        max_bins_per_slice=10,
        significance_mode="asimov",
    )
    builtin = [
        ("greedy", dict(defaults, strategy="greedy", **common)),
        (
            "dp",
            dict(
                defaults,
                strategy="dp",
                dp_max_iterations=10,
                dp_min_bin_gain=0.002,
                **common,
            ),
        ),
    ]
    if args.binning:
        strategies = []
        for spec in args.binning:
            name, _, path = spec.partition("=")
            strategies.append((name, R.load_binning_config(path)))
    else:
        strategies = builtin

    nominal = {name: 0.0 for name, _ in strategies}
    in_sample = {name: np.zeros(args.replicas) for name, _ in strategies}
    out_sample = {name: np.zeros(args.replicas) for name, _ in strategies}
    rng = np.random.default_rng(args.seed)
    started = time.time()
    n_problems = 0

    for mass in masses:
        for channel in channels:
            for category in categories:
                files, signal, backgrounds = load(mass, channel, category)
                if signal is None or not backgrounds:
                    for f in files:
                        f.Close()
                    continue
                n_problems += 1
                clean = D.build_cells(signal, backgrounds)
                for name, knobs in strategies:
                    nominal[name] += score_binning(
                        clean, R.discover_binning(signal, backgrounds, knobs), "asimov"
                    )
                for r in range(args.replicas):
                    rep_signal = perturb(signal, rng)
                    rep_backgrounds = {
                        n: [perturb(h, rng) for h in hs]
                        for n, hs in backgrounds.items()
                    }
                    rep_cells = D.build_cells(rep_signal, rep_backgrounds)
                    for name, knobs in strategies:
                        binning = R.discover_binning(rep_signal, rep_backgrounds, knobs)
                        in_sample[name][r] += score_binning(
                            rep_cells, binning, "asimov"
                        )
                        out_sample[name][r] += score_binning(clean, binning, "asimov")
                print(
                    f"  {channel:4s} {category:11s} MX={mass:4d} done "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )
                for f in files:
                    f.Close()

    print(
        f"\n{n_problems} problems, {args.replicas} replicas, "
        f"{time.time() - started:.0f}s\n"
    )
    header = (
        f"{'strategy':10s} {'Z nominal':>10} {'Z in-samp':>10} {'Z out-samp':>11} "
        f"{'optimism':>10} {'corrected Z':>12}"
    )
    print(header)
    corrected = {}
    for name, _ in strategies:
        optimism = in_sample[name].mean() - out_sample[name].mean()
        corrected[name] = max(nominal[name] - optimism, 0.0)
        print(
            f"{name:10s} {math.sqrt(nominal[name]):10.3f} "
            f"{math.sqrt(in_sample[name].mean()):10.3f} "
            f"{math.sqrt(out_sample[name].mean()):11.3f} "
            f"{optimism:10.2f} {math.sqrt(corrected[name]):12.3f}"
        )
    a, b = strategies[0][0], strategies[1][0]
    raw = math.sqrt(nominal[b] / nominal[a])
    adj = math.sqrt(corrected[b] / corrected[a]) if corrected[a] > 0 else float("nan")
    print(
        f"\n{b} over {a}: {raw:.4f} as measured, {adj:.4f} after correcting both.\n"
        "The second is the one that means anything -- the first credits each strategy "
        "with whatever fluctuations it managed to find."
    )
    # spread across replicas, so a difference can be read against its own noise
    for name, _ in strategies:
        per_replica = np.sqrt(in_sample[name]) - np.sqrt(out_sample[name])
        print(
            f"  {name:10s} per-replica optimism in Z: mean {per_replica.mean():.3f}, "
            f"sd {per_replica.std(ddof=1):.3f}"
        )


if __name__ == "__main__":
    main()
