#!/usr/bin/env python3
"""Where the HME box's advantage over the 2D DNN-slice DP binning comes from.

The 2D binning sees both axes, so it is natural to expect it to do at least as well as a
single HME cut with the DNN binned inside. It does not, and this separates the candidate
reasons on the binner's own figure of merit (Asimov Z^2 with the background MC error
folded in, summed over bins), evaluated on identical cells for every binning:

  full1d    the DNN binned over the whole HME range -- no HME information at all
  box       the production box: one HME window, DNN binned inside (<= max_bins_per_slice)
  box3      the same window, the DNN allowed only 3 bins -- the DP's DNN resolution
  sidebands the two HME regions the box throws away, each DNN-binned the same way:
            an upper bound on what dropping them costs
  dp        the production 2D binning: 3 DNN slices, HME binned inside each

So box - box3 is what DNN resolution is worth, box - full1d what the HME cut is worth,
and sidebands what the box discards. These are statements about the proxy; the limits
themselves were measured separately with combine.

    python3 Studies/BinningScan/box_vs_dp_anatomy.py --masses 500,800
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.append(os.environ.get("ANALYSIS_PATH", "."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bootstrap_bias import load, CHANNELS, CATEGORIES  # noqa: E402
from StatInference.common import binning_dp as D  # noqa: E402
from StatInference.bin_opt_2d import rebin_2d as R  # noqa: E402


def box_value(cells, y0, y1, knobs):
    """Z^2 of the best DNN binning inside HME bins y0..y1, 0 if it cannot be binned."""
    if y1 < y0:
        return 0.0
    value, bins = R._box_value(cells, y0, y1, knobs)
    return value if bins is not None and math.isfinite(value) else 0.0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--masses", default="300,400,500,550,600,650,700,800,900,1000")
    p.add_argument("--dp-config", default="config/Datacards/binning_2d.yaml")
    p.add_argument("--box-config", default="config/Datacards/binning_hmebox.yaml")
    args = p.parse_args()
    dp_knobs = R.load_binning_config(args.dp_config)
    box_knobs = R.load_binning_config(args.box_config)
    box3_knobs = dict(box_knobs, max_bins_per_slice=3)

    keys = ("full1d", "box3", "box", "sidebands", "dp")
    tot = defaultdict(lambda: dict.fromkeys(keys, 0.0))
    top = defaultdict(
        lambda: [0.0, 0.0, 0, 0, 0]
    )  # z2 in top slice, z2 total, hme bins in top, box bins in top, n
    for mass in [int(m) for m in args.masses.split(",")]:
        for channel in CHANNELS:
            for category in CATEGORIES:
                files, signal, backgrounds = load(mass, channel, category)
                try:
                    if signal is None or not backgrounds:
                        continue
                    cells = D.build_cells(signal, backgrounds)
                    ny = cells.ny
                    box = R.discover_binning(signal, backgrounds, box_knobs)
                    dp = R.discover_binning(signal, backgrounds, dp_knobs)
                    if box is None or dp is None:
                        print(
                            f"  [skip] {channel} {category} {mass}: "
                            f"box={box is not None} dp={dp is not None}"
                        )
                        continue
                    y0, y1 = box[0]["y_range"]
                    y0, y1 = max(1, y0), min(ny, y1)
                    r = {
                        "full1d": box_value(cells, 1, ny, box_knobs),
                        "box3": box_value(cells, y0, y1, box3_knobs),
                        "box": D.binning_objective(cells, box, "asimov")[0],
                        "sidebands": box_value(cells, 1, y0 - 1, box_knobs)
                        + box_value(cells, y1 + 1, ny, box_knobs),
                        "dp": 0.0,
                    }
                    dp_total, dp_per = D.binning_objective(cells, dp, "asimov")
                    r["dp"] = dp_total
                    for group in (channel, "ALL"):
                        for k in keys:
                            tot[group][k] += r[k]
                    # the DP's best slice: how finely does each method resolve it in DNN?
                    i = int(np.argmax(dp_per))
                    xlo, xhi = dp[i]["x_range"]
                    box_bins_inside = sum(1 for a, b in box[0]["x_ranges"] if a >= xlo)
                    t = top[channel]
                    t[0] += dp_per[i]
                    t[1] += dp_total
                    t[2] += len(dp[i]["y_ranges"])
                    t[3] += box_bins_inside
                    t[4] += 1
                    print(
                        f"  {channel:4s} {category:11s} MX={mass:4d}  "
                        + "  ".join(f"{k}={math.sqrt(max(r[k],0)):.3f}" for k in keys)
                        + f"  | dp top slice x[{xlo},{xhi}] {len(dp[i]['y_ranges'])} HME bins,"
                        f" box {box_bins_inside}/{len(box[0]['x_ranges'])} DNN bins there",
                        flush=True,
                    )
                finally:
                    for f in files:
                        f.Close()

    print(
        "\nsqrt(sum Z^2) -- the binner's own figure of merit, identical cells for all"
    )
    print(
        f"{'':6s}"
        + "".join(f"{k:>11s}" for k in keys)
        + f"{'box/dp':>9s}{'+sb/dp':>9s}"
    )
    for group in sorted(tot):
        t = tot[group]
        z = {k: math.sqrt(max(t[k], 0)) for k in keys}
        both = math.sqrt(t["box"] + t["sidebands"])
        print(
            f"{group:6s}"
            + "".join(f"{z[k]:11.3f}" for k in keys)
            + f"{z['box']/z['dp']:9.3f}{both/z['dp']:9.3f}"
        )
    print("\nThe DP's highest-Z^2 slice, per channel:")
    for ch, (zt, za, nh, nb, n) in sorted(top.items()):
        print(
            f"  {ch:4s}: holds {100*zt/za:.1f}% of the DP's Z^2; the DP gives it "
            f"{nh/n:.1f} HME bins on average, and the box puts {nb/n:.1f} DNN bins "
            "inside the same DNN range"
        )


if __name__ == "__main__":
    main()
