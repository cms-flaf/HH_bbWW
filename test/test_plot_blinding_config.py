#!/usr/bin/env python3
"""The per-region blind_ranges of config/plot/histograms.yaml against the regions their variables are plotted
in: the base regions of every flavour that plots the variable and its booked sub-regions (the booking plugin,
Analysis/hist_booking.py), which include the datacard input regions and the CI windows. Every such region gets
its windows through FLAF's own rule (FLAF/Common/PlotBlinding.py) without an error; every region pattern
matches a plotted region (a misspelt pattern would leave its region to "*"); in the single-lepton window
flavour no region falls to "*", and the HME window of each mass keeps its 50 %-of-signal threshold.
Plain Python.
"""

import os
import re
import sys
import types
import unittest

import yaml

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo not in sys.path:
    sys.path.insert(0, repo)

from Analysis import hist_booking as HB  # noqa: E402
from FLAF.Common.PlotBlinding import OTHER_REGIONS, blind_windows  # noqa: E402

SL_FLAVOUR = "Signal_Extraction_SL_window"


def load(path):
    with open(os.path.join(repo, path)) as f:
        return yaml.safe_load(f)


GLOBAL = load("config/global.yaml")
HISTS = load("config/plot/histograms.yaml")


def setup_for(flavour):
    return types.SimpleNamespace(
        global_params=GLOBAL,
        histTuple_flavor=flavour,
        ana_path=repo,
        hists=types.SimpleNamespace(config_dict=HISTS),
    )


def plotted_regions():
    """{flavour: {variable: set of regions}}: the base regions for every variable of the flavour, the
    sub-regions for the variables their records book."""
    out = {}
    for flavour, cfg in GLOBAL["histTuple_flavors"].items():
        variables = [
            v["name"] if isinstance(v, dict) else v for v in cfg.get("variables") or []
        ]
        per_var = {v: set(cfg.get("regions") or []) for v in variables}
        for s in HB.sub_regions(setup_for(flavour)):
            for v in s["variables"] if s["variables"] is not None else variables:
                per_var.setdefault(v, set()).add(s["name"])
        out[flavour] = per_var
    return out


PER_REGION = {
    name: entry
    for name, entry in HISTS.items()
    if isinstance(entry, dict) and isinstance(entry.get("blind_ranges"), dict)
}


class PerRegionBlinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions = plotted_regions()

    def test_every_plotted_region_resolves(self):
        n = 0
        for flavour, per_var in self.regions.items():
            for var, regions in per_var.items():
                if var not in PER_REGION:
                    continue
                for region in sorted(regions):
                    with self.subTest(flavour=flavour, var=var, region=region):
                        blind_windows(var, var, PER_REGION[var], region)
                        n += 1
        self.assertGreater(n, 0)

    def test_every_pattern_matches_a_plotted_region(self):
        for var, entry in PER_REGION.items():
            plotted = set().union(*(pv.get(var, set()) for pv in self.regions.values()))
            for pattern in entry["blind_ranges"]:
                if pattern == OTHER_REGIONS:
                    continue
                with self.subTest(var=var, pattern=pattern):
                    self.assertTrue(
                        any(re.fullmatch(pattern, r) for r in plotted),
                        f"{var}: the blind_ranges pattern {pattern!r} matches no plotted region",
                    )

    def test_single_lepton_regions_have_their_own_key(self):
        for var, regions in self.regions[SL_FLAVOUR].items():
            if var not in PER_REGION:
                continue
            patterns = [
                p for p in PER_REGION[var]["blind_ranges"] if p != OTHER_REGIONS
            ]
            for region in regions:
                with self.subTest(var=var, region=region):
                    self.assertTrue(
                        any(re.fullmatch(p, region) for p in patterns), region
                    )

    def test_window_keeps_the_half_signal_threshold(self):
        for var, entry in PER_REGION.items():
            m = re.fullmatch(r"TwoStageDNN_M(\d+)_binary", var)
            if not m:
                continue
            window = f"SR_SL_HMESub_m{m.group(1)}"
            other = entry["blind_ranges"].get(OTHER_REGIONS)
            with self.subTest(var=var):
                ((lo, hi),) = blind_windows(var, var, entry, window)
                self.assertEqual(hi, 23.5)
                # the base region SR_SL holds the window: hidden at least from the same threshold
                ((lo_base, _),) = blind_windows(var, var, entry, "SR_SL")
                self.assertLessEqual(lo_base, lo)
                # regions of other flavours (CI) keep the window's threshold, as before the per-region form
                self.assertEqual(other, [lo, hi])


if __name__ == "__main__":
    unittest.main()
