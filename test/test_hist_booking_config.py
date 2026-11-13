#!/usr/bin/env python3
"""The booking of this analysis' own configuration (config/global.yaml, config/plot/histograms.yaml,
config/sub_regions/), not of a stand-in: every flavour's sub-regions expand; the single-lepton window
flavour and the CI flavour stay within sub_regions_max_key_vars with every configured channel and
category; the SL HME sidebands lie next to their windows, on the DeepHME_mass grid and above the DeepHME
turn-on; each binary DNN of the two-stage DNN is filled at its own mass only. Plain Python.
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

SL_FLAVOUR = "Signal_Extraction_SL_window"
SL_REGIONS = ("SR_SL", "SR_SL_Fakeable")
SB_LABELS = ("L2", "L1", "H1", "H2")
# a sub-region name the datacard tools map to its base region (Analysis/EqualSignalBinning.py)
WINDOW_REGION = re.compile(r"_[A-Za-z0-9]+_m[0-9]+$")


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


def variables(flavour):
    return [
        v["name"] if isinstance(v, dict) else v
        for v in GLOBAL["histTuple_flavors"][flavour]["variables"]
    ]


def interval(expr, edges):
    """[low, high) in GeV of a single-interval DeepHME_mass bin cut (FLAF ``<var>_bin``, -1 the
    underflow)."""
    low = re.search(r"DeepHME_mass_bin >= (\d+)", expr)
    high = re.search(r"DeepHME_mass_bin < (\d+)", expr)
    return (
        edges[int(low.group(1))] if low else edges[0],
        edges[int(high.group(1))] if high else edges[-1],
    )


class Flavours(unittest.TestCase):
    def test_every_flavour_expands(self):
        for flavour, cfg in GLOBAL["histTuple_flavors"].items():
            if not cfg.get("sub_regions"):
                continue
            with self.subTest(flavour=flavour):
                subs = HB.sub_regions(setup_for(flavour))
                self.assertTrue(subs)
                for s in subs:
                    self.assertRegex(s["name"], WINDOW_REGION)

    def test_every_sub_region_base_region_is_booked(self):
        # an unbooked base region is still a region of the flavour (regions()), so HistPlotTask makes
        # plot targets for it, and HistPlotter finds no histograms there
        for flavour, cfg in GLOBAL["histTuple_flavors"].items():
            if not cfg.get("sub_regions"):
                continue
            with self.subTest(flavour=flavour):
                self.assertEqual(
                    sorted(HB.regions(setup_for(flavour))), sorted(cfg["regions"])
                )

    def test_within_the_cap_with_every_channel_and_category(self):
        for flavour in (SL_FLAVOUR, "CI"):
            with self.subTest(flavour=flavour):
                setup = setup_for(flavour)
                keys = [
                    ((ch, reg, cat), "true")
                    for ch in GLOBAL["channelDefinition"]
                    for reg in HB.regions(setup)
                    for cat in GLOBAL["categories"]
                ]
                HB.booking(setup, keys, variables(flavour))  # raises above the cap


class SingleLeptonWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subs = HB.sub_regions(setup_for(SL_FLAVOUR))
        cls.edges = HB.axis_edges(HISTS["DeepHME_mass"]["x_bins"])

    def cuts(self, family, region, mass):
        name = f"{region}_{family}_m{mass}"
        found = [s for s in self.subs if s["name"] == name]
        return found[0]["cuts"] if found else {}

    def test_window_and_sidebands_of_every_mass_in_e_and_mu(self):
        masses = sorted({s["point"]["mass"] for s in self.subs})
        self.assertEqual(len(masses), 25)
        for s in self.subs:
            self.assertIn(s["base_region"], SL_REGIONS)
            self.assertEqual({ch for _, ch in s["cuts"]} - {"e", "mu"}, set())
        for mass in masses:
            for reg in SL_REGIONS:
                self.assertTrue(self.cuts("HMESub", reg, mass), f"{reg} {mass}")

    def test_sidebands_next_to_the_window_on_the_grid(self):
        n_checked = 0
        for s in self.subs:
            if s["sub_region"] != "HMESub":
                continue
            mass, reg = s["point"]["mass"], s["base_region"]
            for (cat, ch), (expr, _) in s["cuts"].items():
                lo, hi = interval(expr, self.edges)
                side = {
                    lab: self.cuts("HMESB" + lab, reg, mass).get((cat, ch))
                    for lab in SB_LABELS
                }
                side = {
                    lab: interval(c[0], self.edges)
                    for lab, c in side.items()
                    if c is not None
                }
                what = (reg, mass, cat, ch)
                # below: L2 | L1 | window; above: window | H1 | H2 (each side all or nothing)
                if "L1" in side:
                    (_, l2_hi), (l1_lo, l1_hi) = side["L2"], side["L1"]
                    self.assertEqual((l1_hi, l2_hi), (lo, l1_lo), what)
                else:
                    self.assertNotIn("L2", side, what)
                if "H1" in side:
                    (h1_lo, h1_hi), (h2_lo, _) = side["H1"], side["H2"]
                    self.assertEqual((h1_lo, h2_lo), (hi, h1_hi), what)
                else:
                    self.assertNotIn("H2", side, what)
                for label, (x, y) in side.items():
                    self.assertGreaterEqual(x, 240.0, (what, label))
                    self.assertIn(x, self.edges)
                    self.assertIn(y, self.edges)
                    one_bin = self.edges.index(y) - self.edges.index(x) == 1
                    self.assertTrue(y - x >= 50.0 or one_bin, (what, label, x, y))
                    n_checked += 1
        self.assertGreater(n_checked, 0)

    def test_each_binary_dnn_at_its_own_mass(self):
        setup = setup_for(SL_FLAVOUR)
        keys = [
            ((ch, reg, cat), "true")
            for ch in ("e", "mu")
            for reg in SL_REGIONS
            for cat in GLOBAL["categories"]
        ]
        for block in HB.booking(setup, keys, variables(SL_FLAVOUR)):
            if "columns" not in block:
                continue
            mass = int(re.search(r"_m(\d+)$", block["keys"][0][0][1]).group(1))
            self.assertEqual(
                block["variables"], ["DeepHME_mass", f"TwoStageDNN_M{mass}_binary"]
            )

    def test_signal_params_of_the_binary_dnns(self):
        for name, entry in HISTS.items():
            m = re.fullmatch(r"TwoStageDNN_M(\d+)_binary", name)
            if m:
                self.assertEqual(entry.get("signal_params"), {"mass": int(m.group(1))})


if __name__ == "__main__":
    unittest.main()
