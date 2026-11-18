#!/usr/bin/env python3
"""The booking of this analysis' own configuration (config/global.yaml, config/plot/histograms.yaml,
config/sub_regions/), not of a stand-in: every flavour's sub-regions expand; the single-lepton window
flavour and the CI flavour stay within sub_regions_max_key_vars with every configured channel and
category; the SL HME sidebands lie next to their windows, on the DeepHME_mass grid, are at least 50 GeV
wide and missing only where they would leave [240, 2000] GeV (the DeepHME turn-on below); the SL
sub-regions book only the binary DNN of their own mass; the CI windows leave room for no, one and two
lower sidebands. Plain Python.
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
SB_REGIONS = (
    "SR_SL",
    "SR_SL_Fakeable",
)  # the fake-estimate regions: window and sidebands
CR_REGIONS = tuple(f"CR_SL_{c}" for c in ("TT", "ST", "WJets", "DY", "H", "VV"))
WINDOW_REGIONS = SB_REGIONS + CR_REGIONS
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
            bases = WINDOW_REGIONS if s["sub_region"] == "HMESub" else SB_REGIONS
            self.assertIn(s["base_region"], bases, s["name"])
            self.assertEqual({ch for _, ch in s["cuts"]} - {"e", "mu"}, set())
        for mass in masses:
            for reg in WINDOW_REGIONS:
                self.assertTrue(self.cuts("HMESub", reg, mass), f"{reg} {mass}")
            for reg in CR_REGIONS:
                for lab in SB_LABELS:
                    self.assertFalse(
                        self.cuts("HMESB" + lab, reg, mass), f"{reg} {mass}"
                    )

    def test_sidebands_next_to_the_window_on_the_grid(self):
        n_checked = 0
        for s in self.subs:
            if s["sub_region"] != "HMESub" or s["base_region"] not in SB_REGIONS:
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
                # below: L2 | L1 | window; above: window | H1 | H2; a side keeps those that fit,
                # nearest first, and stops at the first that would leave [240, 2000] GeV
                for near, d, labels in ((lo, -1, ("L1", "L2")), (hi, +1, ("H1", "H2"))):
                    x = near
                    for label in labels:
                        far = HB.far_edge(self.edges, x, d, 50.0)
                        fits = far is not None and 240.0 <= far <= 2000.0
                        self.assertEqual(label in side, fits, (what, label))
                        if not fits:
                            for rest in labels[labels.index(label) :]:
                                self.assertNotIn(rest, side, (what, rest))
                            break
                        self.assertEqual(side[label], tuple(sorted((x, far))), what)
                        x = far
                for label, (x, y) in side.items():
                    self.assertGreaterEqual(x, 240.0, (what, label))
                    self.assertIn(x, self.edges)
                    self.assertIn(y, self.edges)
                    self.assertGreaterEqual(y - x, 50.0, (what, label, x, y))
                    n_checked += 1
        self.assertGreater(n_checked, 0)

    def test_each_binary_dnn_at_its_own_mass(self):
        setup = setup_for(SL_FLAVOUR)
        keys = [
            ((ch, reg, cat), "true")
            for ch in ("e", "mu")
            for reg in WINDOW_REGIONS
            for cat in GLOBAL["categories"]
        ]
        n_blocks = 0
        for block in HB.booking(setup, keys, variables(SL_FLAVOUR)):
            if "columns" not in block:
                continue
            n_blocks += 1
            mass = int(re.search(r"_m(\d+)$", block["keys"][0][0][1]).group(1))
            self.assertEqual(block["variables"], [f"TwoStageDNN_M{mass}_binary"])
        self.assertEqual(n_blocks, len(HB.sub_regions(setup)))

    def test_signal_params_of_the_binary_dnns(self):
        for name, entry in HISTS.items():
            m = re.fullmatch(r"TwoStageDNN_M(\d+)_binary", name)
            if m:
                self.assertEqual(entry.get("signal_params"), {"mass": int(m.group(1))})


class Intervals(unittest.TestCase):
    """Every SL record's DeepHME_mass interval is the one its cut encodes (read back from the
    bin-index expression, an independent path); test/test_sl_fakes.py checks the cells the SL fake
    estimate builds from them."""

    def test_intervals_equal_the_cuts(self):
        edges = HB.axis_edges(HISTS["DeepHME_mass"]["x_bins"])
        n = 0
        for flavour in (SL_FLAVOUR, "CI"):
            for s in HB.sub_regions(setup_for(flavour)):
                if s["base_region"] not in SB_REGIONS:
                    continue
                self.assertEqual(sorted(s["intervals"]), sorted(s["cuts"]), s["name"])
                for key, (expr, _) in s["cuts"].items():
                    (got,) = s["intervals"][key]["DeepHME_mass"]
                    self.assertEqual(got, interval(expr, edges), (s["name"], key))
                    n += 1
        self.assertGreater(n, 0)


class CiWindows(unittest.TestCase):
    def test_no_one_and_two_lower_sidebands(self):
        subs = HB.sub_regions(setup_for("CI"))
        lower = {
            cat: sorted(
                s["sub_region"]
                for s in subs
                if s["base_region"] == "SR_SL"
                and s["sub_region"] in ("HMESBL1", "HMESBL2")
                and (cat, "e") in s["cuts"]
            )
            for cat in ("res2b", "recovery", "boosted")
        }
        self.assertEqual(
            lower,
            {"res2b": [], "recovery": ["HMESBL1"], "boosted": ["HMESBL1", "HMESBL2"]},
        )
        for s in subs:
            if s["base_region"] in SB_REGIONS:
                self.assertEqual(s["variables"], ["TwoStageDNN_M300_binary"], s["name"])


if __name__ == "__main__":
    unittest.main()
