#!/usr/bin/env python3
"""The blind_ranges of config/plot/histograms.yaml, given by region, channel and category (FLAF
Common/PlotKeys.py), against the plots their variables are made in: the channels (channelSelection) and
categories of global.yaml, and the regions of every flavour that plots the variable with its booked
sub-regions (the booking plugin, Analysis/hist_booking.py), which include the datacard input regions and the
CI windows. Every such plot gets its windows through FLAF's own rule (FLAF/Common/PlotBlinding.py) without an
error; every key names a plotted region, channel or category (a misspelt key would leave its plots to
"other"); in the single-lepton window flavour every region has its own key, and only the signal-like selection
(SR_SL, its HME window and side bins) is blinded; each "other" hides at least what every key beside it hides
(the inclusive category, the other channels and regions); SR_SL is hidden at least from the window's threshold
in each channel and category. The study model of the cross-mass signal filling (SL_CrossMassSignals) is the
only one that lifts signal_params.
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
from FLAF.Common.PlotBlinding import blind_windows  # noqa: E402
from FLAF.Common.PlotKeys import OTHER, leaves  # noqa: E402

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
CHANNELS = list(GLOBAL["channelSelection"])
CATEGORIES = list(GLOBAL["categories"]) + list(GLOBAL.get("boosted_categories") or [])


def windows(var, channel, category, region):
    return blind_windows(
        var, var, PER_REGION[var], channel=channel, category=category, region=region
    )


def lowest(ws):
    """The lowest threshold of windows that each run to the top of the axis; None for []."""
    return min(lo for lo, _ in ws) if ws else None


class PerPlotBlinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions = plotted_regions()

    def plots(self, flavour=None):
        for fl, per_var in self.regions.items():
            if flavour is not None and fl != flavour:
                continue
            for var, regions in per_var.items():
                if var not in PER_REGION:
                    continue
                for region in sorted(regions):
                    for channel in CHANNELS:
                        for category in CATEGORIES:
                            yield fl, var, channel, category, region

    def test_every_plot_resolves(self):
        n = 0
        for flavour, var, channel, category, region in self.plots():
            with self.subTest(
                flavour=flavour,
                var=var,
                channel=channel,
                category=category,
                region=region,
            ):
                windows(var, channel, category, region)
                n += 1
        self.assertGreater(n, 0)

    def test_every_key_names_a_plotted_region_channel_or_category(self):
        for var, entry in PER_REGION.items():
            regions = set().union(*(pv.get(var, set()) for pv in self.regions.values()))
            names = regions | set(CHANNELS) | set(CATEGORIES)
            for path, _ in leaves(entry["blind_ranges"]):
                for key in path:
                    if key == OTHER:
                        continue
                    with self.subTest(var=var, key=key):
                        self.assertIn(key, names, f"{var}: blind_ranges key {key!r}")

    def test_keys_are_regions_then_channels_then_categories(self):
        # a channel or category key outranks a region key at the same level (FLAF PlotKeys), so a
        # top-level "res2b" would override every region's res2b windows: the nesting is pinned
        for var, entry in PER_REGION.items():
            regions = set().union(*(pv.get(var, set()) for pv in self.regions.values()))
            for path, _ in leaves(entry["blind_ranges"]):
                for depth, key in enumerate(path):
                    if key == OTHER:
                        continue
                    allowed = (regions, set(CHANNELS), set(CATEGORIES))[depth]
                    with self.subTest(var=var, path=path):
                        self.assertIn(key, allowed)

    def test_single_lepton_regions_have_their_own_key(self):
        for var, regions in self.regions[SL_FLAVOUR].items():
            if var not in PER_REGION:
                continue
            for region in regions:
                with self.subTest(var=var, region=region):
                    self.assertIn(region, PER_REGION[var]["blind_ranges"])

    def test_only_the_signal_like_selection_is_blinded(self):
        # user, 2026-11-21: the SL window, its HME side bins and SR_SL only; SR_SL_Fakeable and CR_SL_* never
        for _, var, channel, category, region in self.plots(SL_FLAVOUR):
            if region == "SR_SL" or region.startswith(("SR_SL_HMESub_", "SR_SL_HMESB")):
                continue
            with self.subTest(
                var=var, channel=channel, category=category, region=region
            ):
                self.assertEqual(windows(var, channel, category, region), [])

    def test_other_hides_at_least_what_its_keys_hide(self):
        # the inclusive category, the channels and regions without a key of their own: the lowest
        # threshold of the plots beside them
        def check(var, node, path):
            if not isinstance(node, dict):
                return
            for key, value in node.items():
                check(var, value, path + (key,))
            if OTHER not in node:
                return
            other = node[OTHER]
            for key, value in node.items():
                if key == OTHER or not value:
                    continue
                for _, leaf in leaves(value):
                    if not leaf:
                        continue
                    with self.subTest(var=var, path=path, key=key):
                        self.assertTrue(
                            other, f"{var} {path}: other shows what {key} hides"
                        )
                        self.assertLessEqual(other[0], leaf[0])
                        self.assertEqual(other[1], 23.5)

        for var, entry in PER_REGION.items():
            ranges = dict(entry["blind_ranges"])
            # the top-level "other" covers the regions of other flavours, not the SL regions beside it
            ranges.pop(OTHER, None)
            check(var, ranges, ())

    def test_cross_mass_model_is_a_study_model(self):
        # the model of the cross-mass signal filling fills every binary DNN for every signal, and has no
        # background nor data; the models of the results keep the signal_params restriction
        models = load("config/phys_models.yaml")
        cross = models["SL_CrossMassSignals"]
        self.assertIs(cross["apply_signal_params"], False)
        self.assertEqual(cross.get("backgrounds") or [], [])
        self.assertNotIn("data", cross)
        for name, model in models.items():
            if name != "SL_CrossMassSignals":
                with self.subTest(model=name):
                    self.assertTrue(model.get("apply_signal_params", True))

    def test_window_bounds_sr_sl_and_other_flavours(self):
        for var in PER_REGION:
            m = re.fullmatch(r"TwoStageDNN_M(\d+)_binary", var)
            if not m:
                continue
            window = f"SR_SL_HMESub_m{m.group(1)}"
            all_window = []
            for channel in ("e", "mu"):
                for category in CATEGORIES:
                    with self.subTest(var=var, channel=channel, category=category):
                        w = windows(var, channel, category, window)
                        all_window += w
                        self.assertTrue(all(hi == 23.5 for _, hi in w))
                        if w:
                            # the base region SR_SL holds the window: hidden at least from its threshold
                            s = windows(var, channel, category, "SR_SL")
                            self.assertTrue(s)
                            self.assertLessEqual(lowest(s), lowest(w))
            with self.subTest(var=var):
                # the regions of other flavours (CI) keep the lowest threshold of the window
                self.assertTrue(all_window)
                self.assertEqual(
                    PER_REGION[var]["blind_ranges"][OTHER], [lowest(all_window), 23.5]
                )


if __name__ == "__main__":
    unittest.main()
