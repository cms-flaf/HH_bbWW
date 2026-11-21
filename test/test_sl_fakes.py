#!/usr/bin/env python3
"""Analysis/SL_fakes.py, the single-lepton fake estimator, on synthetic merged histograms with a known
excess f in each region, through FLAF's own EstimateContext (FLAF Analysis/HistMergerFromHists.py)
over a stand-in of the merged file, with the real Fakes block of config/processes.yaml and the
binning of config/plot/histograms.yaml.

In every region, data = O + (1 + f) T, so f = (D - O)/T - 1 is known by construction, per DNN part
(below / from the mass's split). Needs ROOT: run inside the analysis environment (source env.sh).
"""

import os
import sys
import types
import unittest
from unittest import mock

import pytest
import yaml

pytestmark = pytest.mark.needs_root

ana_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ana_path not in sys.path:
    sys.path.insert(0, ana_path)
# FLAF's HistHelper reads ANALYSIS_PATH at import
os.environ.setdefault("ANALYSIS_PATH", ana_path)

import ROOT  # noqa: E402

from Analysis import SL_fakes  # noqa: E402
from FLAF.Analysis.HistMergerFromHists import EstimateContext  # noqa: E402

ROOT.gROOT.SetBatch(True)

with open(os.path.join(ana_path, "config", "processes.yaml")) as f:
    CONFIG = yaml.safe_load(f)["Fakes"]["data_driven"]

MASS = 600
VAR = f"TwoStageDNN_M{MASS}_binary"
SPLIT = CONFIG["split"][MASS]
with open(os.path.join(ana_path, "config", "plot", "histograms.yaml")) as f:
    _n, _range = yaml.safe_load(f)[VAR]["x_bins"].split("|")
NB = int(_n)
LO, HI = (float(x) for x in _range.split(":"))
CH, CAT = "mu", "res2b"
DATA, TT, W = "Data_Full", "TT", "W"
UNC = "U1"


def name(base, family):
    return f"{base}_{family}_m{MASS}"


# the HME intervals of the MX=600 res2b regions (as the plugin derives them from hme_SL_v2610.yaml)
WINDOW = (510.0, 680.0)
SIDEBANDS = {
    "L2": (410.0, 460.0),
    "L1": (460.0, 510.0),
    "H1": (680.0, 750.0),
    "H2": (750.0, 800.0),
}


def records(sidebands=SIDEBANDS):
    out = []
    for base in ("SR_SL", "SR_SL_Fakeable"):
        regions = {"HMESub": WINDOW}
        regions.update({"HMESB" + k: v for k, v in sidebands.items()})
        for family, interval in regions.items():
            out.append(
                {
                    "name": name(base, family),
                    "sub_region": family,
                    "base_region": base,
                    "point": {"mass": MASS},
                    "cuts": {(CAT, CH): ("x", [])},
                    "intervals": {(CAT, CH): {"DeepHME_mass": [interval]}},
                    "variables": [VAR],
                }
            )
    return out


def hist(values):
    """A histogram of the DNN axis: ``values(bin)`` for every bin, flows included."""
    h = ROOT.TH1D("h", "h", NB, LO, HI)
    h.SetDirectory(ROOT.nullptr)
    ROOT.SetOwnership(h, True)
    for b in range(NB + 2):
        v = values(b)
        h.SetBinContent(b, v)
        h.SetBinError(b, 0.1 * v)
    return h


def split_bin():
    return ROOT.TH1D("a", "a", NB, LO, HI).GetXaxis().FindFixBin(SPLIT)


class Store:
    """The merged histograms: {(process, region, unc, scale): hist}; f per (region, part)."""

    def __init__(self, f_of, tt_scale=1.0, tt_shift=None, tt_low=1.0):
        self.hists = {}
        b_split = split_bin()
        for base in ("SR_SL", "SR_SL_Fakeable"):
            for family in ["HMESub"] + ["HMESB" + k for k in SIDEBANDS]:
                region = name(base, family)
                for unc, scale in (("Central", "Central"), (UNC, "Up"), (UNC, "Down")):
                    k = 1.2 if scale == "Up" else 0.8 if scale == "Down" else 1.0
                    t = hist(
                        lambda b: tt_scale
                        * k
                        * (tt_low if b < b_split else 1.0)
                        * (10.0 + b % 7)
                    )
                    o = hist(lambda b: 3.0 + b % 3)
                    self.hists[(TT, region, unc, scale)] = t
                    self.hists[(W, region, unc, scale)] = o
                    if unc == "Central" and not (
                        base == "SR_SL" and family == "HMESub"
                    ):
                        f_lo, f_hi = f_of(base, family)
                        self.hists[(DATA, region, unc, scale)] = hist(
                            lambda b: (3.0 + b % 3)
                            + (1 + (f_lo if b < b_split else f_hi)) * (10.0 + b % 7)
                        )

    def reader(self, process, key_dir, unc, scale):
        """The merged file as FLAF's EstimateContext reads it (data already at nominal)."""
        ch, region, cat = key_dir
        assert (ch, cat) == (CH, CAT)
        if process == DATA and region == name("SR_SL", "HMESub"):
            raise AssertionError("the isolated window's data were read")
        return self.hists.get((process, region, unc, scale))


def context(store, unc_source="Central", var=VAR, model="Run3_Model", config=CONFIG):
    variations = (
        [("Central", "Central")]
        if unc_source == "Central"
        else [(unc_source, "Up"), (unc_source, "Down")]
    )
    merge = types.SimpleNamespace(
        var=var,
        setup=types.SimpleNamespace(
            era="Run3_2022", phys_model=types.SimpleNamespace(name=model)
        ),
        channels=[CH],
        categories=[CAT],
        scales=["Up", "Down"],
        data_process_name=DATA,
    )
    return EstimateContext(
        merge=merge,
        process="Fakes",
        config=config,
        unc_source=unc_source,
        variations=variations,
        key_dirs=[(CH, n, CAT) for n in {r["name"] for r in records()}],
        mc_backgrounds=[TT, W],
        signals=[],
        reader=store.reader,
    )


def middle(interval):
    return 0.5 * (interval[0] + interval[1])


def run(store, sidebands=SIDEBANDS, **kw):
    with mock.patch.object(
        SL_fakes.HB, "sub_regions", lambda setup: records(sidebands)
    ):
        return SL_fakes.estimate(context(store, **kw))


def contents(h):
    return [h.GetBinContent(b) for b in range(NB + 2)]


def expected(f_low, f_high, k=1.0):
    b_split = split_bin()
    return [
        max(f_low if b < b_split else f_high, 0.0) * k * (10.0 + b % 7)
        for b in range(NB + 2)
    ]


KEY = (CH, name("SR_SL", "HMESub"), CAT)


class Schemes(unittest.TestCase):
    def assertListAlmostEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)

    def test_h_interpolates_at_the_middles(self):
        # iso: lower sidebands f = (0.1, 0.2), upper (0.3, 0.6); anti: anything
        def f_of(base, family):
            if base == "SR_SL_Fakeable":
                return (0.5, 0.5)
            return (0.1, 0.2) if family.startswith("HMESBL") else (0.3, 0.6)

        out = run(Store(f_of))
        u_lo = middle((410.0, 510.0))
        u_hi = middle((680.0, 800.0))
        w = (middle(WINDOW) - u_lo) / (u_hi - u_lo)
        f_low = 0.1 + w * (0.3 - 0.1)
        f_high = 0.2 + w * (0.6 - 0.2)
        self.assertListAlmostEqual(
            contents(out[(KEY, ("Central", "Central"))]), expected(f_low, f_high)
        )

    def test_v_uses_the_anti_isolated_regions(self):
        def f_of(base, family):
            if base == "SR_SL":
                return (0.3, 0.3)
            return (0.5, 0.5) if family == "HMESub" else (0.6, 0.6)

        config = dict(CONFIG, scheme={**CONFIG["scheme"], MASS: "V"})
        out = run(Store(f_of), config=config)
        self.assertListAlmostEqual(
            contents(out[(KEY, ("Central", "Central"))]), expected(0.25, 0.25)
        )

    def test_v_without_excess_in_the_window_or_the_sidebands_gives_no_fakes(self):
        config = dict(CONFIG, scheme={**CONFIG["scheme"], MASS: "V"})
        for iso, window in ((-0.3, -0.5), (0.3, -0.5), (-0.3, 0.5)):
            with self.subTest(iso=iso, window=window):

                def f_of(base, family):
                    if base == "SR_SL":
                        return (iso, iso)
                    return (window, window) if family == "HMESub" else (0.6, 0.6)

                out = run(Store(f_of), config=config)
                h = contents(out[(KEY, ("Central", "Central"))])
                self.assertEqual(h, [0.0] * (NB + 2))

    def test_negative_f_gives_no_fakes(self):
        out = run(Store(lambda base, family: (-0.2, 0.4)))
        h = contents(out[(KEY, ("Central", "Central"))])
        self.assertTrue(all(v == 0 for v in h[: split_bin()]))
        self.assertTrue(any(v > 0 for v in h[split_bin() :]))

    def test_nuisances_at_the_nominal_pass(self):
        def f_of(base, family):
            if base == "SR_SL_Fakeable":
                return (0.4, 0.4) if family == "HMESub" else (0.2, 0.2)
            return (0.1, 0.1)

        out = run(Store(f_of))
        sources = {unc for (key, (unc, scale)) in out}
        self.assertEqual(
            sources, {"Central", *SL_fakes.NUISANCES}, "every declared nuisance"
        )
        nominal = contents(out[(KEY, ("Central", "Central"))])
        # the method uncertainty: |f(V) - f(H)| = |0.1 * 0.4 / 0.2 - 0.1| = 0.1, low part only
        up = contents(out[(KEY, ("FakesMethodLow", "Up"))])
        self.assertListAlmostEqual(up[: split_bin()], expected(0.2, 0.1)[: split_bin()])
        self.assertListAlmostEqual(up[split_bin() :], nominal[split_bin() :])
        stat_up = contents(out[(KEY, ("FakesStatHigh", "Up"))])
        self.assertTrue(all(a >= b for a, b in zip(stat_up, nominal)))
        self.assertListAlmostEqual(stat_up[: split_bin()], nominal[: split_bin()])

    def test_a_variation_pass_recomputes_f_with_the_varied_simulation(self):
        f_of = lambda base, family: (0.3, 0.3)  # noqa: E731
        out = run(Store(f_of), unc_source=UNC)
        self.assertEqual(
            {k[1] for k in out}, {(UNC, "Up"), (UNC, "Down")}, "no nuisance off nominal"
        )
        # data fixed, TT x 1.2 in every region: f = (O + 1.3 T - O)/(1.2 T) - 1, TT x 1.2
        f_up = 1.3 / 1.2 - 1
        self.assertListAlmostEqual(
            contents(out[(KEY, (UNC, "Up"))]), expected(f_up, f_up, k=1.2)
        )

    def test_a_variation_without_f_keeps_the_nominal_f(self):
        # V with an anti-isolated sideband excess of 0.1: TT x 1.2 (Up) turns it negative, so f
        # is not determined at Up; TT x 0.8 (Down) keeps it
        def f_of(base, family):
            if base == "SR_SL":
                return (0.3, 0.3)
            return (0.5, 0.5) if family == "HMESub" else (0.1, 0.1)

        config = dict(CONFIG, scheme={**CONFIG["scheme"], MASS: "V"})
        out = run(Store(f_of), unc_source=UNC, config=config)
        f_nom = 0.3 / 0.1 * 0.5
        self.assertListAlmostEqual(
            contents(out[(KEY, (UNC, "Up"))]), expected(f_nom, f_nom, k=1.2)
        )
        g = lambda f: (1 + f) / 0.8 - 1  # noqa: E731
        f_down = g(0.3) / g(0.1) * g(0.5)
        self.assertListAlmostEqual(
            contents(out[(KEY, (UNC, "Down"))]), expected(f_down, f_down, k=0.8)
        )

    def test_a_variation_without_f_in_one_part_keeps_the_nominal_f_there_only(self):
        # V: the anti-isolated sideband excess is 0.1 below the split and 0.5 from it, so TT x 1.2
        # (Up) leaves f undetermined in the low part only
        def f_of(base, family):
            if base == "SR_SL":
                return (0.3, 0.3)
            return (0.5, 0.5) if family == "HMESub" else (0.1, 0.5)

        config = dict(CONFIG, scheme={**CONFIG["scheme"], MASS: "V"})
        out = run(Store(f_of), unc_source=UNC, config=config)
        g = lambda f: (1 + f) / 1.2 - 1  # noqa: E731
        f_low = 0.3 / 0.1 * 0.5
        f_high = g(0.3) / g(0.5) * g(0.5)
        self.assertListAlmostEqual(
            contents(out[(KEY, (UNC, "Up"))]), expected(f_low, f_high, k=1.2)
        )


class Guards(unittest.TestCase):
    def test_a_split_that_starves_one_part_everywhere_is_an_error(self):
        # TT below the split 1e-4 of its nominal in every region: the low part is short of TT in
        # the only cell, the high part determined
        with self.assertRaisesRegex(RuntimeError, "the split is misplaced"):
            run(Store(lambda b, f: (0.3, 0.3), tt_low=1e-4))

    def test_not_determined_without_enough_tt(self):
        # TT of each part, pooled over two sidebands, below min_tt
        out = run(Store(lambda b, f: (0.3, 0.3), tt_scale=1e-4))
        self.assertEqual(
            {k[1] for k in out},
            {("Central", "Central")}
            | {(n, s) for n in SL_fakes.NUISANCES for s in ("Up", "Down")},
        )
        for hist in out.values():
            self.assertEqual(contents(hist), [0.0] * (NB + 2))
        varied = run(Store(lambda b, f: (0.3, 0.3), tt_scale=1e-4), unc_source=UNC)
        self.assertEqual({k[1] for k in varied}, {(UNC, "Up"), (UNC, "Down")})
        for hist in varied.values():
            self.assertEqual(contents(hist), [0.0] * (NB + 2))

    def test_a_scheme_without_its_regions_is_a_configuration_error(self):
        upper = {k: v for k, v in SIDEBANDS.items() if k.startswith("H")}
        with self.assertRaisesRegex(RuntimeError, "scheme H needs regions"):
            run(Store(lambda b, f: (0.3, 0.3)), sidebands=upper)

    def test_the_other_scheme_missing_gives_no_method_shift(self):
        # V at a mass without lower sidebands: H is not there for FakesMethod
        upper = {k: v for k, v in SIDEBANDS.items() if k.startswith("H")}
        config = dict(CONFIG, scheme={**CONFIG["scheme"], MASS: "V"})
        out = run(Store(lambda b, f: (0.3, 0.3)), sidebands=upper, config=config)
        nominal = contents(out[(KEY, ("Central", "Central"))])
        for part in ("Low", "High"):
            for scale in ("Up", "Down"):
                self.assertEqual(
                    contents(out[(KEY, (f"FakesMethod{part}", scale))]), nominal
                )

    def test_a_binary_dnn_without_sl_windows_gets_nothing(self):
        # e.g. TwoStageDNN_M1600_binary of the default flavour: no split, no sub-regions
        with mock.patch.object(SL_fakes.HB, "sub_regions", lambda setup: []):
            ctx = context(
                Store(lambda b, f: (0.3, 0.3)), var="TwoStageDNN_M1600_binary"
            )
            self.assertEqual(SL_fakes.estimate(ctx), {})

    def test_other_variables_get_nothing(self):
        self.assertEqual(run(Store(lambda b, f: (0.3, 0.3)), var="DeepHME_mass"), {})

    def test_split_must_be_a_bin_edge(self):
        config = dict(CONFIG, split={**CONFIG["split"], MASS: -2.75})
        with self.assertRaisesRegex(RuntimeError, "not a bin edge"):
            run(Store(lambda b, f: (0.3, 0.3)), config=config)

    def test_configuration(self):
        with self.assertRaisesRegex(RuntimeError, "no tt_process for physics model"):
            run(Store(lambda b, f: (0.3, 0.3)), model="Other")
        config = dict(CONFIG, nuisances={})
        with self.assertRaisesRegex(RuntimeError, "must declare"):
            run(Store(lambda b, f: (0.3, 0.3)), config=config)

    def test_every_mass_of_the_windows_has_a_split_on_the_grid_and_a_scheme(self):
        with open(
            os.path.join(ana_path, "config", "sub_regions", "hme_SL_v2610.yaml")
        ) as f:
            masses = set(yaml.safe_load(f)["cuts"])
        self.assertEqual(set(CONFIG["split"]), masses)
        self.assertEqual(set(CONFIG["scheme"]), masses)
        axis = ROOT.TH1D("g", "g", NB, LO, HI).GetXaxis()
        for m in masses:
            b = axis.FindFixBin(CONFIG["split"][m])
            self.assertAlmostEqual(axis.GetBinLowEdge(b), CONFIG["split"][m])
            self.assertEqual(CONFIG["scheme"][m], "V" if m <= 400 else "H")


class RealConfiguration(unittest.TestCase):
    """The estimator's cells and configuration on the real booking of the SL and CI flavours."""

    def setup_for(self, flavour):
        with open(os.path.join(ana_path, "config", "global.yaml")) as f:
            global_params = yaml.safe_load(f)
        with open(os.path.join(ana_path, "config", "plot", "histograms.yaml")) as f:
            hists = yaml.safe_load(f)
        return types.SimpleNamespace(
            global_params=global_params,
            histTuple_flavor=flavour,
            ana_path=ana_path,
            hists=types.SimpleNamespace(config_dict=hists),
        )

    def test_a_cell_with_both_windows_in_every_channel_and_category(self):
        for flavour, masses, n_cells in (
            ("Signal_Extraction_SL_window", sorted(CONFIG["split"]), 150),
            ("CI", [300], 6),
        ):
            setup = self.setup_for(flavour)
            total = 0
            for mass in masses:
                ctx = types.SimpleNamespace(config=CONFIG, setup=setup)
                cells = SL_fakes._cells(ctx, mass)
                total += len(cells)
                for cell in cells.values():
                    self.assertIsNotNone(cell.window)
                    self.assertIsNotNone(cell.anti_window)
                    # the provisional scheme has its regions in every cell
                    self.assertTrue(
                        SL_fakes.Estimate.available(
                            CONFIG["scheme"].get(mass, "V"), cell
                        )
                    )
            self.assertEqual(total, n_cells, flavour)

    def test_a_tt_process_for_every_model_listing_fakes(self):
        with open(os.path.join(ana_path, "config", "phys_models.yaml")) as f:
            models = yaml.safe_load(f)
        listing = [
            name
            for name, model in models.items()
            if isinstance(model, dict) and "Fakes" in (model.get("backgrounds") or [])
        ]
        self.assertTrue(listing)
        for name in listing:
            self.assertIn(name, CONFIG["tt_process"], name)
            self.assertIn(CONFIG["tt_process"][name], models[name]["backgrounds"], name)


if __name__ == "__main__":
    unittest.main()
