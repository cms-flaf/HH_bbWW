#!/usr/bin/env python3
"""Analysis/SL_fakes.py, the single-lepton fake estimator, on synthetic merged histograms with a known
excess f in each region, through a stand-in of FLAF's EstimateContext (FLAF Analysis/
HistMergerFromHists.py) and the real Fakes block of config/processes.yaml.

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

import ROOT  # noqa: E402

from Analysis import SL_fakes  # noqa: E402

ROOT.gROOT.SetBatch(True)

with open(os.path.join(ana_path, "config", "processes.yaml")) as f:
    CONFIG = yaml.safe_load(f)["Fakes"]["data_driven"]

MASS = 600
VAR = f"TwoStageDNN_M{MASS}_binary"
SPLIT = CONFIG["split"][MASS]
NB, LO, HI = 94, -23.5, 23.5
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

    def __init__(self, f_of, tt_scale=1.0, tt_shift=None):
        self.hists = {}
        b_split = split_bin()
        for base in ("SR_SL", "SR_SL_Fakeable"):
            for family in ["HMESub"] + ["HMESB" + k for k in SIDEBANDS]:
                region = name(base, family)
                for unc, scale in (("Central", "Central"), (UNC, "Up"), (UNC, "Down")):
                    k = 1.2 if scale == "Up" else 0.8 if scale == "Down" else 1.0
                    t = hist(lambda b: tt_scale * k * (10.0 + b % 7))
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

    def get(self, process, region, unc, scale):
        if process == DATA:
            if region == name("SR_SL", "HMESub"):
                raise AssertionError("the isolated window's data were read")
            unc, scale = "Central", "Central"
        h = self.hists.get((process, region, unc, scale))
        return None if h is None else h.Clone()


def context(store, unc_source="Central", var=VAR, model="Run3_Model", config=CONFIG):
    variations = (
        [("Central", "Central")]
        if unc_source == "Central"
        else [(unc_source, "Up"), (unc_source, "Down")]
    )

    def get(process, key_dir, unc="Central", scale="Central"):
        ch, region, cat = key_dir
        assert (ch, cat) == (CH, CAT)
        return store.get(process, region, unc, scale)

    def require(process, key_dir, unc="Central", scale="Central"):
        h = get(process, key_dir, unc, scale)
        if h is None:
            raise KeyError((process, key_dir, unc, scale))
        return h

    key_dirs = [(CH, n, CAT) for n in {r["name"] for r in records()}]
    return types.SimpleNamespace(
        config=config,
        var=var,
        era="Run3_2022",
        setup=types.SimpleNamespace(phys_model=types.SimpleNamespace(name=model)),
        unc_source=unc_source,
        variations=variations,
        nominal_pass=unc_source == "Central",
        key_dirs=key_dirs,
        data_process=DATA,
        mc_backgrounds=[TT, W],
        get=get,
        require=require,
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


class Guards(unittest.TestCase):
    def test_not_determined_without_enough_tt(self):
        # TT of each part, pooled over two sidebands, below min_tt
        out = run(Store(lambda b, f: (0.3, 0.3), tt_scale=1e-4))
        self.assertEqual(out, {})

    def test_h_without_a_lower_side_is_not_determined(self):
        upper = {k: v for k, v in SIDEBANDS.items() if k.startswith("H")}
        out = run(Store(lambda b, f: (0.3, 0.3)), sidebands=upper)
        self.assertEqual(out, {})

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


if __name__ == "__main__":
    unittest.main()
