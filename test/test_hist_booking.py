#!/usr/bin/env python3
"""Analysis/hist_booking.py, the FLAF booking plugin of HH->bbWW: the regions it gives FLAF, the booking
blocks of a HistFromNtuple job, sub-regions indexed by model points (mass; mass and spin; couplings), the
variables of a point (signal_params), the cut forms and their guards. Plain Python, with a stand-in
setup."""

import os
import sys
import tempfile
import types
import unittest

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo not in sys.path:
    sys.path.insert(0, repo)

from Analysis import hist_booking as HB  # noqa: E402

# a 10 GeV grid from 200 to 400 and one bin below and above, as the HME grid starts
HME_BINS = [0.0] + [200.0 + 10.0 * i for i in range(21)] + [500.0]
SIGNAL_PARAMS = {
    "DNN_M300_Signal": {"mass": 300},
    "DNN_M400_Signal": {"mass": 400},
    "DNN_M300_S0": {"mass": 300, "spin": 0},
    "DNN_M300_S2": {"mass": 300, "spin": 2},
    "DNN_M300_S02": [{"mass": 300, "spin": 0}, {"mass": 300, "spin": 2}],
    "DNN_kl2p45": {"kl": 2.45},
}

HEADER = 'parameters: [ mass ]\nregion_name: "{region}_{name}_m{mass}"\n'
CUTS = HEADER + """
fits: { note: other keys of the file are left alone }
cuts:
  300:
    res2b: { DeepHME_mass: [ 250, 350 ] }
    boosted: { DeepHME_mass: [ 0, 400 ] }
  400:
    res2b: { DeepHME_mass: [ 300, 500 ] }
"""

CHANNELS = {"eE": 1, "muMu": 2, "eMu": 3}
KEYS = [
    ((ch, reg, cat), f"f_{ch}_{reg}_{cat}")
    for ch in ("eE", "muMu")
    for reg in ("SR", "TT_CR")
    for cat in ("res2b", "boosted")
]
VARS = ["DeepHME_mass", "lep1_pt", "DNN_M300_Signal", "DNN_M400_Signal"]


class Hists:
    def __init__(self):
        self.config_dict = {
            "DeepHME_mass": {"x_bins": HME_BINS},
            "lep1_pt": {"x_bins": "40|0:200"},  # 5 GeV bins
        }
        for var, params in SIGNAL_PARAMS.items():
            self.config_dict[var] = {"x_bins": "10|0:1", "signal_params": params}


class BookingTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "config"))
        self.write_cuts(CUTS)

    def write_cuts(self, text):
        with open(os.path.join(self.tmp.name, "config", "cuts.yaml"), "w") as f:
            f.write(text)

    def setup_for(
        self,
        regions=("SR", "TT_CR"),
        entries=None,
        fullres=("DeepHME_mass",),
        variables=VARS,
        **gp,
    ):
        flavour = {
            "variables": list(variables),
            "fullResolution_variables": list(fullres),
            "regions": list(regions),
        }
        if entries is not None:
            flavour["sub_regions"] = entries
        global_params = {
            "histTuple_flavors": {"F": flavour},
            "categories": ["res2b", "recovery", "boosted"],
            "channelDefinition": CHANNELS,
        }
        global_params.update(gp)
        return types.SimpleNamespace(
            global_params=global_params,
            histTuple_flavor="F",
            ana_path=self.tmp.name,
            hists=Hists(),
        )

    def entry(self, **changes):
        e = {
            "name": "HMESub",
            "cuts": "config/cuts.yaml",
            "regions": ["SR"],
            "points": "all",
        }
        e.update(changes)
        return e

    def one(self, cut_text, **entry):
        """The sub-region of mass 300 and base region SR from a file whose mass 300 is `cut_text`."""
        self.write_cuts(HEADER + "cuts:\n  300:\n" + cut_text)
        setup = self.setup_for(entries=[self.entry(points=[{"mass": 300}], **entry)])
        (s,) = HB.sub_regions(setup)
        return s


class Regions(BookingTestBase):
    def test_booked_regions_then_the_sub_region_bases(self):
        setup = self.setup_for(
            regions=["SR"], entries=[self.entry(regions=["SR", "DY_CR"])]
        )
        self.assertEqual(HB.regions(setup), ["SR", "DY_CR"])

    def test_sub_regions_only(self):
        setup = self.setup_for(
            regions=[], entries=[self.entry(regions=["SR", "TT_CR"])]
        )
        self.assertEqual(HB.regions(setup), ["SR", "TT_CR"])

    def test_nothing_at_all(self):
        with self.assertRaisesRegex(
            HB.HistBookingError, "no regions and no sub-regions"
        ):
            HB.regions(self.setup_for(regions=[]))

    def test_regions_must_be_a_list(self):
        setup = self.setup_for()
        setup.global_params["histTuple_flavors"]["F"]["regions"] = "SR"
        with self.assertRaisesRegex(
            HB.HistBookingError, "must be a list of region names"
        ):
            HB.regions(setup)


class Booking(BookingTestBase):
    def test_every_variable_in_the_booked_regions_and_the_sub_regions_by_point(self):
        setup = self.setup_for(regions=["SR", "TT_CR"], entries=[self.entry()])
        blocks = HB.booking(setup, KEYS, VARS)
        self.assertEqual(blocks[0], {"keys": KEYS, "variables": VARS})
        subs = {b["keys"][0][0][1]: b for b in blocks[1:]}
        self.assertEqual(sorted(subs), ["SR_HMESub_m300", "SR_HMESub_m400"])
        # mass 300: its own DNN output and the variables without signal_params
        self.assertEqual(
            subs["SR_HMESub_m300"]["variables"],
            ["DeepHME_mass", "lep1_pt", "DNN_M300_Signal"],
        )
        self.assertEqual(
            subs["SR_HMESub_m400"]["variables"],
            ["DeepHME_mass", "lep1_pt", "DNN_M400_Signal"],
        )
        self.assertEqual(
            subs["SR_HMESub_m300"]["keys"][:2],
            [
                (
                    ("eE", "SR_HMESub_m300", "res2b"),
                    "(f_eE_SR_res2b) && (DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16)",
                ),
                (
                    ("eE", "SR_HMESub_m300", "boosted"),
                    "(f_eE_SR_boosted) && (DeepHME_mass_bin < 21)",
                ),
            ],
        )
        # mass 400 has no boosted cut: res2b keys only
        self.assertEqual(
            [k[2] for k, _ in subs["SR_HMESub_m400"]["keys"]], ["res2b", "res2b"]
        )
        self.assertEqual(subs["SR_HMESub_m300"]["columns"], ["DeepHME_mass_bin"])

    def test_sub_regions_only_book_no_base_key(self):
        setup = self.setup_for(
            regions=[], entries=[self.entry(regions=["SR", "TT_CR"])]
        )
        names = {k[1] for b in HB.booking(setup, KEYS, VARS) for k, _ in b["keys"]}
        self.assertEqual(
            names,
            {
                "SR_HMESub_m300",
                "SR_HMESub_m400",
                "TT_CR_HMESub_m300",
                "TT_CR_HMESub_m400",
            },
        )

    def test_without_sub_regions_every_variable_in_every_booked_key(self):
        setup = self.setup_for(regions=["SR", "TT_CR"])
        self.assertEqual(
            HB.booking(setup, KEYS, VARS), [{"keys": KEYS, "variables": VARS}]
        )

    def test_the_cap(self):
        setup = self.setup_for(entries=[self.entry()], sub_regions_max_key_vars=5)
        with self.assertRaisesRegex(
            HB.HistBookingError, "above sub_regions_max_key_vars = 5"
        ):
            HB.booking(setup, KEYS, VARS)


class ModelPoints(BookingTestBase):
    SPIN_CUTS = """
parameters: [ mass, spin ]
region_name: "{region}_{name}_m{mass}_s{spin}"
cuts:
  300:
    0: { res2b: "DeepHME_mass >= 250 && DeepHME_mass < {mass} + 100" }
    2: { res2b: { DeepHME_mass: [ 240, 400 ] } }
"""
    SPIN_VARS = [
        "DeepHME_mass",
        "DNN_M300_Signal",
        "DNN_M300_S0",
        "DNN_M300_S2",
        "DNN_M300_S02",
    ]

    def spin_blocks(self):
        self.write_cuts(self.SPIN_CUTS)
        setup = self.setup_for(entries=[self.entry()], variables=self.SPIN_VARS)
        return {
            b["keys"][0][0][1]: b for b in HB.booking(setup, KEYS, self.SPIN_VARS)[1:]
        }

    def test_mass_and_spin(self):
        subs = self.spin_blocks()
        self.assertEqual(sorted(subs), ["SR_HMESub_m300_s0", "SR_HMESub_m300_s2"])
        # a mass-only variable in both spins, a spin variable in its own, a two-entry variable in both
        self.assertEqual(
            subs["SR_HMESub_m300_s0"]["variables"],
            ["DeepHME_mass", "DNN_M300_Signal", "DNN_M300_S0", "DNN_M300_S02"],
        )
        self.assertEqual(
            subs["SR_HMESub_m300_s2"]["variables"],
            ["DeepHME_mass", "DNN_M300_Signal", "DNN_M300_S2", "DNN_M300_S02"],
        )

    def test_parameters_in_a_cpp_cut(self):
        subs = self.spin_blocks()
        self.assertIn(
            "DeepHME_mass >= 250 && DeepHME_mass < 300 + 100",
            subs["SR_HMESub_m300_s0"]["keys"][0][1],
        )

    def test_couplings_and_their_names(self):
        self.write_cuts(
            'parameters: [ kl ]\nregion_name: "{region}_{name}_kl{kl}"\n'
            "cuts:\n  2.45: { res2b: { DeepHME_mass: [ 250, 350 ] } }\n"
            "  -1.5: { res2b: { DeepHME_mass: [ 250, 350 ] } }\n  1.0: { res2b: { DeepHME_mass: [ 250, 350 ] } }\n"
        )
        variables = ["DeepHME_mass", "DNN_kl2p45"]
        setup = self.setup_for(entries=[self.entry(name="HHSub")], variables=variables)
        subs = {s["name"]: s for s in HB.sub_regions(setup)}
        self.assertEqual(
            sorted(subs), ["SR_HHSub_kl1", "SR_HHSub_kl2p45", "SR_HHSub_klm1p5"]
        )
        blocks = {b["keys"][0][0][1]: b for b in HB.booking(setup, KEYS, variables)[1:]}
        self.assertEqual(blocks["SR_HHSub_kl2p45"]["variables"], variables)
        self.assertEqual(blocks["SR_HHSub_kl1"]["variables"], ["DeepHME_mass"])

    def test_listed_points(self):
        self.write_cuts(self.SPIN_CUTS)
        setup = self.setup_for(entries=[self.entry(points=[{"mass": 300, "spin": 2}])])
        self.assertEqual(
            [s["name"] for s in HB.sub_regions(setup)], ["SR_HMESub_m300_s2"]
        )

    def test_point_guards(self):
        self.write_cuts(self.SPIN_CUTS)
        for points, msg in (
            ([{"mass": 300}], "a point is a mapping of \\['mass', 'spin'\\]"),
            ([{"mass": 300, "spin": 1}], "has no cut"),
            ("some", "points must be 'all' or a non-empty list"),
        ):
            with self.subTest(points=points):
                with self.assertRaisesRegex(HB.HistBookingError, msg):
                    HB.sub_regions(self.setup_for(entries=[self.entry(points=points)]))

    def test_file_guards(self):
        for text, msg in (
            (
                "cuts: { 300: { res2b: { DeepHME_mass: [ 250, 350 ] } } }\n",
                "needs `parameters`",
            ),
            (
                "parameters: [ mass ]\ncuts: { 300: { res2b: { DeepHME_mass: [ 250, 350 ] } } }\n",
                "needs `region_name`",
            ),
            (
                'parameters: [ mass, spin ]\nregion_name: "{region}_{name}_m{mass}"\n'
                "cuts: { 300: { 0: { res2b: { DeepHME_mass: [ 250, 350 ] } } } }\n",
                "lacks \\['spin'\\]",
            ),
            (
                'parameters: [ mass, spin ]\nregion_name: "{region}_{name}_m{mass}_s{spin}"\n'
                "cuts: { 300: { res2b: { DeepHME_mass: [ 250, 350 ] } } }\n",
                "category 'DeepHME_mass'|expected the values|has no category cuts|not a configured",
            ),
        ):
            with self.subTest(text=text):
                self.write_cuts(text)
                with self.assertRaisesRegex(HB.HistBookingError, msg):
                    HB.sub_regions(self.setup_for(entries=[self.entry()]))


class BinFormCuts(BookingTestBase):
    def cut(self, s, cat="res2b", ch="eE"):
        return s["cuts"][(cat, ch)]

    def test_flows_at_the_grid_ends(self):
        s = self.one(
            "    res2b: { DeepHME_mass: [ 250, 350 ] }\n    boosted: { DeepHME_mass: [ 0, 400 ] }\n"
        )
        self.assertEqual(
            self.cut(s),
            ("DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16", ["DeepHME_mass_bin"]),
        )
        self.assertEqual(self.cut(s, "boosted")[0], "DeepHME_mass_bin < 21")

    def test_intervals_of_one_variable_are_ored_and_variables_anded(self):
        s = self.one(
            "    res2b: { DeepHME_mass: [ [ 250, 300 ], [ 350, 400 ] ], lep1_pt: [ 30, 200 ] }\n"
        )
        self.assertEqual(
            self.cut(s),
            (
                "((DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 11) || "
                "(DeepHME_mass_bin >= 16 && DeepHME_mass_bin < 21)) && lep1_pt_bin >= 6",
                ["DeepHME_mass_bin", "lep1_pt_bin"],
            ),
        )

    def test_the_whole_range_cuts_nothing(self):
        self.assertEqual(
            self.cut(self.one("    res2b: { lep1_pt: [ 0, 200 ] }\n")), ("true", [])
        )

    def test_edge_off_the_binning(self):
        with self.assertRaisesRegex(HB.HistBookingError, "31.0 is not a bin edge"):
            self.one("    res2b: { lep1_pt: [ 31, 200 ] }\n")

    def test_empty_and_malformed_intervals(self):
        with self.assertRaisesRegex(HB.HistBookingError, "empty interval"):
            self.one("    res2b: { lep1_pt: [ 50, 30 ] }\n")
        with self.assertRaisesRegex(HB.HistBookingError, "expected \\[low, high\\]"):
            self.one("    res2b: { lep1_pt: [ 30, 50, 70 ] }\n")


class CppAndCombinedCuts(BookingTestBase):
    def test_used_as_written_with_the_parameters(self):
        s = self.one(
            '    res2b: "DeepHME_mass > 0.8 * {mass} && DeepHME_mass < 1.4 * {mass}"\n'
        )
        self.assertEqual(
            s["cuts"][("res2b", "eE")],
            ("DeepHME_mass > 0.8 * 300 && DeepHME_mass < 1.4 * 300", ["DeepHME_mass"]),
        )

    def test_functions_literals_and_namespaces_are_not_variables(self):
        s = self.one('    res2b: "std::abs(DeepHME_mass - 300.f) < 1e2 && true"\n')
        self.assertEqual(s["cuts"][("res2b", "eE")][1], ["DeepHME_mass"])

    def test_a_variable_not_at_full_resolution_is_refused(self):
        with self.assertRaisesRegex(
            HB.HistBookingError,
            "uses lep1_pt, not in fullResolution_variables of flavour F; add it there "
            "\\(it is not added automatically\\)",
        ):
            self.one('    res2b: "lep1_pt > 30"\n')

    def test_a_cpp_part_and_a_bin_form_part(self):
        s = self.one(
            '    res2b: [ "DeepHME_mass >= 250 && DeepHME_mass < 400", { lep1_pt: [ 30, 200 ] } ]\n'
        )
        self.assertEqual(
            s["cuts"][("res2b", "eE")],
            (
                "(DeepHME_mass >= 250 && DeepHME_mass < 400) && (lep1_pt_bin >= 6)",
                ["DeepHME_mass", "lep1_pt_bin"],
            ),
        )

    def test_a_part_that_is_neither(self):
        with self.assertRaisesRegex(HB.HistBookingError, "a cut part is a"):
            self.one("    res2b: [ 5 ]\n")
        with self.assertRaisesRegex(HB.HistBookingError, "empty cut"):
            self.one("    res2b: [ ]\n")


class ChannelMaps(BookingTestBase):
    def test_a_cut_per_channel(self):
        s = self.one(
            "    res2b:\n"
            "      eE: { lep1_pt: [ 30, 200 ] }\n"
            '      muMu: [ "DeepHME_mass > 250", { lep1_pt: [ 25, 200 ] } ]\n'
        )
        self.assertEqual(s["cuts"][("res2b", "eE")][0], "lep1_pt_bin >= 6")
        self.assertEqual(
            s["cuts"][("res2b", "muMu")][0],
            "(DeepHME_mass > 250) && (lep1_pt_bin >= 5)",
        )
        self.assertNotIn(("res2b", "eMu"), s["cuts"])

    def test_a_cut_without_map_is_the_same_for_every_channel(self):
        s = self.one("    res2b: { lep1_pt: [ 30, 200 ] }\n")
        self.assertEqual({ch for _, ch in s["cuts"]}, set(CHANNELS))

    def test_a_channel_missing_from_the_map_books_no_key(self):
        self.write_cuts(
            HEADER + "cuts:\n  300:\n    res2b:\n      eE: { lep1_pt: [ 30, 200 ] }\n"
        )
        setup = self.setup_for(entries=[self.entry()])
        (sub,) = HB.booking(setup, KEYS, VARS)[1:]
        self.assertEqual(
            [k for k, _ in sub["keys"]], [("eE", "SR_HMESub_m300", "res2b")]
        )

    def test_channels_mixed_with_variables(self):
        with self.assertRaisesRegex(HB.HistBookingError, "neither all channels"):
            self.one(
                "    res2b: { eE: { lep1_pt: [ 30, 200 ] }, lep1_pt: [ 30, 200 ] }\n"
            )


class Guards(BookingTestBase):
    def refused(self, msg, **changes):
        with self.assertRaisesRegex(HB.HistBookingError, msg):
            HB.sub_regions(self.setup_for(entries=[self.entry(**changes)]))

    def test_old_keys_are_refused(self):
        self.refused("unknown keys \\['vars'\\]", vars=["DNN_M{MX}_Signal"])
        with self.assertRaisesRegex(HB.HistBookingError, "unknown keys \\['masses'\\]"):
            e = self.entry(masses="all")
            del e["points"]
            HB.sub_regions(self.setup_for(entries=[e]))

    def test_name_must_be_a_word(self):
        self.refused("name must be a word", name="HME Sub")

    def test_cuts_file_missing_or_empty(self):
        self.refused("not found", cuts="config/none.yaml")
        self.write_cuts(HEADER + "windows: {}\n")
        self.refused("has no 'cuts' mapping")

    def test_category_unknown(self):
        self.write_cuts(
            HEADER + "cuts:\n  300:\n    res3b: { DeepHME_mass: [ 250, 350 ] }\n"
        )
        self.refused("category 'res3b'")

    def test_variable_not_binned(self):
        self.write_cuts(HEADER + "cuts:\n  300:\n    res2b: { mbb: [ 70, 150 ] }\n")
        self.refused("neither all channels .* nor all binned variables")

    def test_region_defined_twice(self):
        with self.assertRaisesRegex(HB.HistBookingError, "defined twice"):
            HB.sub_regions(self.setup_for(entries=[self.entry(), self.entry()]))


class Helpers(unittest.TestCase):
    def test_cpp_identifiers(self):
        self.assertEqual(
            HB.cpp_variables("x > 1.5e-3 && f(y) < 2.f && ROOT::VecOps::Max(z) || !w"),
            ["x", "y", "z", "w"],
        )

    def test_name_values(self):
        self.assertEqual(
            [HB.name_value(v) for v in (300, 300.0, 2.45, -1.5, "a")],
            ["300", "300", "2p45", "m1p5", "a"],
        )


if __name__ == "__main__":
    unittest.main()
