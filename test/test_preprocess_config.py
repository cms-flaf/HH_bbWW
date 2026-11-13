#!/usr/bin/env python3
"""Analysis/preprocess_config.py: the DL card's `preprocess:` blocks and the binning file.

- the DL cards and the binning file as they are parse to the expected model;
- every strict check raises: keys that moved out of the binning file, unknown keys, a
  `preprocess:` on a non-shape uncertainty, member values whose processes do not hold the
  members, estimates without their nuisance (or the reverse), and the others below.

Only PyYAML is needed (no ROOT).

    python3 test/test_preprocess_config.py
"""

import copy
import os
import sys
import unittest

import yaml

ANA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ANA)
sys.path.insert(0, os.path.join(ANA, "Analysis"))
try:
    from StatInference.common.card_config import load_card_config
except ImportError:
    # The unit-tests workflow checks out the FLAF submodule only. Then the cards are merged
    # here by the same rule (StatInference/common/card_config.py): the included files first,
    # the named lists joined, every other top-level key set in one file only.
    _LISTS = ("processes", "uncertainties", "limit_plots", "impact_plots")

    def load_card_config(path):
        with open(path) as f:
            top = yaml.safe_load(f) or {}
        merged = {}
        base = os.path.dirname(path)
        for part in [os.path.join(base, inc) for inc in top.get("include") or []] + [
            path
        ]:
            with open(part) as f:
                cfg = top if part == path else (yaml.safe_load(f) or {})
            for key, value in cfg.items():
                if key == "include":
                    continue
                if key in _LISTS:
                    merged.setdefault(key, []).extend(value or [])
                elif key in merged:
                    raise RuntimeError(f"{key} is set twice (again in {part})")
                else:
                    merged[key] = value
        return merged


from preprocess_config import (  # noqa: E402
    PreprocessConfigError,
    PreprocessModel,
    check_binning_config,
    read_saved_edges,
)

DL = os.path.join(ANA, "config", "Datacards", "DL")
BINNING = os.path.join(DL, "binning_eqsig_DL_v2610.yaml")


def card():
    return load_card_config(os.path.join(DL, "dl_spin0.yaml"))


def entry(cfg, name):
    return next(u for u in cfg["uncertainties"] if u.get("name") == name)


def process(cfg, name):
    return next(
        p for p in cfg["processes"] if isinstance(p, dict) and p["process"] == name
    )


class DLCards(unittest.TestCase):
    def test_both_spins_parse(self):
        for spin in (0, 2):
            PreprocessModel(load_card_config(os.path.join(DL, f"dl_spin{spin}.yaml")))

    def test_model(self):
        m = PreprocessModel(card())
        self.assertEqual(
            m.members,
            {
                "TT": ["TT", "ttV"],
                "DY": ["DY_M_10to50", "DY_M_50"],
                "ST": ["ST"],
                "Minor": ["VV", "VVV", "H", "W"],
            },
        )
        self.assertEqual(m.dropped_in("SR_res2b"), {"W"})
        self.assertEqual(m.dropped_in("TT_boosted"), {"W", "DY_M_10to50"})
        self.assertEqual(m.dropped_in("SR_recovery"), set())
        self.assertEqual(m.member_uncs["QCD_scale_TT"], [("TT", (-0.036, 0.025))])
        self.assertEqual(m.member_uncs["QCD_scale_ttV"], [("ttV", (-0.12, 0.12))])
        self.assertEqual(len(m.member_uncs["PDF_alphas"]), 6)
        self.assertEqual(
            m.tilts("eE", "DY_res2b"), [("CMS_bbww_dycr_shape_res2b", "DY", 0.04)]
        )
        self.assertEqual(
            m.tilts("muMu", "TT_boosted"), [("CMS_bbww_ttcr_shape_boosted", "TT", 0.02)]
        )
        self.assertEqual(m.tilts("eE", "SR_res2b"), [])
        self.assertEqual(len(m.templates), 7)
        self.assertEqual(
            m.template("eE", "SR_res2b"), ("tail", "CMS_bbww_dy_fit_eE_res2b")
        )
        self.assertEqual(
            m.template("eMu", "SR_boosted"), ("fit_only", "CMS_bbww_dy_fit_eMu_boosted")
        )
        self.assertEqual(m.template("eMu", "SR_res2b"), (None, None))
        self.assertTrue(m.lowmass_applies("eMu", "SR_boosted"))
        self.assertFalse(m.lowmass_applies("eMu", "SR_recovery"))
        self.assertEqual(m.lowmass["nuisance"], "CMS_bbww_dy_lowmass")
        self.assertEqual(m.lowmass["nuisance_factors"], [0.0, 2.0])
        self.assertEqual(m.lowmass["output"], "DY")
        self.assertEqual(m.dy_template["process"], "DY")

    def test_binning_file_is_binning_only(self):
        with open(BINNING) as f:
            check_binning_config(yaml.safe_load(f), BINNING)


class Strict(unittest.TestCase):
    def assertRaisesWith(self, text, cfg):
        with self.assertRaises(PreprocessConfigError) as ctx:
            PreprocessModel(cfg)
        self.assertIn(text, str(ctx.exception))

    def test_moved_binning_keys(self):
        with open(BINNING) as f:
            base = yaml.safe_load(f)
        for key in (
            "backgrounds",
            "drop",
            "member_uncertainties",
            "control_region_shapes",
            "dy_template",
            "lowmass_dy",
        ):
            cfg = dict(base, **{key: {}})
            with self.assertRaises(PreprocessConfigError) as ctx:
                check_binning_config(cfg)
            self.assertIn(f"`{key}` is now in", str(ctx.exception))
        with self.assertRaises(PreprocessConfigError):
            check_binning_config(dict(base, n_bin={}))
        with self.assertRaises(PreprocessConfigError):
            check_binning_config(dict(base, gates=dict(base["gates"], min_each=1)))

    def test_unknown_uncertainty_key(self):
        cfg = card()
        entry(cfg, "PDF_alphas")["preprocess"] = {"member_value": {"TT": 0.1}}
        self.assertRaisesWith("unknown key member_value", cfg)

    def test_two_keys(self):
        cfg = card()
        entry(cfg, "PDF_alphas")["preprocess"]["cr_tilt"] = 0.1
        self.assertRaisesWith("exactly one of", cfg)

    def test_not_a_shape(self):
        cfg = card()
        entry(cfg, "CMS_bbww_norm_TT_res2b")["preprocess"] = {
            "member_values": {"TT": 0.1}
        }
        self.assertRaisesWith("its type must be shape", cfg)

    def test_members_and_preprocess(self):
        cfg = card()
        entry(cfg, "QCDscale_shape_TT")["preprocess"] = {"member_values": {"TT": 0.1}}
        self.assertRaisesWith("cannot also be built", cfg)

    def test_member_processes_must_hold_the_members(self):
        cfg = card()
        entry(cfg, "PDF_alphas")["processes"] = ["TT", "ST", "DY"]  # H is in Minor
        self.assertRaisesWith("must be the backgrounds holding its members", cfg)
        cfg = card()
        del entry(cfg, "QCD_scale_TT")["processes"]  # all backgrounds
        self.assertRaisesWith("must be the backgrounds holding its members", cfg)

    def test_unknown_member(self):
        cfg = card()
        entry(cfg, "PDF_alphas")["preprocess"]["member_values"]["tt"] = 0.1
        self.assertRaisesWith("tt is a member of no background", cfg)

    def test_member_value_format(self):
        for bad in ("0.1", [0.1], [0.1, 0.2, 0.3], float("nan"), True):
            cfg = card()
            entry(cfg, "QCD_scale_ttV")["preprocess"]["member_values"]["ttV"] = bad
            self.assertRaises(PreprocessConfigError, PreprocessModel, cfg)

    def test_unknown_process_key(self):
        cfg = card()
        process(cfg, "TT")["preprocess"]["member"] = ["TT"]
        self.assertRaisesWith("unknown keys ['member']", cfg)

    def test_preprocess_on_a_signal(self):
        cfg = card()
        sig = next(p for p in cfg["processes"] if p.get("is_signal"))
        sig["preprocess"] = {"members": ["x"]}
        self.assertRaisesWith("for backgrounds only", cfg)

    def test_member_in_two_backgrounds(self):
        cfg = card()
        process(cfg, "TT")["preprocess"]["members"] = ["TT", "ttV", "W"]
        self.assertRaisesWith("member of both", cfg)

    def test_drop(self):
        cfg = card()
        process(cfg, "Minor")["preprocess"]["drop"]["W"] = ["res2b"]
        self.assertRaisesWith("card categories", cfg)
        cfg = card()
        process(cfg, "Minor")["preprocess"]["drop"]["ttV"] = ["SR_res2b"]
        self.assertRaisesWith("not one of its members", cfg)

    def test_estimate_settings(self):
        cfg = card()
        process(cfg, "DY")["preprocess"]["dy_template"]["guard"] = 3
        self.assertRaisesWith("unknown keys ['guard']", cfg)
        cfg = card()
        process(cfg, "DY")["preprocess"]["dy_template"]["fit"] = {"degree": 3}
        self.assertRaisesWith("fit: unknown keys", cfg)
        cfg = card()
        process(cfg, "DY")["preprocess"]["dy_template"]["share_by"] = "TT"
        self.assertRaisesWith("share_by TT is not one of its members", cfg)
        cfg = card()
        del process(cfg, "DY")["preprocess"]["lowmass_dy"]["reference"]
        self.assertRaisesWith("missing ['reference']", cfg)
        cfg = card()
        process(cfg, "DY")["preprocess"]["dy_template"]["pool"] = [["eE"], ["eE"]]
        self.assertRaisesWith("in two groups", cfg)

    def test_template_without_settings(self):
        cfg = card()
        del process(cfg, "DY")["preprocess"]["dy_template"]
        self.assertRaisesWith("no process has preprocess: dy_template", cfg)

    def test_settings_without_template(self):
        cfg = card()
        cfg["uncertainties"] = [
            u
            for u in cfg["uncertainties"]
            if "dy_template" not in (u.get("preprocess") or {})
        ]
        self.assertRaisesWith("no uncertainty has preprocess: dy_template", cfg)

    def test_template_mode_and_scope(self):
        cfg = card()
        entry(cfg, "CMS_bbww_dy_fit_eE_res2b")["preprocess"]["dy_template"] = "fit"
        self.assertRaisesWith("must be one of", cfg)
        cfg = card()
        del entry(cfg, "CMS_bbww_dy_fit_eE_res2b")["channels"]
        self.assertRaisesWith("explicit channels", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dy_fit_eE_res2b")["channels"] = ["eE", "muMu"]
        self.assertRaisesWith("already has the DY template", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dy_fit_eE_res2b")["processes"] = ["TT"]
        self.assertRaisesWith("processes must be [DY]", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dy_fit_eE_res2b")["eras"] = ["Run3_2022"]
        self.assertRaisesWith("cannot have eras", cfg)

    def test_pooled_channels_all_fit_only(self):
        cfg = card()
        entry(cfg, "CMS_bbww_dy_fit_muMu_boosted")["preprocess"]["dy_template"] = "tail"
        self.assertRaisesWith("fitted pooled with muMu", cfg)

    def test_lowmass(self):
        cfg = card()
        cfg["uncertainties"] = [
            u for u in cfg["uncertainties"] if u["name"] != "CMS_bbww_dy_lowmass"
        ]
        self.assertRaisesWith("exactly one uncertainty with", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dy_lowmass")["preprocess"]["lowmass_dy"] = 2.0
        self.assertRaisesWith("factors [Down, Up]", cfg)
        cfg = card()
        del process(cfg, "DY")["preprocess"]["lowmass_dy"]
        self.assertRaisesWith("no process has preprocess: lowmass_dy", cfg)

    def test_cr_tilt(self):
        cfg = card()
        entry(cfg, "CMS_bbww_dycr_shape_res2b")["processes"] = ["DY", "TT"]
        self.assertRaisesWith("exactly one background", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dycr_shape_res2b")["preprocess"] = {"cr_tilt": "4%"}
        self.assertRaisesWith("must be a number", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dycr_shape_res2b")["categories"] = ["DY_res2"]
        self.assertRaisesWith("match no card category", cfg)

    def test_lowmass_scope(self):
        cfg = card()
        entry(cfg, "CMS_bbww_dy_lowmass")["channels"] = ["eE"]
        self.assertRaisesWith("split the DY template pool", cfg)
        cfg = card()
        entry(cfg, "CMS_bbww_dy_lowmass")["categories"] = ["SR_boosted", "SR_recovery"]
        self.assertRaisesWith("its own reference category", cfg)

    def test_scope_by_card_process_name(self):
        """`processes:` names the card process, as the datacard maker matches it, also when
        its histogram name differs."""
        cfg = card()
        tt = process(cfg, "TT")
        tt["process"], tt["hist_name"] = "TTbar", "TT"
        for u in cfg["uncertainties"]:
            procs = u.get("processes")
            if isinstance(procs, list) and "TT" in procs:
                u["processes"] = ["TTbar" if q == "TT" else q for q in procs]
        model = PreprocessModel(cfg)
        self.assertEqual(model.members["TT"], ["TT", "ttV"])
        self.assertIn("PDF_alphas", model.member_uncs)
        self.assertIn(
            ("CMS_bbww_ttcr_shape_res2b", "TT", 0.02), model.tilts("eE", "TT_res2b")
        )

    def test_card_is_not_modified(self):
        cfg = card()
        before = copy.deepcopy(cfg)
        PreprocessModel(cfg)
        self.assertEqual(cfg, before)


class BinningFile(unittest.TestCase):
    def test_n_bins_and_gates_optional_with_saved_edges(self):
        only_windows = {"windows": "config/sub_regions/hme_DL_v2610.yaml"}
        check_binning_config(only_windows, edges=True)
        with self.assertRaises(PreprocessConfigError) as ctx:
            check_binning_config(only_windows)
        self.assertIn("are required", str(ctx.exception))


class SavedEdges(unittest.TestCase):
    def test_read(self):
        data = {"binning": {"Run3_22_23": {"500": {}}}}
        self.assertEqual(read_saved_edges(data, "Run3_22_23"), {"500": {}})
        with self.assertRaises(PreprocessConfigError):
            read_saved_edges(data, "Run3_2022")
        with self.assertRaises(PreprocessConfigError):
            read_saved_edges({"edges": []}, "Run3_22_23")


if __name__ == "__main__":
    unittest.main()
