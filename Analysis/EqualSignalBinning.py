"""Equal-signal binning of the 2D DNN x HME fit inputs inside fixed HME windows.

The `preprocess:` step of a datacard configuration (StatInference's PreprocessShapesTask calls
it with --input, --output, --era and --config). For every mass point and datacard category it
cuts the category's HME window out of the 2D merged histograms, merges the merged-histogram
processes into the card's backgrounds, and bins the DNN score inside the window into bins of
equal signal, from the top of the score down. A bin is only closed once every background in it
passes the gates of the binning configuration, for the nominal and for every variation.

Input:  <input>/<era>/<variable>/<variable>.root for every era of the card (the merged
        histograms), with <channel>/<input category>/<process>[_<uncertainty>_Up|Down] 2D
        histograms, DNN score on x and HME mass on y.
Output: the same layout under <output>, 1D in the DNN score, one file per era of the card, and
        <output>/binning.json with the bins of every window.

What is merged, dropped and estimated, the windows, the gates and the target number of bins per
window are all in the binning configuration (--binning-config); see
config/Datacards/binning_eqsig_DL_v2610.yaml for what each entry means.
"""

import argparse
import json
import os
from string import Template

import numpy as np
import uproot
import yaml
import ROOT

ROOT.gROOT.SetBatch(True)
# config/... paths are relative to the analysis area this script belongs to
ANALYSIS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analysis_file(path):
    return path if os.path.isabs(path) else os.path.join(ANALYSIS_PATH, path)


def signed(value):
    """[down, up] relative shifts of a member uncertainty; a single number v is [-v, v]."""
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return -float(value), float(value)


def cumulative(arr):
    return np.concatenate([[0.0], np.cumsum(arr)])


def in_range(c, i, j):
    """Sum of fine bins i..j from a cumulative array."""
    return c[j + 1] - c[i]


def window_mask(y_edges, low, high):
    """Flow-inclusive mask of the y (HME) bins inside [low, high]; a cut on the first (last)
    edge takes the underflow (overflow) too."""
    mask = np.zeros(len(y_edges) + 1, bool)
    mask[1:-1] = (y_edges[:-1] >= low - 1e-6) & (y_edges[1:] <= high + 1e-6)
    mask[0] = low <= y_edges[0] + 1e-6
    mask[-1] = high >= y_edges[-1] - 1e-6
    return mask


def project(hist, mask):
    """(values, variances) of a flow-inclusive 2D histogram summed over the masked y bins, the
    x flows folded into the first and last bin."""
    out = []
    for arr in (hist.values(flow=True), hist.variances(flow=True)):
        p = arr[:, mask].sum(axis=1)
        q = p[1:-1].copy()
        q[0] += p[0]
        q[-1] += p[-1]
        out.append(q)
    return tuple(out)


def variation_suffixes(names, process):
    """'_<uncertainty>_Up|Down' of every variation of `process` among `names`."""
    return {
        n[len(process) :]
        for n in names
        if n.startswith(process + "_") and n.endswith(("_Up", "_Down"))
    }


class Card:
    """What the binning needs from the datacard configuration."""

    def __init__(self, path, era):
        cfg = yaml.safe_load(open(path))
        self.eras = cfg.get("era_groups", {}).get(era, [era])
        self.channels = cfg["channels"]
        mapping = cfg.get("input_categories") or {}
        # card category -> its directory in the input (and output) files
        self.categories = {c: mapping.get(c, c) for c in cfg["categories"]}
        model = cfg["model"]
        self.param = model["parameters"][0]
        self.pattern = model["input_file_pattern"]
        self.signals, self.backgrounds, self.data, masses = [], [], [], set()
        for p in cfg["processes"]:
            if isinstance(p, str):
                self.backgrounds.append(p)
                continue
            masses.update(p.get("param_values") or [])
            name = p.get("hist_name", p["process"])
            if p.get("is_signal"):
                self.signals.append(name)
            elif p.get("is_data"):
                if not p.get("is_asimov_data"):
                    self.data.append(name)
            else:
                self.backgrounds.append(name)
        self.masses = sorted(masses)

    def input_file(self, era, mass):
        return Template(self.pattern).substitute(ERA=era, **{self.param: mass})

    def resolved(self, names, mass):
        return [Template(n).safe_substitute(**{self.param: mass}) for n in names]


class Binning:
    def __init__(self, card, era, cfg, input_dir, output_dir):
        self.card, self.era, self.input_dir, self.output_dir = (
            card,
            era,
            input_dir,
            output_dir,
        )
        self.windows = yaml.safe_load(open(analysis_file(cfg["windows"])))["windows"]
        self.n_bins = cfg["n_bins"]
        self.gates = cfg["gates"]
        self.members = {
            b: cfg.get("backgrounds", {}).get(b, [b]) for b in card.backgrounds
        }
        self.drop = cfg.get("drop") or {}
        self.lowmass = cfg.get("lowmass_dy")
        self.member_uncs = {
            name: [(t["member"], signed(t["value"])) for t in terms]
            for name, terms in (cfg.get("member_uncertainties") or {}).items()
        }
        self.record = {}

    @staticmethod
    def window_key(directory):
        """The windows file keys a category by its base name: SR/res2b -> res2b."""
        return directory.rsplit("/", 1)[-1]

    def lowmass_applies(self, directory):
        return (
            self.lowmass is not None
            and self.window_key(directory) in self.lowmass["categories"]
        )

    def read(self, files, channel, directory, window):
        """{era: {merged-histogram name: (values, variances)}} of one category, projected on
        `window`, plus the DNN-score bin edges."""
        out, x_edges = {}, None
        for era, f in files.items():
            d = f[f"{channel}/{directory}"]
            names = d.keys(recursive=False, cycle=False)
            first = d[names[0]]
            x_edges = first.axes[0].edges()
            mask = window_mask(first.axes[1].edges(), *window)
            out[era] = {n: project(d[n], mask) for n in names}
        return out, x_edges

    def build_processes(self, raw, category_key, lowmass, mass):
        """{output process: {suffix: (values, variances)}} of one era, suffix '' = nominal.

        With `lowmass` the low-mass member is left out of DY here; it is added per bin
        range from the transfer estimate (see bin_window)."""
        dropped = {m for m, cats in self.drop.items() if category_key in cats}
        out = {}
        for proc, members in self.members.items():
            members = [m for m in members if m in raw and m not in dropped]
            if lowmass and proc == "DY":
                members = [m for m in members if m != self.lowmass["process"]]
            if not members:
                continue
            suffixes = set().union(*(variation_suffixes(raw, m) for m in members))
            shapes = {}
            for s in [""] + sorted(suffixes):
                parts = [raw.get(m + s, raw[m]) for m in members]
                shapes[s] = (sum(p[0] for p in parts), sum(p[1] for p in parts))
            for unc, terms in self.member_uncs.items():
                terms = [(m, v) for m, v in terms if m in members]
                if not terms:
                    continue
                for k, direction in ((0, "Down"), (1, "Up")):
                    y = shapes[""][0] + sum(v[k] * raw[m][0] for m, v in terms)
                    shapes[f"_{unc}_{direction}"] = (y, shapes[""][1])
            out[proc] = shapes
        for name in self.card.resolved(self.card.signals, mass) + self.card.data:
            if name in raw:
                out[name] = {
                    s: raw[name + s]
                    for s in [""] + sorted(variation_suffixes(raw, name))
                }
        return out

    def lowmass_factors(self, suffixes):
        """How the low-mass estimate enters each DY variation: at its nominal, scaled by a
        member uncertainty on the low-mass process, or 2x / 0x for its own nuisance."""
        factors = {}
        lm = self.lowmass
        for s in suffixes:
            f = 1.0
            for unc, terms in self.member_uncs.items():
                for k, direction in ((0, "Down"), (1, "Up")):
                    if s == f"_{unc}_{direction}":
                        f += sum(v[k] for m, v in terms if m == lm["process"])
            factors[s] = f
        factors[f"_{lm['nuisance']}_Up"] = 2.0
        factors[f"_{lm['nuisance']}_Down"] = 0.0
        return factors

    def bin_window(self, mass, channel, category, directory, files):
        key = self.window_key(directory)
        window = self.windows[mass][key]
        raw, x_edges = self.read(files, channel, directory, window)
        nb = len(x_edges) - 1
        lowmass = self.lowmass_applies(directory)
        per_era = {
            e: self.build_processes(r, key, lowmass, mass) for e, r in raw.items()
        }
        bkgs = [b for b in self.members if any(b in p for p in per_era.values())]
        signals = self.card.resolved(self.card.signals, mass)

        def summed(proc, s):
            parts = [p[proc].get(s, p[proc][""]) for p in per_era.values() if proc in p]
            return sum(p[0] for p in parts), sum(p[1] for p in parts)

        suffixes = {
            b: sorted(
                set().union(*(set(p[b]) for p in per_era.values() if b in p)) - {""}
            )
            for b in bkgs
        }
        L, factors = None, {}
        if lowmass and "DY" in bkgs:
            lm = self.lowmass
            ref_dir = next(
                d
                for d in self.card.categories.values()
                if self.window_key(d) == lm["reference"]
            )
            # the reference category projected on THIS category's window
            ref, _ = self.read(files, channel, ref_dir, window)
            zero = np.zeros(nb)
            low = [
                sum(r.get(lm["process"], (zero, zero))[k] for r in ref.values())
                for k in (0, 1)
            ]
            transfer = [
                sum(r.get(lm["transfer"], (zero, zero))[k] for r in ref.values())
                for k in (0, 1)
            ]
            c_low, c_low_w = cumulative(low[0]), cumulative(np.maximum(low[1], 0.0))
            c_tr = cumulative(transfer[0])
            c_m50 = cumulative(summed("DY", "")[0])

            def estimate(i, j):
                """(low-mass DY, its variance) over fine bins i..j"""
                t = in_range(c_tr, i, j)
                if j < i or t <= 0:
                    return 0.0, 0.0
                r = in_range(c_m50, i, j) / t
                return in_range(c_low, i, j) * r, in_range(c_low_w, i, j) * r * r

            L = estimate
            factors = self.lowmass_factors(suffixes["DY"])
            suffixes["DY"] = sorted(set(suffixes["DY"]) | set(factors))

        series = {
            b: {
                s: tuple(cumulative(a) for a in summed(b, s))
                for s in [""] + suffixes[b]
            }
            for b in bkgs
        }
        signal = np.zeros(nb)
        for s in signals:
            if any(s in p for p in per_era.values()):
                signal = signal + summed(s, "")[0]
        c_sig = cumulative(signal)

        def get(b, s, i, j):
            c = series[b][s if s in series[b] else ""]
            y, w = in_range(c[0], i, j), in_range(c[1], i, j)
            if L is not None and b == "DY":
                lv, lw = L(i, j)
                f = factors.get(s, 1.0)
                # a variation without the estimate (dy_lowmass Down) has M>50's MC stat alone
                y, w = y + f * lv, w + (lw if f > 0 else 0.0)
            return y, w

        g = self.gates
        totals = {b: get(b, "", 0, nb - 1)[0] for b in bkgs}
        exempt = {
            b for b in bkgs if totals[b] < g["exempt_fraction"] * sum(totals.values())
        }

        def passes(i, j):
            nominal = {b: get(b, "", i, j) for b in bkgs}
            bt = sum(v[0] for v in nominal.values())
            bw = sum(v[1] for v in nominal.values())
            if bt < g["min_bkg"] or bw <= 0 or bt * bt / bw < g["min_bkg_neff"]:
                return False
            for b in bkgs:
                y, w = nominal[b]
                if y < -1e-12:
                    return False
                strict = b not in exempt
                if strict and (
                    y <= g["min_each"] or w <= 0 or y * y / w < g["min_each_neff"]
                ):
                    return False
                for s in suffixes[b]:
                    yv, wv = get(b, s, i, j)
                    if yv < -1e-12:
                        return False
                    if strict and (wv <= 0 or yv * yv / wv < g["min_variation_neff"]):
                        return False
            return True

        n = self.n_bins[mass][key]
        bins, hi = [], nb - 1
        # equal signal from the top of the score down, each bin as narrow as the gates allow
        while hi >= 0 and len(bins) < n - 1:
            target = in_range(c_sig, 0, hi) / (n - len(bins))
            found = next(
                (
                    i
                    for i in range(hi, -1, -1)
                    if in_range(c_sig, i, hi) >= target and passes(i, hi)
                ),
                None,
            )
            if found is None or found == 0:
                break
            bins.append((found, hi))
            hi = found - 1
        if hi >= 0:
            # the rest of the window is the lowest bin; merge upwards until it passes
            last = (0, hi)
            while not passes(*last):
                if not bins:
                    raise RuntimeError(
                        f"{channel}/{directory} MX={mass}: the whole window fails the gates"
                    )
                last = (0, bins.pop()[1])
            bins.append(last)
        bins.sort()
        edges = [float(x_edges[i]) for i, _ in bins] + [float(x_edges[-1])]

        self.record.setdefault(str(mass), {}).setdefault(channel, {})[category] = {
            "slices": [
                {"y_range": [float(window[0]), float(window[1])], "edges": edges}
            ],
            "n_bins_target": n,
            "n_bins": len(bins),
            "signal": [float(in_range(c_sig, i, j)) for i, j in bins],
            "background": [
                float(sum(get(b, "", i, j)[0] for b in bkgs)) for i, j in bins
            ],
        }

        # the low-mass estimate is shared out to the eras by their M>50 yield in each bin
        lowmass_by_era = {}
        if L is not None:
            m50 = {e: p["DY"][""][0] for e, p in per_era.items() if "DY" in p}
            for e in m50:
                vals = []
                for i, j in bins:
                    tot = sum(m50[x][i : j + 1].sum() for x in m50)
                    share = m50[e][i : j + 1].sum() / tot if tot > 0 else 1.0 / len(m50)
                    lv, lw = L(i, j)
                    vals.append((lv * share, lw * share))
                lowmass_by_era[e] = vals
        return per_era, bins, edges, factors, lowmass_by_era

    def run(self):
        card = self.card
        for mass in card.masses:
            files = {
                e: uproot.open(os.path.join(self.input_dir, card.input_file(e, mass)))
                for e in card.eras
            }
            out = {}
            for e in card.eras:
                path = os.path.join(self.output_dir, card.input_file(e, mass))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                out[e] = ROOT.TFile(path, "RECREATE")
            reached = {}
            for channel in card.channels:
                for category, directory in card.categories.items():
                    per_era, bins, edges, factors, lowmass_by_era = self.bin_window(
                        mass, channel, category, directory, files
                    )
                    reached[f"{channel}/{category}"] = len(bins)
                    for e, procs in per_era.items():
                        self.write(
                            out[e].mkdir(f"{channel}/{directory}", "", True),
                            procs,
                            bins,
                            edges,
                            factors,
                            lowmass_by_era.get(e),
                        )
            for f in out.values():
                f.Close()
            print(f"MX={mass}: bins per window {json.dumps(reached)}", flush=True)
        with open(os.path.join(self.output_dir, "binning.json"), "w") as f:
            json.dump({"binning": {self.era: self.record}}, f, indent=1)

    @staticmethod
    def write(directory, procs, bins, edges, factors, lowmass):
        """One era's histograms of a category: the variations that era's input has, and
        for DY with the low-mass estimate also its own variations (every era)."""
        directory.cd()
        edges = np.array(edges, dtype=float)

        def hist(name, y, w):
            h = ROOT.TH1D(name, name, len(edges) - 1, edges)
            h.Sumw2()
            for k, (i, j) in enumerate(bins, start=1):
                h.SetBinContent(k, float(y[i : j + 1].sum()))
                h.SetBinError(k, float(np.sqrt(max(w[i : j + 1].sum(), 0.0))))
            return h

        for proc, shapes in procs.items():
            names = set(shapes) - {""}
            if proc == "DY" and lowmass is not None:
                # the variations that move the estimate exist in every era
                names |= {s for s, f in factors.items() if f != 1.0}
            for s in [""] + sorted(names):
                y, w = shapes.get(s, shapes[""])
                h = hist(proc + s, y, w)
                if proc == "DY" and lowmass is not None:
                    f = factors.get(s, 1.0)
                    for k, (lv, lw) in enumerate(lowmass, start=1):
                        h.SetBinContent(k, h.GetBinContent(k) + f * lv)
                        if f > 0:
                            h.SetBinError(k, float(np.sqrt(h.GetBinError(k) ** 2 + lw)))
                h.Write()
                h.Delete()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--input", required=True, help="merged histograms, <era>/<var>/<var>.root"
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--era", required=True, help="the card's era (a group era is summed)"
    )
    ap.add_argument("--config", required=True, help="the datacard configuration")
    ap.add_argument("--binning-config", required=True)
    ap.add_argument(
        "--masses",
        default="",
        help="comma-separated subset of the card's masses (tests)",
    )
    args = ap.parse_args()

    card = Card(args.config, args.era)
    if args.masses:
        card.masses = [int(m) for m in args.masses.split(",")]
    binning = Binning(
        card,
        args.era,
        yaml.safe_load(open(analysis_file(args.binning_config))),
        args.input,
        args.output,
    )
    os.makedirs(args.output, exist_ok=True)
    binning.run()


if __name__ == "__main__":
    main()
