"""Single-lepton fakes: the data-driven process ``Fakes`` (config/processes.yaml), estimated by FLAF's
merge (FLAF Common/DataDriven.py) from the HME sub-regions of the booking plugin
(Analysis/hist_booking.py, flavour Signal_Extraction_SL_window).

For the binary DNN of a mass MX (``TwoStageDNN_M<MX>_binary``), in each channel and category of the
SL window ``SR_SL_HMESub_m<MX>``: Fakes = max(f, 0) x TT there, bin by bin, with f the relative
excess of data over the simulation,

    f = (D - O) / T - 1     (D data, O the simulated backgrounds other than TT, T TT),

measured without the data of the isolated window, separately in two parts of the DNN axis (below
and from the mass's ``split`` on), each applied to the TT bins of its part. Schemes (``scheme`` per
mass):

- H: f from the isolated HME sidebands, lower (L1+L2) and upper (H1+H2) pooled, interpolated
  linearly in HME to the window: f = f_lo + w (f_hi - f_lo), w = (u_win - u_lo) / (u_hi - u_lo),
  u the middle of each region's HME interval;
- V: f = f_iso(sidebands) / f_anti(sidebands) x f_anti(window), anti = SR_SL_Fakeable; f = 0 where
  the isolated sidebands or the anti-isolated window show no excess (f_iso <= 0 or f_anti(window)
  <= 0), not determined where the anti-isolated sidebands show none (f_anti <= 0).

A scheme whose regions are missing is a configuration error. Nuisances (declared in the process's
``nuisances``): FakesStat{Low,High}, max(f, 0) +- the statistical error of f in that part;
FakesMethod{Low,High}, max(f, 0) +- |max(f_other, 0) - max(f, 0)| with the other scheme (no shift
where its regions are missing or it is not determined). A region group with less TT than ``min_tt``
does not determine f at the nominal: Fakes = 0 in that cell (both parts, every variation and
nuisance), with a message; a split that leaves one part with too little TT in every cell while the
other part is determined somewhere is a configuration error (raised). Where a variation leaves f
undetermined in a part (no TT where the nominal had enough or, for V, no anti-isolated excess), that
part keeps the nominal f, applied to the varied TT, with a message. Data are read in the isolated
sidebands and in SR_SL_Fakeable only, never in the isolated window; no data-derived number is
printed.
"""

import ctypes
import re

import ROOT

from Analysis import hist_booking as HB

NUISANCES = ("FakesStatLow", "FakesStatHigh", "FakesMethodLow", "FakesMethodHigh")
PARTS = ("Low", "High")


class Cell:
    """The regions of one (channel, category) of one mass: names and HME intervals."""

    def __init__(self, window, sidebands, anti_window, anti_sidebands):
        self.window = window  # (name, (lo, hi))
        self.sidebands = sidebands  # {label: (name, (lo, hi))}
        self.anti_window = anti_window
        self.anti_sidebands = anti_sidebands

    def side(self, prefix, anti=False):
        regions = self.anti_sidebands if anti else self.sidebands
        return [regions[k] for k in sorted(regions) if k.startswith(prefix)]


def _check_config(cfg, model):
    required = ("variable", "hme_variable", "iso", "anti", "window", "sidebands")
    required += ("split", "scheme", "min_tt", "tt_process")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RuntimeError(f"Fakes: data_driven parameters {missing} are missing")
    undeclared = [n for n in NUISANCES if n not in cfg.get("nuisances", {})]
    if undeclared:
        raise RuntimeError(f"Fakes: data_driven.nuisances must declare {undeclared}")
    if model not in cfg["tt_process"]:
        raise RuntimeError(f"Fakes: no tt_process for physics model {model}")


def _cells(ctx, mass):
    """{(channel, category): Cell} of the mass, from the plugin's sub-regions."""
    cfg = ctx.config
    hme = cfg["hme_variable"]
    found = {}
    for sreg in HB.sub_regions(ctx.setup):
        if not HB._same(sreg["point"].get("mass"), mass):
            continue
        base, family = sreg["base_region"], sreg["sub_region"]
        if base not in (cfg["iso"], cfg["anti"]):
            continue
        if family == cfg["window"]:
            label = None
        elif family.startswith(cfg["sidebands"]):
            label = family[len(cfg["sidebands"]) :]
        else:
            continue
        for (cat, ch), intervals in sreg["intervals"].items():
            if len(intervals.get(hme, [])) != 1:
                raise RuntimeError(
                    f"Fakes: {sreg['name']} {cat} {ch} has no single {hme} interval"
                )
            (interval,) = intervals[hme]
            found.setdefault((ch, cat), {})[(base, label)] = (sreg["name"], interval)
    cells = {}
    for (ch, cat), regions in found.items():
        win = regions.get((cfg["iso"], None))
        anti_win = regions.get((cfg["anti"], None))
        if win is None:
            continue

        def sbs(base):
            return {lab: v for (b, lab), v in regions.items() if b == base and lab}

        cells[(ch, cat)] = Cell(win, sbs(cfg["iso"]), anti_win, sbs(cfg["anti"]))
    return cells


class Yields:
    """D, O, T of one DNN part over a group of regions, at one variation."""

    def __init__(self, D=0.0, vD=0.0, O=0.0, vO=0.0, T=0.0, vT=0.0):
        self.D, self.vD, self.O, self.vO, self.T, self.vT = D, vD, O, vO, T, vT

    def __add__(self, other):
        return Yields(*(a + b for a, b in zip(self.tuple(), other.tuple())))

    def tuple(self):
        return (self.D, self.vD, self.O, self.vO, self.T, self.vT)

    def f(self, min_tt):
        """(f, variance) or None if TT does not determine it."""
        if self.T < min_tt:
            return None
        n = self.D - self.O
        f = n / self.T - 1.0
        var = (self.vD + self.vO) / self.T**2 + n**2 * self.vT / self.T**4
        return f, var


def _integral(hist, lo_bin, hi_bin):
    err = ctypes.c_double(0.0)
    value = hist.IntegralAndError(lo_bin, hi_bin, err)
    return value, err.value**2


class Estimate:
    def __init__(self, ctx, mass):
        cfg = ctx.config
        self.ctx = ctx
        self.cfg = cfg
        self.mass = mass
        self.tt = cfg["tt_process"][ctx.setup.phys_model.name]
        if self.tt not in ctx.mc_backgrounds:
            raise RuntimeError(f"Fakes: {self.tt} is not a merged simulated background")
        self.others = [p for p in ctx.mc_backgrounds if p != self.tt]
        self.min_tt = float(cfg["min_tt"])
        self.split = float(cfg["split"][mass])
        self.scheme = cfg["scheme"][mass]
        if self.scheme not in ("H", "V"):
            raise RuntimeError(f"Fakes: scheme {self.scheme!r} of M{mass}: H or V")
        self.parts = (
            None  # [(lo_bin, hi_bin), (lo_bin, hi_bin)], set from the first histogram
        )
        self._iso_window = (
            None  # the isolated window of the current cell: its data never read
        )

    def _parts(self, hist):
        if self.parts is None:
            axis = hist.GetXaxis()
            b = axis.FindFixBin(self.split)
            if abs(axis.GetBinLowEdge(b) - self.split) > 1e-6 * max(
                1.0, abs(self.split)
            ):
                raise RuntimeError(
                    f"Fakes: split {self.split} of M{self.mass} is not a bin edge of "
                    f"{self.ctx.var}"
                )
            self.parts = [(0, b - 1), (b, axis.GetNbins() + 1)]
        return self.parts

    def yields(self, ch, cat, regions, unc, scale):
        """[Yields of each part] summed over ``regions`` (names) at (unc, scale)."""
        total = [Yields(), Yields()]
        for name in regions:
            key = (ch, name, cat)
            if name == self._iso_window:
                raise AssertionError("the isolated window's data are never read")
            hists = {"D": self.ctx.require(self.ctx.data_process, key, unc, scale)}
            hists["T"] = self.ctx.require(self.tt, key, unc, scale)
            other = None
            for p in self.others:
                h = self.ctx.get(p, key, unc, scale)
                if h is None:
                    continue
                if other is None:
                    other = h
                else:
                    other.Add(h)
            for i, (lo, hi) in enumerate(self._parts(hists["T"])):
                D, vD = _integral(hists["D"], lo, hi)
                T, vT = _integral(hists["T"], lo, hi)
                O, vO = _integral(other, lo, hi) if other is not None else (0.0, 0.0)
                total[i] = total[i] + Yields(D, vD, O, vO, T, vT)
        return total

    @staticmethod
    def available(scheme, cell):
        """Whether the cell has the regions of a scheme."""
        if scheme == "H":
            return bool(cell.side("L")) and bool(cell.side("H"))
        return (
            cell.anti_window is not None
            and bool(cell.anti_sidebands)
            and bool(cell.sidebands)
        )

    def f_of(self, scheme, cell, ch, cat, unc, scale):
        """[(f, var) or None for each part] of a scheme in one cell. Whether TT determines f is
        decided at the nominal (min_tt); a variation needs TT > 0 and, for V, f_anti > 0.
        """
        min_tt = self.min_tt if (unc, scale) == ("Central", "Central") else 1e-300
        names = lambda regions: [n for n, _ in regions]  # noqa: E731
        lower, upper = cell.side("L"), cell.side("H")
        if scheme == "H":
            y_lo = self.yields(ch, cat, names(lower), unc, scale)
            y_hi = self.yields(ch, cat, names(upper), unc, scale)
            u_lo = 0.5 * (min(i[0] for _, i in lower) + max(i[1] for _, i in lower))
            u_hi = 0.5 * (min(i[0] for _, i in upper) + max(i[1] for _, i in upper))
            u_win = 0.5 * sum(cell.window[1])
            w = (u_win - u_lo) / (u_hi - u_lo)
            out = []
            for a, b in zip(y_lo, y_hi):
                fa, fb = a.f(min_tt), b.f(min_tt)
                if fa is None or fb is None:
                    out.append(None)
                else:
                    out.append(
                        (
                            fa[0] + w * (fb[0] - fa[0]),
                            (1 - w) ** 2 * fa[1] + w**2 * fb[1],
                        )
                    )
            return out
        # V
        iso = self.yields(ch, cat, names(lower + upper), unc, scale)
        anti = self.yields(
            ch, cat, names(cell.side("L", True) + cell.side("H", True)), unc, scale
        )
        anti_win = self.yields(ch, cat, [cell.anti_window[0]], unc, scale)
        out = []
        for a, b, c in zip(iso, anti, anti_win):
            fi, fa, fw = a.f(min_tt), b.f(min_tt), c.f(min_tt)
            if fi is None or fa is None or fw is None or fa[0] <= 0:
                out.append(None)
                continue
            # no excess in the isolated sidebands or in the anti-isolated window: no fakes
            f = fi[0] / fa[0] * fw[0] if fi[0] > 0 and fw[0] > 0 else 0.0
            var = (
                (fw[0] / fa[0]) ** 2 * fi[1]
                + (fi[0] * fw[0] / fa[0] ** 2) ** 2 * fa[1]
                + (fi[0] / fa[0]) ** 2 * fw[1]
            )
            out.append((f, var))
        return out

    def fakes(self, tt_window, fs):
        """max(f, 0) x TT, f of each part applied to its bins."""
        hist = tt_window.Clone()
        hist.SetDirectory(ROOT.nullptr)
        ROOT.SetOwnership(hist, True)
        for (lo, hi), f in zip(self._parts(tt_window), fs):
            scale = max(f, 0.0)
            for b in range(lo, hi + 1):
                hist.SetBinContent(b, scale * tt_window.GetBinContent(b))
                hist.SetBinError(b, scale * tt_window.GetBinError(b))
        return hist

    def tt_short(self, scheme, cell, ch, cat):
        """[for each part: whether a region group of the scheme has less TT than min_tt] at the
        nominal."""
        names = lambda regions: [n for n, _ in regions]  # noqa: E731
        lower, upper = cell.side("L"), cell.side("H")
        if scheme == "H":
            groups = [names(lower), names(upper)]
        else:
            groups = [
                names(lower + upper),
                names(cell.side("L", True) + cell.side("H", True)),
                [cell.anti_window[0]],
            ]
        short = [False, False]
        for group in groups:
            for i, y in enumerate(self.yields(ch, cat, group, "Central", "Central")):
                short[i] = short[i] or y.T < self.min_tt
        return short

    def check_split(self, cells):
        """A split that leaves one part of the DNN with too little TT in every cell, while the other
        part is determined somewhere, is misplaced: raise rather than give Fakes = 0 to the whole
        mass."""
        short, determined = [True, True], [False, False]
        for (ch, cat), cell in sorted(cells.items()):
            key = (ch, cell.window[0], cat)
            if key not in self.ctx.key_dirs or not self.available(self.scheme, cell):
                continue
            self._iso_window = cell.window[0]
            cell_short = self.tt_short(self.scheme, cell, ch, cat)
            nominal = self.f_of(self.scheme, cell, ch, cat, "Central", "Central")
            for i in range(2):
                short[i] = short[i] and cell_short[i]
                determined[i] = determined[i] or nominal[i] is not None
        for i, j in ((0, 1), (1, 0)):
            if short[i] and determined[j]:
                raise RuntimeError(
                    f"Fakes {self.ctx.era} M{self.mass}: the {PARTS[i]} part of the DNN "
                    f"(split {self.split}) has less TT than min_tt in a region group of every "
                    f"cell, while the {PARTS[j]} part is determined: the split is misplaced"
                )

    def run(self, cells):
        ctx = self.ctx
        out = {}
        other = "V" if self.scheme == "H" else "H"
        if ctx.nominal_pass:
            self.check_split(cells)
        for (ch, cat), cell in sorted(cells.items()):
            self._iso_window = cell.window[0]
            key = (ch, cell.window[0], cat)
            what = f"Fakes {ctx.era} M{self.mass} {ch} {cat}"
            if tuple(key) not in ctx.key_dirs:
                continue
            if not self.available(self.scheme, cell):
                raise RuntimeError(
                    f"{what}: scheme {self.scheme} needs regions the cell does not have "
                    "(the sidebands / SR_SL_Fakeable sub-regions of the flavour)"
                )
            nominal = self.f_of(self.scheme, cell, ch, cat, "Central", "Central")
            if any(f is None for f in nominal):
                # no estimate in this cell: zero templates (nominal, variations, nuisances), so
                # that a sum of eras and the datacards see the process in every cell
                if ctx.nominal_pass:
                    print(
                        f"{what}: not estimated (scheme {self.scheme} not determined), "
                        "Fakes = 0"
                    )
                zero = [0.0, 0.0]
                for unc, scale in ctx.variations:
                    tt = ctx.require(self.tt, key, unc, scale)
                    out[(key, (unc, scale))] = self.fakes(tt, zero)
                if ctx.nominal_pass:
                    tt = ctx.require(self.tt, key)
                    for source in NUISANCES:
                        for name in ("Up", "Down"):
                            out[(key, (source, name))] = self.fakes(tt, zero)
                continue
            for unc, scale in ctx.variations:
                fs = (
                    nominal
                    if (unc, scale) == ("Central", "Central")
                    else self.f_of(self.scheme, cell, ch, cat, unc, scale)
                )
                if any(f is None for f in fs):
                    # f determined at the nominal but not at this variation (no TT, or for V
                    # no anti-isolated excess): that part keeps the nominal f, applied to the
                    # varied TT
                    parts = [p for p, f in zip(PARTS, fs) if f is None]
                    print(
                        f"{what}: f not determined at ({unc}, {scale}) in "
                        f"{', '.join(parts)}: the nominal f is used there"
                    )
                    fs = [n if f is None else f for f, n in zip(fs, nominal)]
                tt = ctx.require(self.tt, key, unc, scale)
                out[(key, (unc, scale))] = self.fakes(tt, [f for f, _ in fs])
            if not ctx.nominal_pass:
                continue
            tt = ctx.require(self.tt, key)
            f_nom = [f for f, _ in nominal]
            alt = (
                self.f_of(other, cell, ch, cat, "Central", "Central")
                if self.available(other, cell)
                else [None, None]
            )
            for i, part in enumerate(PARTS):
                sigma = nominal[i][1] ** 0.5
                delta = (
                    None
                    if alt[i] is None
                    else abs(max(alt[i][0], 0) - max(f_nom[i], 0))
                )
                for source, shift in (
                    (f"FakesStat{part}", sigma),
                    (f"FakesMethod{part}", 0.0 if delta is None else delta),
                ):
                    for name, sign in (("Up", 1), ("Down", -1)):
                        fs = list(f_nom)
                        fs[i] = max(f_nom[i], 0.0) + sign * shift
                        out[(key, (source, name))] = self.fakes(tt, fs)
            note = (
                ""
                if all(a is not None for a in alt)
                else f", no {other} for FakesMethod"
            )
            print(f"{what}: scheme {self.scheme}{note}")
        return out


def estimate(ctx):
    """FLAF data-driven estimator of the single-lepton fakes (see the module docstring)."""
    cfg = ctx.config
    _check_config(cfg, ctx.setup.phys_model.name)
    pattern = re.escape(cfg["variable"]).replace(re.escape("{mass}"), r"(\d+)")
    match = re.fullmatch(pattern, ctx.var)
    if not match:
        return {}
    mass = int(match.group(1))
    # the masses of the SL windows only: other binary DNNs (e.g. 1600-2000) have no sub-regions
    cells = _cells(ctx, mass)
    if not cells:
        return {}
    if mass not in cfg["split"] or mass not in cfg["scheme"]:
        raise RuntimeError(f"Fakes: no split or scheme for M{mass}")
    return Estimate(ctx, mass).run(cells)
