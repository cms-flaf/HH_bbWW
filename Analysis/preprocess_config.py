"""The settings Analysis/EqualSignalBinning.py reads from a DL datacard configuration.

No ROOT here, so the parsing and its errors can be tested on their own (test/).

Two inputs, with no overlap:

* The **card** (the merged datacard configuration) holds the process model and every
  uncertainty, in `preprocess:` blocks of its process and uncertainty entries. The datacard
  maker reads entries with .get() and ignores `preprocess:`; this module reads it, strictly:

  - a background process entry:
      preprocess:
        members: [ <merged-histogram process>, ... ]   # summed into the background
        drop: { <member>: [ <card category>, ... ] }   # members left out there
        dy_template: { share_by, fit, guard_sigma, pool }  # settings of the DY templates
        lowmass_dy: { process, transfer, reference }       # settings of the low-mass estimate
  - a `type: shape` uncertainty entry, exactly one of:
      preprocess: { member_values: { <member>: v or [ down, up ], ... } }
      preprocess: { cr_tilt: <size> }
      preprocess: { dy_template: tail | fit_only }
      preprocess: { lowmass_dy: [ <factor Down>, <factor Up> ] }
    The entry's name is the nuisance written, and its processes / channels / categories are
    where it is built (for dy_template and lowmass_dy: where the estimate replaces the MC).

* The **binning configuration** (--binning-config) holds the binning only: n_bins, gates,
  control_region_bins and, for 2D inputs, windows.

Unknown keys raise, and so do the keys that used to be in the binning configuration.
"""

import math
import re

#: keys of a binning configuration
BINNING_KEYS = {"n_bins", "gates", "control_region_bins", "windows"}

#: keys that moved out of the binning configuration, and where to
MOVED_BINNING_KEYS = {
    "backgrounds": "the card's background processes, preprocess: members",
    "drop": "the card's background processes, preprocess: drop (card categories)",
    "member_uncertainties": "the systematics entries, preprocess: member_values",
    "control_region_shapes": "the systematics entries (DL/systematics/cr_shapes.yaml), "
    "preprocess: cr_tilt",
    "dy_template": "the DY process entry (preprocess: dy_template, the settings) and the "
    "systematics entries (DL/systematics/dy_template.yaml, preprocess: dy_template: "
    "tail | fit_only)",
    "lowmass_dy": "the DY process entry (preprocess: lowmass_dy, the settings) and the "
    "systematics entry CMS_bbww_dy_lowmass (preprocess: lowmass_dy: [down, up])",
}

PROCESS_KEYS = {"members", "drop", "dy_template", "lowmass_dy"}
UNCERTAINTY_KEYS = {"member_values", "cr_tilt", "dy_template", "lowmass_dy"}
DY_TEMPLATE_KEYS = {"share_by", "fit", "guard_sigma", "pool"}
DY_TEMPLATE_REQUIRED = {"share_by"}
FIT_KEYS = {"deg", "shift", "merge"}
LOWMASS_KEYS = {"process", "transfer", "reference"}
TEMPLATE_MODES = ("tail", "fit_only")


class PreprocessConfigError(ValueError):
    """A preprocess setting that is missing, unknown or inconsistent."""


def matches(value, patterns):
    """StatInference's scope rule (dc_make/uncertainty.py, Uncertainty.hasMatch): no
    patterns match everything, a pattern starting with ^ is a regex, else an exact name.
    """
    if not patterns:
        return True
    return any(
        re.match(p, value) if p.startswith("^") else value == p for p in patterns
    )


def pattern_list(entry, key):
    v = entry.get(key)
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def signed(value, what):
    """[down, up] relative shifts: a number v is [-v, v]."""
    if isinstance(value, bool):
        raise PreprocessConfigError(f"{what}: {value!r} is not a number")
    if isinstance(value, (int, float)):
        pair = (-float(value), float(value))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
        ):
            raise PreprocessConfigError(f"{what}: {value!r} is not two numbers")
        pair = (float(value[0]), float(value[1]))
    else:
        raise PreprocessConfigError(
            f"{what}: {value!r} must be a number v (for [-v, v]) or [down, up]"
        )
    if not all(math.isfinite(v) for v in pair):
        raise PreprocessConfigError(f"{what}: {value!r} is not finite")
    return pair


def check_binning_config(cfg, path="the binning configuration"):
    """Raise unless `cfg` holds binning only."""
    if not isinstance(cfg, dict):
        raise PreprocessConfigError(f"{path}: expected a mapping")
    moved = [k for k in cfg if k in MOVED_BINNING_KEYS]
    if moved:
        raise PreprocessConfigError(
            f"{path}: "
            + "; ".join(f"`{k}` is now in {MOVED_BINNING_KEYS[k]}" for k in moved)
            + ". The binning configuration holds the binning only "
            f"({', '.join(sorted(BINNING_KEYS))}); see docs/stat_inference.md."
        )
    if "single_bin_regions" in cfg:
        raise PreprocessConfigError(
            f"{path}: single_bin_regions is replaced by control_region_bins: "
            "{region: number of bins}"
        )
    unknown = sorted(set(cfg) - BINNING_KEYS)
    if unknown:
        raise PreprocessConfigError(
            f"{path}: unknown keys {unknown}; allowed: {sorted(BINNING_KEYS)}"
        )
    missing = sorted({"n_bins", "gates"} - set(cfg))
    if missing:
        raise PreprocessConfigError(f"{path}: {missing} are required")
    if "min_each" in (cfg.get("gates") or {}):
        raise PreprocessConfigError(
            f"{path}: gates: min_each is no longer configurable; the per-background "
            "floor is min_bkg x exempt_fraction. Remove it."
        )


class PreprocessModel:
    """What the card says about building the backgrounds and their uncertainties.

    members      {background: [merged-histogram processes]}, in the card's process order
    drop         {member: {card categories}}
    member_uncs  {uncertainty: [(member, (down, up))]}
    cr_tilts     [(nuisance, process, size, channel patterns, {card categories})]
    templates    {(channel, card category): (mode, nuisance)}
    dy_template  the DY template settings, with `process` (the background they belong to),
                 or None
    lowmass      the low-mass estimate's settings, with `output` (the background),
                 `nuisance`, `nuisance_factors` and its scope, or None
    """

    def __init__(self, cfg):
        self.channels = list(cfg["channels"])
        self.categories = list(cfg["categories"])
        self.members, self.drop = {}, {}
        self.dy_template, self.lowmass = None, None
        self._read_processes(cfg.get("processes") or [])
        self.owner = {}  # member -> its background
        for b, ms in self.members.items():
            for m in ms:
                if m in self.owner:
                    raise PreprocessConfigError(
                        f"process {m} is a member of both {self.owner[m]} and {b}"
                    )
                self.owner[m] = b
        self._check_settings()
        self.member_uncs, self.cr_tilts, self.templates = {}, [], {}
        lowmass_entries = []
        for u in cfg.get("uncertainties") or []:
            if isinstance(u, dict) and "preprocess" in u:
                self._read_uncertainty(u, lowmass_entries)
        self._check_estimates(lowmass_entries)

    # -- processes -------------------------------------------------------------------

    def _read_processes(self, processes):
        for p in processes:
            if isinstance(p, str):
                self.members[p] = [p]
                continue
            name = p.get("hist_name", p["process"])
            pre = p.get("preprocess")
            if p.get("is_signal") or p.get("is_data"):
                if pre is not None:
                    raise PreprocessConfigError(
                        f"process {p['process']}: preprocess: is for backgrounds only"
                    )
                continue
            if pre is None:
                self.members[name] = [name]
                continue
            what = f"process {name}: preprocess"
            if not isinstance(pre, dict):
                raise PreprocessConfigError(f"{what} must be a mapping")
            unknown = sorted(set(pre) - PROCESS_KEYS)
            if unknown:
                raise PreprocessConfigError(
                    f"{what}: unknown keys {unknown}; allowed: {sorted(PROCESS_KEYS)}"
                )
            members = pre.get("members", [name])
            if (
                not isinstance(members, list)
                or not members
                or not all(isinstance(m, str) for m in members)
                or len(set(members)) != len(members)
            ):
                raise PreprocessConfigError(
                    f"{what}: members must be a non-empty list of distinct process names"
                )
            self.members[name] = list(members)
            for member, cats in (pre.get("drop") or {}).items():
                if member not in members:
                    raise PreprocessConfigError(
                        f"{what}: drop names {member}, which is not one of its members "
                        f"{members}"
                    )
                cats = [cats] if isinstance(cats, str) else list(cats or [])
                bad = [c for c in cats if c not in self.categories]
                if bad or not cats:
                    raise PreprocessConfigError(
                        f"{what}: drop {member}: {bad or 'no categories'} "
                        f"(card categories: {self.categories})"
                    )
                self.drop[member] = set(cats)
            for key in ("dy_template", "lowmass_dy"):
                if key in pre:
                    if getattr(self, _attr(key)) is not None:
                        raise PreprocessConfigError(
                            f"{what}: {key} is set on two processes "
                            f"({getattr(self, _attr(key))['process']} and {name})"
                        )
                    setattr(self, _attr(key), self._read_estimate(key, pre[key], name))

    def _read_estimate(self, key, settings, name):
        what = f"process {name}: preprocess: {key}"
        if not isinstance(settings, dict):
            raise PreprocessConfigError(f"{what} must be a mapping")
        allowed = DY_TEMPLATE_KEYS if key == "dy_template" else LOWMASS_KEYS
        required = DY_TEMPLATE_REQUIRED if key == "dy_template" else LOWMASS_KEYS
        unknown = sorted(set(settings) - allowed)
        missing = sorted(required - set(settings))
        if unknown or missing:
            raise PreprocessConfigError(
                f"{what}: "
                + (f"unknown keys {unknown}; " if unknown else "")
                + (f"missing {missing}; " if missing else "")
                + f"allowed: {sorted(allowed)}"
            )
        out = dict(settings)
        if key == "dy_template":
            out["process"] = name
            fit = out.get("fit") or {}
            bad = sorted(set(fit) - FIT_KEYS)
            if bad:
                raise PreprocessConfigError(
                    f"{what}: fit: unknown keys {bad}; allowed: {sorted(FIT_KEYS)}"
                )
            out["fit"] = dict(fit)
            out["guard_sigma"] = float(out.get("guard_sigma", 3.0))
            pool = out.get("pool") or []
            seen = set()
            for group in pool:
                if not isinstance(group, list) or not group:
                    raise PreprocessConfigError(
                        f"{what}: pool must be a list of channel lists"
                    )
                for ch in group:
                    if ch not in self.channels or ch in seen:
                        raise PreprocessConfigError(
                            f"{what}: pool: channel {ch} is "
                            + ("in two groups" if ch in seen else "not a card channel")
                        )
                    seen.add(ch)
            out["pool"] = [list(g) for g in pool]
        else:
            out["output"] = name
        return out

    def _check_settings(self):
        t = self.dy_template
        if t is not None and self.owner.get(t["share_by"]) != t["process"]:
            raise PreprocessConfigError(
                f"process {t['process']}: dy_template: share_by {t['share_by']} is not one "
                f"of its members {self.members[t['process']]}"
            )
        lm = self.lowmass
        if lm is not None:
            for key in ("process", "transfer"):
                if self.owner.get(lm[key]) != lm["output"]:
                    raise PreprocessConfigError(
                        f"process {lm['output']}: lowmass_dy: {key} {lm[key]} is not one "
                        f"of its members {self.members[lm['output']]}"
                    )

    # -- uncertainties ---------------------------------------------------------------

    def _read_uncertainty(self, u, lowmass_entries):
        name = u["name"]
        pre = u["preprocess"]
        what = f"uncertainty {name}: preprocess"
        if u.get("type") != "shape":
            raise PreprocessConfigError(
                f"{what}: the preprocess step builds shapes, so its type must be shape "
                f"(not {u.get('type')})"
            )
        if "members" in u:
            raise PreprocessConfigError(
                f"{what}: an uncertainty with members: is reduced by the datacard maker; "
                "it cannot also be built by the preprocess step"
            )
        if not isinstance(pre, dict) or len(pre) != 1:
            raise PreprocessConfigError(
                f"{what} must be a mapping with exactly one of {sorted(UNCERTAINTY_KEYS)}"
            )
        ((key, value),) = pre.items()
        if key not in UNCERTAINTY_KEYS:
            raise PreprocessConfigError(
                f"{what}: unknown key {key}; allowed: {sorted(UNCERTAINTY_KEYS)}"
            )
        procs = [b for b in self.members if matches(b, pattern_list(u, "processes"))]
        channels = pattern_list(u, "channels")
        cats = [c for c in self.categories if matches(c, pattern_list(u, "categories"))]
        if key == "member_values":
            self._read_member_values(name, value, procs, what)
            return
        # the others are computed on the summed eras: no era scope
        if u.get("eras"):
            raise PreprocessConfigError(
                f"{what}: {key} is computed on the eras summed; it cannot have eras:"
            )
        if not cats:
            raise PreprocessConfigError(
                f"{what}: its categories match no card category"
            )
        if key == "cr_tilt":
            if len(procs) != 1 or len(pattern_list(u, "processes")) != 1:
                raise PreprocessConfigError(
                    f"{what}: cr_tilt needs exactly one background in processes:"
                )
            if not u.get("categories"):
                raise PreprocessConfigError(f"{what}: cr_tilt needs categories:")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PreprocessConfigError(f"{what}: cr_tilt must be a number (size)")
            self.cr_tilts.append((name, procs[0], float(value), channels, set(cats)))
        elif key == "dy_template":
            t = self.dy_template
            if t is None:
                raise PreprocessConfigError(
                    f"{what}: no process has preprocess: dy_template settings"
                )
            if value not in TEMPLATE_MODES:
                raise PreprocessConfigError(
                    f"{what}: dy_template must be one of {list(TEMPLATE_MODES)}"
                )
            if procs != [t["process"]] or not u.get("processes"):
                raise PreprocessConfigError(
                    f"{what}: processes must be [{t['process']}], the process with the "
                    "dy_template settings"
                )
            if not u.get("channels") or not u.get("categories"):
                raise PreprocessConfigError(
                    f"{what}: a DY template needs explicit channels: and categories:"
                )
            for ch in self.channels:
                if not matches(ch, channels):
                    continue
                for c in cats:
                    if (ch, c) in self.templates:
                        raise PreprocessConfigError(
                            f"{what}: {ch} {c} already has the DY template "
                            f"{self.templates[(ch, c)][1]}"
                        )
                    self.templates[(ch, c)] = (value, name)
        else:  # lowmass_dy
            lm = self.lowmass
            if lm is None:
                raise PreprocessConfigError(
                    f"{what}: no process has preprocess: lowmass_dy settings"
                )
            if procs != [lm["output"]] or not u.get("processes"):
                raise PreprocessConfigError(
                    f"{what}: processes must be [{lm['output']}], the process with the "
                    "lowmass_dy settings"
                )
            if not u.get("categories"):
                raise PreprocessConfigError(f"{what}: lowmass_dy needs categories:")
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in value
                )
            ):
                raise PreprocessConfigError(
                    f"{what}: lowmass_dy must be the estimate's factors [Down, Up]"
                )
            lowmass_entries.append((name, value, channels, cats))

    def _read_member_values(self, name, values, procs, what):
        if not isinstance(values, dict) or not values:
            raise PreprocessConfigError(
                f"{what}: member_values must map members to values"
            )
        terms = []
        for member, v in values.items():
            if member not in self.owner:
                raise PreprocessConfigError(
                    f"{what}: member_values: {member} is a member of no background "
                    f"({self.members})"
                )
            terms.append((member, signed(v, f"{what}: member_values: {member}")))
        holders = {self.owner[m] for m, _ in terms}
        if set(procs) != holders:
            raise PreprocessConfigError(
                f"{what}: its processes ({sorted(procs)}) must be the backgrounds holding "
                f"its members ({sorted(holders)})"
            )
        self.member_uncs[name] = terms

    def _check_estimates(self, lowmass_entries):
        t = self.dy_template
        if t is not None:
            if not self.templates:
                raise PreprocessConfigError(
                    f"process {t['process']}: dy_template settings, but no uncertainty "
                    "has preprocess: dy_template (its nuisance, which also says where "
                    "the template applies)"
                )
            # a pooled fit-only template: every channel of the pool is fit-only there
            for (ch, c), (mode, name) in self.templates.items():
                if mode != "fit_only":
                    continue
                group = next((g for g in t["pool"] if ch in g), [ch])
                for other in group:
                    if self.templates.get((other, c), (None,))[0] != "fit_only":
                        raise PreprocessConfigError(
                            f"uncertainty {name}: {ch} {c} is fitted pooled with {other}, "
                            f"which has no fit_only DY template in {c}"
                        )
        lm = self.lowmass
        if lm is not None or lowmass_entries:
            if len(lowmass_entries) != 1:
                raise PreprocessConfigError(
                    "the low-mass DY estimate needs exactly one uncertainty with "
                    f"preprocess: lowmass_dy (found {[e[0] for e in lowmass_entries]})"
                )
            name, factors, channels, cats = lowmass_entries[0]
            lm["nuisance"] = name
            lm["nuisance_factors"] = [float(f) for f in factors]
            lm["channels"] = channels
            lm["categories"] = set(cats)

    # -- queries ---------------------------------------------------------------------

    def dropped_in(self, category):
        return {m for m, cats in self.drop.items() if category in cats}

    def lowmass_applies(self, channel, category):
        lm = self.lowmass
        return (
            lm is not None
            and category in lm["categories"]
            and matches(channel, lm["channels"])
        )

    def template(self, channel, category):
        """(mode, nuisance) of the DY template of a cell, or (None, None)."""
        return self.templates.get((channel, category), (None, None))

    def tilts(self, channel, category):
        """[(nuisance, process, size)] of the CR tilts of a cell."""
        return [
            (n, p, s)
            for n, p, s, chs, cats in self.cr_tilts
            if category in cats and matches(channel, chs)
        ]


def _attr(key):
    return "dy_template" if key == "dy_template" else "lowmass"


def read_saved_edges(data, era, path="the edges file"):
    """{mass (str): {channel: {category: record}}} of one era of a binning.json."""
    if not isinstance(data, dict) or "binning" not in data:
        raise PreprocessConfigError(f"{path}: no `binning` (expected a binning.json)")
    if era not in data["binning"]:
        raise PreprocessConfigError(
            f"{path}: no binning for era {era} (has {sorted(data['binning'])})"
        )
    return data["binning"][era]
