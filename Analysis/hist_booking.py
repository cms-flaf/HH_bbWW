"""HH->bbWW histogram booking: the FLAF booking plugin (FLAF Common/HistBooking.py) of this analysis.

global.yaml names it (``hist_booking_module: Analysis.hist_booking``). It gives FLAF

- ``regions(setup)``: the regions of the active histTuple flavour (``regions``), plus the base regions
  of its sub-regions: the selection keys are built for them, and the histTuples keep their events;
- ``booking(setup, keys, variables)``: what a HistFromNtupleProducer job fills: every variable in the
  keys of the flavour's ``regions``, and each sub-region's variables in its own keys.

A flavour::

    histTuple_flavors:
      Signal_Extraction_DL_window:
        variables: [ ... ]
        regions: [ SR, TT_CR, DY_CR ]           # booked; [ ] books sub-regions only
        sub_regions:
          - name: HMESub                         # -> SR_HMESub_m300
            cuts: config/sub_regions/hme_DL_v2610.yaml
            regions: [ SR, TT_CR, DY_CR ]        # the base regions it restricts (booked or not)
            masses: all                          # or a list of masses of the cuts file
            # parameter: MX                      # optional: the placeholder name (default MX)
            # region_name: "{region}_{name}_m{MX}"   # optional: the region name pattern (this default)

The variables of a sub-region are the flavour's, by mass: a variable whose histograms.yaml entry has
``signal_params: {mass: M}`` is filled in the sub-regions of mass M only, a variable without a mass in
those of every mass.

The cuts file maps mass -> category -> cut, optionally per channel::

    cuts:
      300:
        res2b:
          eE:   [ "DeepHME_mass >= 250 && DeepHME_mass < 400", { lep1_pt: [ 30, 500 ] } ]
          muMu: [ "DeepHME_mass >= 250 && DeepHME_mass < 400", { lep1_pt: [ 25, 500 ] } ]
        boosted: { DeepHME_mass: [ [ 260, 400 ], [ 450, 470 ] ] }

A cut is one part or a list of parts, all required:

- bin form ``{variable: [low, high] or [[low, high], ...]}``: each variable a binned variable of the
  flavour, every edge a bin edge of its binning, the cut on the stored bin index (``<var>_bin``); the
  intervals of a variable are OR'ed, the variables AND'ed; an interval from the first (to the last) edge
  takes the underflow (overflow) too. No histTuple rerun is needed to change it.
- C++ form, a string: used as written, ``{MX}`` replaced by the mass. Every variable it names must be in
  the flavour's ``fullResolution_variables``; nothing is added automatically, so a variable that is not
  there is an error. Function names (``abs(``), namespaced names (``std::abs``), numeric literals and
  ``true``/``false`` are not variables; a function's arguments are.

A category's value is a cut or a channel map ``{channel: cut}``: a dict whose keys are all channels is a
channel map, one whose keys are all binned variables a bin-form cut; anything else is an error. A channel
missing from a map, or a category missing from a mass, has no sub-region key.

Every ambiguity is an error, and a job refuses more sub-region (key, variable) pairs than
``sub_regions_max_key_vars`` (default 5000; 0 disables the check).
"""

import os
import re

import yaml

DEFAULT_PARAMETER = "MX"
DEFAULT_MAX_SUB_REGION_KEY_VARS = 5000
ENTRY_KEYS = {"name", "cuts", "regions", "masses"}
OPTIONAL_ENTRY_KEYS = {"parameter", "region_name"}
# words of a C++ cut that are not column names
CPP_WORDS = {
    "true",
    "false",
    "and",
    "or",
    "not",
    "static_cast",
    "double",
    "float",
    "int",
    "bool",
    "unsigned",
    "long",
    "const",
}


class HistBookingError(RuntimeError):
    pass


# ---------------------------------------------------------------- the FLAF plugin interface


def regions(setup):
    """The flavour's regions, then the base regions of its sub-regions not among them."""
    flavour = _flavour(setup)
    out = list(_booked_regions(setup, flavour))
    # expand the sub-regions now, so that a configuration error stops every task at once
    for sreg in sub_regions(setup):
        if sreg["base_region"] not in out:
            out.append(sreg["base_region"])
    if not out:
        raise HistBookingError(
            f"histTuple flavor {setup.histTuple_flavor}: no regions and no sub-regions"
        )
    return out


def booking(setup, keys, variables):
    """The blocks of a HistFromNtupleProducer job: every variable in the keys of the booked regions, and
    one block per sub-region with its variables, in its keys (a base key's filter and the cut).
    """
    flavour = _flavour(setup)
    booked = set(_booked_regions(setup, flavour))
    variables = list(variables)
    blocks = []
    base_keys = [(k, f) for k, f in keys if len(k) >= 2 and k[1] in booked]
    if base_keys:
        blocks.append({"keys": base_keys, "variables": variables})
    n_pairs = 0
    for sreg in sub_regions(setup):
        block_vars = [v for v in variables if _in_mass(setup, v, sreg["mass"])]
        if not block_vars:
            continue
        block_keys, columns = [], []
        for key_tuple, flt in keys:
            if len(key_tuple) < 3 or key_tuple[1] != sreg["base_region"]:
                continue
            cut = sreg["cuts"].get((key_tuple[2], key_tuple[0]))
            if cut is None:
                continue
            expr, cut_columns = cut
            new_key = (key_tuple[0], sreg["name"]) + tuple(key_tuple[2:])
            block_keys.append((new_key, f"({flt}) && ({expr})"))
            columns += [c for c in cut_columns if c not in columns]
        if block_keys:
            blocks.append(
                {"keys": block_keys, "variables": block_vars, "columns": columns}
            )
            n_pairs += len(block_keys) * len(block_vars)
    max_pairs = int(
        setup.global_params.get(
            "sub_regions_max_key_vars", DEFAULT_MAX_SUB_REGION_KEY_VARS
        )
    )
    if 0 < max_pairs < n_pairs:
        raise HistBookingError(
            f"the sub-regions book {n_pairs} (key, variable) pairs in this job, above "
            f"sub_regions_max_key_vars = {max_pairs}: check the flavour's sub_regions and its "
            "variables, or raise the limit on purpose"
        )
    # a variable in no block (e.g. a mass without sub-region when no region is booked) is refused by
    # FLAF, which names it
    return blocks


# ---------------------------------------------------------------- the flavour


def _flavour(setup):
    return setup.global_params["histTuple_flavors"][setup.histTuple_flavor]


def _booked_regions(setup, flavour):
    booked = flavour.get("regions")
    if not isinstance(booked, list) or not all(isinstance(r, str) for r in booked):
        raise HistBookingError(
            f"histTuple flavor {setup.histTuple_flavor}: `regions` must be a list of region names "
            f"([ ] for sub-regions only), got {booked!r}"
        )
    return booked


def _categories(setup):
    gp = setup.global_params
    categories = list(gp.get("categories") or [])
    custom = gp.get("custom_categories")
    if custom:
        categories += list(gp.get(custom) or [])
    return categories


def _channels(setup):
    return list(setup.global_params.get("channelDefinition") or {})


def _hist_entry(setup, var):
    """The histograms.yaml entry of a variable (its patterns are regular expressions)."""
    hists = setup.hists
    matches = [p for p in hists.config_dict if re.fullmatch(p, var)]
    if len(matches) != 1:
        raise HistBookingError(
            f"{var}: {'no' if not matches else 'more than one'} histograms.yaml entry matches it"
        )
    return hists.config_dict[matches[0]]


def _var_masses(setup, var):
    """The masses of a variable's signal_params, or None if it has none (every mass)."""
    cache = setup.__dict__.setdefault("_bbww_var_masses", {})
    if var not in cache:
        params = _hist_entry(setup, var).get("signal_params")
        if params is None:
            cache[var] = None
        else:
            entries = params if isinstance(params, list) else [params]
            masses = {
                int(e["mass"]) for e in entries if isinstance(e, dict) and "mass" in e
            }
            cache[var] = masses or None
    return cache[var]


def _in_mass(setup, var, mass):
    masses = _var_masses(setup, var)
    return masses is None or mass in masses


# ---------------------------------------------------------------- cuts


def axis_edges(x_bins):
    """The bin edges of a histograms.yaml ``x_bins``: a list of edges or ``"n|low:high"``."""
    if isinstance(x_bins, list):
        return [float(e) for e in x_bins]
    n_bins, bin_range = str(x_bins).split("|")
    low, high = (float(v) for v in bin_range.split(":"))
    n_bins = int(n_bins)
    return [low + (high - low) * i / n_bins for i in range(n_bins + 1)]


def edge_index(edges, value, what):
    """The index of ``value`` among ``edges``; an error if it is not one of them."""
    for i, edge in enumerate(edges):
        if abs(edge - float(value)) <= 1e-6 * max(1.0, abs(edge)):
            return i
    raise HistBookingError(
        f"{what}: {value} is not a bin edge of the variable's binning; a bin-form cut must sit on "
        "the stored edges (use the C++ form, on a full-resolution variable, for other edges)"
    )


def bin_cut(bin_column, edges, low, high, what):
    """The cut ``low <= x < high`` on the bin index column (``FindFixBin - 1``: -1 is the underflow,
    ``n_bins`` the overflow). An interval from the first edge takes the underflow too, one to the last
    edge the overflow."""
    i_low = edge_index(edges, low, f"{what} low edge")
    i_high = edge_index(edges, high, f"{what} high edge")
    if i_high <= i_low:
        raise HistBookingError(f"{what}: empty interval [{low}, {high}]")
    n_bins = len(edges) - 1
    cuts = []
    if i_low > 0:
        cuts.append(f"{bin_column} >= {i_low}")
    if i_high < n_bins:
        cuts.append(f"{bin_column} < {i_high}")
    return " && ".join(cuts) if cuts else "true"


def _intervals(value, what):
    """``[low, high]`` or a non-empty list of them, as a list of (low, high)."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        return [(float(value[0]), float(value[1]))]
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, (list, tuple)) for v in value)
    ):
        out = []
        for v in value:
            out.extend(_intervals(v, what))
        return out
    raise HistBookingError(
        f"{what}: expected [low, high] or a list of [low, high], got {value!r}"
    )


def dict_cut(cut, edges_of, binned, what):
    """The bin-index cut of a bin-form part ``{variable: intervals}`` and its columns."""
    parts, columns = [], []
    for var, value in cut.items():
        if var not in binned:
            raise HistBookingError(
                f"{what}: {var} is not a binned variable of the flavour, so it has no bin column "
                "to cut on"
            )
        edges = edges_of(var)
        terms = [
            bin_cut(f"{var}_bin", edges, low, high, f"{what} {var}")
            for low, high in _intervals(value, f"{what} {var}")
        ]
        if "true" in terms:
            continue  # the whole range: no cut, no column needed
        columns.append(f"{var}_bin")
        if len(terms) == 1:
            parts.append(terms[0])
        else:
            parts.append("(" + " || ".join(f"({t})" for t in terms) + ")")
    return (" && ".join(parts) if parts else "true"), columns


def cpp_variables(expr):
    """The column names a C++ cut uses: its identifiers, without numeric literals, function names
    (followed by ``(``; their arguments are columns), namespaced names (``std::abs``) and C++ words.
    A ``<`` after a name is a comparison, not a template."""
    text = re.sub(r'"[^"]*"', " ", expr)
    text = re.sub(
        r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlLuU]*(?![\w.])", " ", text
    )
    names = []
    for m in re.finditer(r"[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*", text):
        word = m.group(0)
        following = text[m.end() :].lstrip()[:1]
        if "::" in word or following == "(" or word in CPP_WORDS:
            continue
        if word not in names:
            names.append(word)
    return names


def cpp_cut(expr, fullres, what, flavour):
    """A C++ part and its columns; every variable it names must be a full-resolution variable."""
    if not expr.strip():
        raise HistBookingError(f"{what}: empty cut")
    names = cpp_variables(expr)
    if not names:
        raise HistBookingError(f"{what}: the cut '{expr}' uses no variable")
    missing = [n for n in names if n not in fullres]
    if missing:
        raise HistBookingError(
            f"{what}: the cut '{expr}' uses {', '.join(missing)}, not in fullResolution_variables "
            f"of flavour {flavour}; add {'it' if len(missing) == 1 else 'them'} there (it is not "
            "added automatically)"
        )
    return expr, names


def _format(template, parameter, value, what, allowed=("region", "name")):
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    unknown = placeholders - {parameter, *allowed}
    if unknown:
        raise HistBookingError(
            f"{what}: unknown placeholder(s) {sorted(unknown)} in '{template}'"
        )
    return template.replace("{" + parameter + "}", str(value))


def cut_expression(cut, ctx, what):
    """The expression and columns of a cut: a bin-form dict, a C++ string, or a list of them (AND)."""
    parts = cut if isinstance(cut, list) else [cut]
    if not parts:
        raise HistBookingError(f"{what}: empty cut")
    exprs, columns = [], []
    for i, part in enumerate(parts):
        part_what = f"{what}" if len(parts) == 1 else f"{what} part {i}"
        if isinstance(part, dict):
            if not part:
                raise HistBookingError(f"{part_what}: empty cut")
            expr, cols = dict_cut(part, ctx["edges"], ctx["binned"], part_what)
        elif isinstance(part, str):
            text = _format(part, ctx["parameter"], ctx["mass"], part_what, allowed=())
            expr, cols = cpp_cut(text, ctx["fullres"], part_what, ctx["flavour"])
        else:
            raise HistBookingError(
                f"{part_what}: a cut part is a {{variable: [low, high]}} dict or a C++ string, "
                f"got {part!r}"
            )
        if expr != "true":
            exprs.append(expr)
        columns += [c for c in cols if c not in columns]
    if not exprs:
        return "true", columns
    if len(exprs) == 1:
        return exprs[0], columns
    return " && ".join(f"({e})" for e in exprs), columns


def _category_cuts(value, ctx, channels, what):
    """``{channel: (expression, columns)}`` of a category's value: a cut, or a channel map."""
    if isinstance(value, dict) and value:
        keys = set(value)
        is_channels = keys <= set(channels)
        is_vars = keys <= ctx["binned"]
        if is_channels and is_vars:
            raise HistBookingError(
                f"{what}: {sorted(keys)} are both channels and variables: ambiguous"
            )
        if is_channels:
            return {
                ch: cut_expression(cut, ctx, f"{what} {ch}")
                for ch, cut in value.items()
            }
        if not is_vars:
            raise HistBookingError(
                f"{what}: {sorted(keys)} are neither all channels ({channels}) nor all binned "
                "variables of the flavour"
            )
    expr = cut_expression(value, ctx, what)
    return {ch: expr for ch in channels}


# ---------------------------------------------------------------- sub-regions


def sub_regions(setup):
    """The sub-regions of the flavour, cached on the setup: a list of
    ``{"name", "sub_region", "base_region", "mass", "cuts": {(category, channel): (expr, columns)}}``.
    """
    cache = setup.__dict__.get("_bbww_sub_regions")
    if cache is not None:
        return cache
    flavour_name = setup.histTuple_flavor
    flavour = _flavour(setup)
    entries = flavour.get("sub_regions") or []
    if not isinstance(entries, list):
        raise HistBookingError(f"flavour {flavour_name}: sub_regions must be a list")
    flavour_vars = flavour.get("variables") or []
    binned = set()
    for v in flavour_vars:
        if isinstance(v, dict) and "vars" in v:
            binned.update(v["vars"])
        else:
            binned.add(v["name"] if isinstance(v, dict) else v)
    fullres = set(flavour.get("fullResolution_variables") or [])
    categories = _categories(setup)
    channels = _channels(setup)
    region_columns = set(_booked_regions(setup, flavour))

    def edges_of(var):
        entry = _hist_entry(setup, var)
        if "x_bins" not in entry:
            raise HistBookingError(f"{var} has no x_bins in histograms.yaml")
        return axis_edges(entry["x_bins"])

    out = []
    names_seen = set(region_columns)
    for i_entry, entry in enumerate(entries):
        what = f"flavour {flavour_name} sub_regions[{i_entry}]"
        if not isinstance(entry, dict):
            raise HistBookingError(f"{what}: an entry is a mapping, got {entry!r}")
        unknown = set(entry) - ENTRY_KEYS - OPTIONAL_ENTRY_KEYS
        missing = ENTRY_KEYS - set(entry)
        if unknown or missing:
            raise HistBookingError(
                f"{what}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
            )
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise HistBookingError(f"{what}: name must be a word, got {name!r}")
        what = f"{what} ({name})"
        parameter = entry.get("parameter", DEFAULT_PARAMETER)
        region_name = entry.get("region_name", "{region}_{name}_m{" + parameter + "}")
        cuts, cuts_path = _load_cuts(entry["cuts"], setup.ana_path, what)
        base_regions = entry["regions"]
        if (
            not isinstance(base_regions, list)
            or not base_regions
            or not all(isinstance(r, str) for r in base_regions)
        ):
            raise HistBookingError(f"{what}: regions must be a non-empty list of names")
        masses = entry["masses"]
        if masses == "all":
            masses = sorted(cuts)
        elif isinstance(masses, list) and masses:
            masses = [int(m) for m in masses]
            for m in masses:
                if m not in cuts:
                    raise HistBookingError(
                        f"{what}: mass {m} has no cut in {cuts_path}"
                    )
        else:
            raise HistBookingError(f"{what}: masses must be 'all' or a non-empty list")
        for mass in masses:
            ctx = {
                "edges": edges_of,
                "binned": binned,
                "fullres": fullres,
                "flavour": flavour_name,
                "parameter": parameter,
                "mass": mass,
            }
            mass_cuts = {}
            for cat, value in cuts[mass].items():
                if cat not in categories:
                    raise HistBookingError(
                        f"{what}: category '{cat}' of {cuts_path} (mass {mass}) is not a configured "
                        f"category ({categories})"
                    )
                for ch, cut in _category_cuts(
                    value, ctx, channels, f"{what} m={mass} {cat}"
                ).items():
                    mass_cuts[(cat, ch)] = cut
            for reg in base_regions:
                reg_name = _format(region_name, parameter, mass, what)
                reg_name = reg_name.replace("{region}", reg).replace("{name}", name)
                if reg_name in names_seen:
                    raise HistBookingError(
                        f"{what}: region name '{reg_name}' is defined twice"
                    )
                names_seen.add(reg_name)
                out.append(
                    {
                        "name": reg_name,
                        "sub_region": name,
                        "base_region": reg,
                        "mass": mass,
                        "cuts": dict(mass_cuts),
                    }
                )
    setup.__dict__["_bbww_sub_regions"] = out
    return out


def _load_cuts(path, ana_path, what):
    """``{mass: {category: value}}`` of a cuts file."""
    if not os.path.isabs(path):
        path = os.path.join(ana_path, path)
    if not os.path.exists(path):
        raise HistBookingError(f"{what}: cuts file {path} not found")
    with open(path) as f:
        content = yaml.safe_load(f) or {}
    raw = content.get("cuts") if isinstance(content, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise HistBookingError(f"{what}: {path} has no 'cuts' mapping")
    cuts = {}
    for mass, per_cat in raw.items():
        if not isinstance(per_cat, dict) or not per_cat:
            raise HistBookingError(f"{what}: {path} mass {mass} has no category cuts")
        cuts[int(mass)] = {str(cat): value for cat, value in per_cat.items()}
    return cuts, path
