"""HH->bbWW histogram booking: the FLAF booking plugin (FLAF Common/HistBooking.py) of this analysis.

global.yaml names it (``hist_booking_module: Analysis.hist_booking``). It gives FLAF

- ``regions(setup)``: the regions of the active histTuple flavour (``regions``), plus the base regions
  of its sub-regions; FLAF puts them in the list ``custom_regions`` names, so the selection keys are
  built, and the histTuples keep events, for them;
- ``booking(setup, keys, variables)``: what a HistFromNtupleProducer job fills: every variable in the
  keys of the flavour's ``regions``, and each sub-region's variables in its own keys.

A flavour::

    histTuple_flavors:
      Signal_Extraction_DL_window:
        variables: [ ... ]
        regions: [ SR, TT_CR, DY_CR ]            # booked; [ ] books sub-regions only
        sub_regions:
          - name: HMESub
            cuts: config/sub_regions/hme_DL_v2610.yaml
            regions: [ SR, TT_CR, DY_CR ]         # the base regions it restricts (booked or not)
            points: all                           # or a list of model points, e.g. [ { mass: 300 } ]

**Model points.** A sub-region family is indexed by the model points of its cuts file: any parameters
(the resonance mass, its spin, couplings, ...). The file declares them and the region names::

    parameters: [ mass, spin ]                  # the nesting order of `cuts`
    region_name: "{region}_{name}_m{mass}_s{spin}"
    cuts:
      300:                                      # mass
        0:                                      # spin
          res2b: "DeepHME_mass >= 250 && DeepHME_mass < 400"
          boosted: { DeepHME_mass: [ 260, 470 ] }

Under the last parameter come the categories, each a cut or a channel map ``{channel: cut}``. In region
names a value is written as is, a non-integer number with ``p`` for the point and ``m`` for a minus
(``kl: -2.45`` -> ``klm2p45``). ``{<parameter>}`` in a C++ cut is replaced by the value as written.

**Variables.** The flavour's variables are filled in the sub-regions of the points they belong to: a
variable whose histograms.yaml entry has ``signal_params`` (an entry or a list of entries) belongs to a
point if one entry agrees with the point on every parameter both have; a variable without
``signal_params`` belongs to every point. So ``{mass: 300}`` is in both spins of mass 300,
``{mass: 300, spin: 0}`` only in spin 0, and ``DeepHME_mass`` everywhere.

**Cuts.** A cut is one part or a list of parts, all required:

- bin form ``{variable: [low, high] or [[low, high], ...]}``: each variable a binned variable of the
  flavour, every edge a bin edge of its binning, the cut on the stored bin index (``<var>_bin``); the
  intervals of a variable are OR'ed, the variables AND'ed; an interval from the first (to the last) edge
  takes the underflow (overflow) too. No histTuple rerun is needed to change it.
- C++ form, a string: used as written. Every variable it names must be in the flavour's
  ``fullResolution_variables``; nothing is added automatically, so a variable that is not there is an
  error. Function names (``abs(``), namespaced names (``std::abs``), numeric literals and ``true``/
  ``false`` are not variables; a function's arguments are.

A dict whose keys are all channels is a channel map, one whose keys are all binned variables a bin-form
cut; anything else is an error. A channel missing from a map, or a category missing from a point, has no
sub-region key.

Every ambiguity is an error, and a job refuses more sub-region (key, variable) pairs than
``sub_regions_max_key_vars`` (default 5000; 0 disables the check).
"""

import os
import re

import yaml

DEFAULT_MAX_SUB_REGION_KEY_VARS = 5000
ENTRY_KEYS = {"name", "cuts", "regions", "points"}
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
    out = list(_booked_regions(setup, _flavour(setup)))
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
    one block per sub-region with the variables of its point, in its keys (a base key's filter and the
    cut)."""
    booked = set(_booked_regions(setup, _flavour(setup)))
    variables = list(variables)
    blocks = []
    base_keys = [(k, f) for k, f in keys if len(k) >= 2 and k[1] in booked]
    if base_keys:
        blocks.append({"keys": base_keys, "variables": variables})
    n_pairs = 0
    for sreg in sub_regions(setup):
        block_vars = [v for v in variables if _in_point(setup, v, sreg["point"])]
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


def _same(a, b):
    """Parameter values as equal: numbers by value (300 == 300.0), anything else as text."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return str(a) == str(b)


def _var_entries(setup, var):
    """The signal_params entries of a variable, or None if it has none (every point)."""
    cache = setup.__dict__.setdefault("_bbww_var_entries", {})
    if var not in cache:
        params = _hist_entry(setup, var).get("signal_params")
        if params is None:
            cache[var] = None
        else:
            entries = params if isinstance(params, list) else [params]
            if not entries or not all(isinstance(e, dict) for e in entries):
                raise HistBookingError(
                    f"{var}: signal_params must be a mapping or a list of mappings, got {params!r}"
                )
            cache[var] = entries
    return cache[var]


def _in_point(setup, var, point):
    """Whether a variable belongs to a model point: one of its signal_params entries agrees with the
    point on every parameter both have (or it has none)."""
    entries = _var_entries(setup, var)
    if entries is None:
        return True
    return any(
        all(_same(point[k], v) for k, v in e.items() if k in point) for e in entries
    )


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


def _substitute(template, values, what, allowed=()):
    """``template`` with ``{parameter}`` replaced by ``values[parameter]``; other placeholders than
    those and ``allowed`` are an error."""
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    unknown = placeholders - set(values) - set(allowed)
    if unknown:
        raise HistBookingError(
            f"{what}: unknown placeholder(s) {sorted(unknown)} in '{template}'"
        )
    for name, value in values.items():
        template = template.replace("{" + name + "}", str(value))
    return template


def name_value(value):
    """A parameter value for a region name: integers as is, other numbers with p for the point and m for
    a minus (-2.45 -> m2p45), text as is."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace("-", "m").replace(".", "p")


def cut_expression(cut, ctx, what):
    """The expression and columns of a cut: a bin-form dict, a C++ string, or a list of them (AND)."""
    parts = cut if isinstance(cut, list) else [cut]
    if not parts:
        raise HistBookingError(f"{what}: empty cut")
    exprs, columns = [], []
    for i, part in enumerate(parts):
        part_what = what if len(parts) == 1 else f"{what} part {i}"
        if isinstance(part, dict):
            if not part:
                raise HistBookingError(f"{part_what}: empty cut")
            expr, cols = dict_cut(part, ctx["edges"], ctx["binned"], part_what)
        elif isinstance(part, str):
            text = _substitute(part, ctx["point"], part_what)
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


# ---------------------------------------------------------------- cuts files and sub-regions


def load_cuts(path, ana_path, what):
    """A cuts file: ``(parameters, region_name, {point tuple: {category: value}}, path)``, the point
    tuple the parameter values in the order of ``parameters``."""
    if not os.path.isabs(path):
        path = os.path.join(ana_path, path)
    if not os.path.exists(path):
        raise HistBookingError(f"{what}: cuts file {path} not found")
    with open(path) as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise HistBookingError(f"{what}: {path} is not a mapping")
    parameters = content.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or not all(isinstance(p, str) and re.fullmatch(r"\w+", p) for p in parameters)
        or len(set(parameters)) != len(parameters)
    ):
        raise HistBookingError(
            f"{what}: {path} needs `parameters`, a list of distinct parameter names, got "
            f"{parameters!r}"
        )
    region_name = content.get("region_name")
    if not isinstance(region_name, str) or "{region}" not in region_name:
        raise HistBookingError(
            f"{what}: {path} needs `region_name`, a pattern with {{region}}, {{name}} and the "
            f"parameters, got {region_name!r}"
        )
    missing = [p for p in parameters if "{" + p + "}" not in region_name]
    if missing:
        raise HistBookingError(
            f"{what}: {path} region_name '{region_name}' lacks {missing}: regions of different "
            "points would share a name"
        )
    raw = content.get("cuts")
    if not isinstance(raw, dict) or not raw:
        raise HistBookingError(f"{what}: {path} has no 'cuts' mapping")
    points = {}

    def walk(node, values):
        depth = len(values)
        if depth == len(parameters):
            if not isinstance(node, dict) or not node:
                raise HistBookingError(
                    f"{what}: {path} point {dict(zip(parameters, values))} has no category cuts"
                )
            points[tuple(values)] = {str(cat): v for cat, v in node.items()}
            return
        if not isinstance(node, dict) or not node:
            raise HistBookingError(
                f"{what}: {path}: expected the values of `{parameters[depth]}` under "
                f"{dict(zip(parameters, values))}, got {node!r}"
            )
        for value, sub in node.items():
            walk(sub, values + [value])

    walk(raw, [])
    return parameters, region_name, points, path


def sub_regions(setup):
    """The sub-regions of the flavour, cached on the setup: a list of
    ``{"name", "sub_region", "base_region", "point": {parameter: value},
    "cuts": {(category, channel): (expression, columns)}}``."""
    cache = setup.__dict__.get("_bbww_sub_regions")
    if cache is not None:
        return cache
    flavour_name = setup.histTuple_flavor
    flavour = _flavour(setup)
    entries = flavour.get("sub_regions") or []
    if not isinstance(entries, list):
        raise HistBookingError(f"flavour {flavour_name}: sub_regions must be a list")
    binned = set()
    for v in flavour.get("variables") or []:
        if isinstance(v, dict) and "vars" in v:
            binned.update(v["vars"])
        else:
            binned.add(v["name"] if isinstance(v, dict) else v)
    fullres = set(flavour.get("fullResolution_variables") or [])
    categories = _categories(setup)
    channels = _channels(setup)

    def edges_of(var):
        entry = _hist_entry(setup, var)
        if "x_bins" not in entry:
            raise HistBookingError(f"{var} has no x_bins in histograms.yaml")
        return axis_edges(entry["x_bins"])

    out = []
    names_seen = set(_booked_regions(setup, flavour))
    for i_entry, entry in enumerate(entries):
        what = f"flavour {flavour_name} sub_regions[{i_entry}]"
        if not isinstance(entry, dict):
            raise HistBookingError(f"{what}: an entry is a mapping, got {entry!r}")
        unknown, missing = set(entry) - ENTRY_KEYS, ENTRY_KEYS - set(entry)
        if unknown or missing:
            raise HistBookingError(
                f"{what}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
            )
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise HistBookingError(f"{what}: name must be a word, got {name!r}")
        what = f"{what} ({name})"
        parameters, region_name, points, cuts_path = load_cuts(
            entry["cuts"], setup.ana_path, what
        )
        base_regions = entry["regions"]
        if (
            not isinstance(base_regions, list)
            or not base_regions
            or not all(isinstance(r, str) for r in base_regions)
        ):
            raise HistBookingError(f"{what}: regions must be a non-empty list of names")
        selected = entry["points"]
        if selected == "all":
            selected = list(points)
        elif isinstance(selected, list) and selected:
            chosen = []
            for p in selected:
                if not isinstance(p, dict) or set(p) != set(parameters):
                    raise HistBookingError(
                        f"{what}: a point is a mapping of {parameters}, got {p!r}"
                    )
                match = [
                    t
                    for t in points
                    if all(_same(p[k], v) for k, v in zip(parameters, t))
                ]
                if not match:
                    raise HistBookingError(
                        f"{what}: point {p} has no cut in {cuts_path}"
                    )
                chosen.append(match[0])
            selected = chosen
        else:
            raise HistBookingError(f"{what}: points must be 'all' or a non-empty list")
        for values in selected:
            point = dict(zip(parameters, values))
            ctx = {
                "edges": edges_of,
                "binned": binned,
                "fullres": fullres,
                "flavour": flavour_name,
                "point": point,
            }
            point_cuts = {}
            for cat, value in points[values].items():
                if cat not in categories:
                    raise HistBookingError(
                        f"{what}: category '{cat}' of {cuts_path} (point {point}) is not a "
                        f"configured category ({categories})"
                    )
                for ch, cut in _category_cuts(
                    value, ctx, channels, f"{what} {point} {cat}"
                ).items():
                    point_cuts[(cat, ch)] = cut
            name_values = {k: name_value(v) for k, v in point.items()}
            for reg in base_regions:
                reg_name = _substitute(
                    region_name,
                    dict(name_values, region=reg, name=name),
                    what,
                )
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
                        "point": dict(point),
                        "cuts": dict(point_cuts),
                    }
                )
    setup.__dict__["_bbww_sub_regions"] = out
    return out
