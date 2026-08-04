#!/usr/bin/env python3
"""Fit ExpPolyLogY2D (N × pdf) on an adaptive cell grid (chi2_norm).

Input: grid JSON from ``grid_merge.py``.  ROOT Minuit2 on a Python χ² with
Gauss–Legendre cell integrals of the continuous density:

  s(x,y) = exp(poly on xn, log-mapped yn)
  pdf    = s / ∬_window s
  μ_cell = N · ∬_cell pdf

**Empty meta-cluster (default):** all zero-yield cells → one χ² term
(Garwood σ).  Disable with ``--no-empty-meta``.

**Accepted uncertainty thr** (default max 0.1)::

  σ_eff² = σ_stat² + (thr · |content|)²

**Automatic selection (default):** if ``--degree`` is omitted, scan
``degree = 1, 2, …, --max-degree``.  For each degree, scan
``thr = 0, thr_step, …, thr`` (1% steps by default) and stop at the first
configuration with p≥0.05 and a usable covariance.  Returns fitted
parameters plus the chosen ``degree`` and ``thr_min``.

If ``--degree N`` is set, thr is fixed to ``--thr`` (use ``--auto-thr`` to
still scan thr at that degree).

Example::

  # Auto degree + thr ≤ 10%
  python3 -u fit_one.py --grid grid_m500.json --output m500.json

  # Fixed degree, thr = 10%
  python3 -u fit_one.py --grid grid_m500.json --output m500.json --degree 4 --thr 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import ROOT

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from core import ExpPolyLogY2D  # noqa: E402

P_MIN = 0.05

# ROOT TH1 kPoisson alpha (Garwood)
_POISSON_ALPHA = 1.0 - 0.682689492137085897
_SIGMA_N0 = None


class CellObs(object):
    """One χ² observation: a single rectangle or a multi-rect meta-cluster.

    For a normal cell, ``rects`` is None and (xlo,xhi,ylo,yhi) define the
    integral domain.  For the empty meta-cluster, ``rects`` is a list of
    (xlo,xhi,ylo,yhi) (not necessarily connected); content is the summed
    yield (0) and μ is the sum of continuous integrals over every sub-rect.
    """

    __slots__ = (
        "id",
        "xlo",
        "xhi",
        "ylo",
        "yhi",
        "content",
        "error",
        "error_stat",
        "x",
        "y",
        "rects",
        "n_subrects",
        "is_empty_meta",
    )

    def __init__(
        self,
        cell_id,
        xlo,
        xhi,
        ylo,
        yhi,
        content,
        error,
        rects=None,
        is_empty_meta=False,
    ):
        self.id = int(cell_id)
        self.content = float(content)
        # error_stat = pure statistical (hist / Garwood); error = χ² sigma
        # after optional acceptable_unc_thr (set via apply_acceptable_thr).
        self.error_stat = float(error)
        self.error = float(error)
        self.is_empty_meta = bool(is_empty_meta)
        if rects is not None:
            self.rects = [
                (float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rects
            ]
            self.n_subrects = len(self.rects)
            # Bbox of the union (diagnostic only; integral uses rects)
            self.xlo = min(r[0] for r in self.rects)
            self.xhi = max(r[1] for r in self.rects)
            self.ylo = min(r[2] for r in self.rects)
            self.yhi = max(r[3] for r in self.rects)
        else:
            self.rects = None
            self.n_subrects = 1
            self.xlo = float(xlo)
            self.xhi = float(xhi)
            self.ylo = float(ylo)
            self.yhi = float(yhi)
        self.x = 0.5 * (self.xlo + self.xhi)
        self.y = 0.5 * (self.ylo + self.yhi)


def _garwood_poisson_sigma0():
    return 0.5 * float(ROOT.TMath.ChisquareQuantile(1.0 - 0.5 * _POISSON_ALPHA, 2))


def _sigma0_from_th1():
    h = ROOT.TH1D("_sig0_ref", "", 1, 0.0, 1.0)
    h.SetDirectory(0)
    h.SetBinContent(1, 0.0)
    h.SetBinErrorOption(ROOT.TH1.kPoisson)
    return float(h.GetBinErrorUp(1))


def _cell_sigma(content, error, cell_id=None):
    """Statistical χ² bin error: hist error if n>0; Garwood Up for n=0."""
    global _SIGMA_N0
    c = float(content)
    e = float(error)
    if c < 0:
        raise ValueError(
            "cell id=%s has negative content %g (grid must be non-negative)"
            % (cell_id, c)
        )
    if c == 0.0:
        if _SIGMA_N0 is None:
            try:
                _SIGMA_N0 = _sigma0_from_th1()
            except Exception:
                _SIGMA_N0 = _garwood_poisson_sigma0()
        return _SIGMA_N0
    if e > 0.0 and math.isfinite(e):
        return e
    raise ValueError("cell id=%s n=%g has non-positive error %g" % (cell_id, c, e))


def effective_sigma(content, error_stat, thr):
    """σ_eff = sqrt(σ_stat² + (thr · |content|)²).

    ``thr`` is the overall accepted relative uncertainty (e.g. 0.1 = 10%).
    ``thr=0`` recovers pure statistical errors (legacy).  Empty bins
    (content=0) keep σ_stat only (Garwood).
    """
    e = float(error_stat)
    t = float(thr)
    if t <= 0.0 or not math.isfinite(t):
        return e
    c = abs(float(content))
    return math.sqrt(e * e + (t * c) * (t * c))


def apply_acceptable_thr(obs, thr):
    """Set each observation's χ² error from error_stat + thr (in place)."""
    t = float(thr)
    for b in obs:
        b.error = effective_sigma(b.content, b.error_stat, t)
    return obs


def load_grid_json(path):
    with open(path) as fh:
        return json.load(fh)


def fit_range_from_positive_cells(cells):
    """Bounding box of cells with content > 0 (outer edges)."""
    pos = [c for c in cells if float(c.get("content", 0)) > 0.0]
    if not pos:
        raise ValueError("no cells with content > 0 — cannot define fit range")
    return (
        min(float(c["xmin"]) for c in pos),
        max(float(c["xmax"]) for c in pos),
        min(float(c["ymin"]) for c in pos),
        max(float(c["ymax"]) for c in pos),
    )


def _coalesce_index_rects(index_rects):
    """Greedy merge of axis-aligned integer index boxes [ix0,ix1)×[iy0,iy1).

    1) Horizontal runs on each row-band of equal (iy0,iy1).
    2) Vertical merge of identical-x strips with adjacent y.
    Returns a (usually much smaller) list of index rects covering the same area.
    """
    if not index_rects:
        return []
    # Expand multi-row/col boxes to unit strips for clean RLE, then re-merge.
    # For general rects: group by (iy0, iy1) and merge along x.
    by_y = {}
    for ix0, ix1, iy0, iy1 in index_rects:
        key = (int(iy0), int(iy1))
        by_y.setdefault(key, []).append((int(ix0), int(ix1)))
    hmerged = []
    for (iy0, iy1), xs in by_y.items():
        xs.sort()
        a, b = xs[0]
        for c, d in xs[1:]:
            if c <= b:  # overlap or touch
                b = max(b, d)
            else:
                hmerged.append((a, b, iy0, iy1))
                a, b = c, d
        hmerged.append((a, b, iy0, iy1))
    # Vertical merge: group by (ix0, ix1)
    by_x = {}
    for ix0, ix1, iy0, iy1 in hmerged:
        key = (ix0, ix1)
        by_x.setdefault(key, []).append((iy0, iy1))
    out = []
    for (ix0, ix1), ys in by_x.items():
        ys.sort()
        a, b = ys[0]
        for c, d in ys[1:]:
            if c <= b:
                b = max(b, d)
            else:
                out.append((ix0, ix1, a, b))
                a, b = c, d
        out.append((ix0, ix1, a, b))
    return out


def _empty_rects_from_cells(empty_cells, x_edges, y_edges):
    """Build physical empty rects, coalescing via fine-bin indices when possible.

    Falls back to one rect per cell if indices / edges are unavailable.
    """
    if not empty_cells:
        return []
    have_idx = all(
        ("ix0" in c and "ix1" in c and "iy0" in c and "iy1" in c) for c in empty_cells
    )
    if not have_idx or x_edges is None or y_edges is None:
        return [
            (
                float(c["xmin"]),
                float(c["xmax"]),
                float(c["ymin"]),
                float(c["ymax"]),
            )
            for c in empty_cells
        ]
    idx = [
        (int(c["ix0"]), int(c["ix1"]), int(c["iy0"]), int(c["iy1"]))
        for c in empty_cells
    ]
    merged = _coalesce_index_rects(idx)
    xe = list(x_edges)
    ye = list(y_edges)
    rects = []
    for ix0, ix1, iy0, iy1 in merged:
        rects.append((float(xe[ix0]), float(xe[ix1]), float(ye[iy0]), float(ye[iy1])))
    return rects


def cells_to_observations(grid, dnn_min, dnn_max, hme_min, hme_max, empty_meta=True):
    """Build CellObs list: centres inside the fit window, content ≥ 0.

    If ``empty_meta`` is True (default), all zero-yield cells are merged into
    **one** meta-cluster observation (disconnected rectangles allowed):
    content=0, Garwood σ once, μ = Σ_empty ∬_rect pdf.  Empty fine-bin
    rectangles are **coalesced** via index RLE so GL cost scales with the
    number of maximal empty blocks, not the empty-cell count.
    """
    pos = []
    empty_cells = []
    for c in grid["cells"]:
        xlo, xhi = float(c["xmin"]), float(c["xmax"])
        ylo, yhi = float(c["ymin"]), float(c["ymax"])
        cx = 0.5 * (xlo + xhi)
        cy = 0.5 * (ylo + yhi)
        if cx < dnn_min or cx > dnn_max or cy < hme_min or cy > hme_max:
            continue
        cont = float(c["content"])
        if cont < 0:
            raise ValueError(
                "grid cell id=%s has negative content %g" % (c.get("id"), cont)
            )
        if cont == 0.0:
            empty_cells.append(c)
            continue
        err = _cell_sigma(cont, float(c.get("error") or 0.0), c.get("id"))
        pos.append(CellObs(c.get("id", len(pos)), xlo, xhi, ylo, yhi, cont, err))

    out = list(pos)
    n_empty_cells = len(empty_cells)
    x_edges = grid.get("x_edges_fine") or grid.get("x_edges")
    y_edges = grid.get("y_edges_fine") or grid.get("y_edges")
    empty_rects = _empty_rects_from_cells(empty_cells, x_edges, y_edges)
    n_empty_rects = len(empty_rects)
    if empty_rects and empty_meta:
        # Single χ² term; coalesced rects for cheap multi-rect GL.
        sig0 = _cell_sigma(0.0, 0.0, cell_id="empty_meta")
        meta = CellObs(
            cell_id=-1,
            xlo=empty_rects[0][0],
            xhi=empty_rects[0][1],
            ylo=empty_rects[0][2],
            yhi=empty_rects[0][3],
            content=0.0,
            error=sig0,
            rects=empty_rects,
            is_empty_meta=True,
        )
        out.append(meta)
    elif empty_rects:
        for i, (xlo, xhi, ylo, yhi) in enumerate(empty_rects):
            cid = int(empty_cells[i].get("id", i)) if i < len(empty_cells) else i
            err = _cell_sigma(0.0, 0.0, cid)
            out.append(CellObs(cid, xlo, xhi, ylo, yhi, 0.0, err))

    return out, {
        "n_pos_obs": len(pos),
        "n_empty_cells": n_empty_cells,
        "n_empty_rects": n_empty_rects,
        "n_empty_obs": (1 if (empty_meta and n_empty_cells) else n_empty_cells),
        "empty_meta": bool(empty_meta and n_empty_cells > 0),
        "n_obs": len(out),
    }


def _reseed_N(model, obs, inits, gl_pack=None):
    """Closed-form N* for fixed shape (χ² with free yield)."""
    unit = [1.0] + [float(v) for v in inits[1:]]
    if gl_pack is not None:
        mu = model.expected_cells_batch(unit, gl_pack)
        y = np.array([b.content for b in obs], dtype=float)
        e = np.array([b.error for b in obs], dtype=float)
        w = 1.0 / (e * e)
        mask = (mu > 0) & np.isfinite(mu)
        if not np.any(mask):
            return inits
        num = float(np.sum(y[mask] * mu[mask] * w[mask]))
        den = float(np.sum(mu[mask] * mu[mask] * w[mask]))
    else:
        num = den = 0.0
        for b in obs:
            try:
                mu1 = model.expected_bin(unit, b.xlo, b.xhi, b.ylo, b.yhi)
            except Exception:
                return inits
            if not (mu1 == mu1 and mu1 > 0 and abs(mu1) < 1e300):
                return inits
            w = 1.0 / (b.error * b.error)
            num += b.content * mu1 * w
            den += mu1 * mu1 * w
    if den > 0 and math.isfinite(num / den) and (num / den) > 0:
        inits = list(inits)
        inits[0] = num / den
    return inits


def _read_cov(mini, npar):
    try:
        mat = [[float(mini.CovMatrix(i, j)) for j in range(npar)] for i in range(npar)]
        diags = [mat[i][i] for i in range(npar)]
        if all(math.isfinite(d) and d >= 0 for d in diags) and max(diags) > 0.0:
            return mat
    except Exception:
        pass
    return None


def fit_chi2_norm_cells(
    model,
    obs,
    max_calls=50000,
    strategy=2,
    extract_covariance=True,
):
    """Minuit2 χ²: data vs continuous N×pdf integrated over each cell."""
    npar = model.npar
    key = model.key
    if len(obs) <= npar:
        return {
            "key": key,
            "converged": False,
            "error": "ndata <= npar",
            "ndata": len(obs),
            "npar": npar,
            "chi2": float("inf"),
            "ndf": 0,
            "chi2ndf": float("inf"),
            "p_value": 0.0,
            "parameters": [],
            "errors": [],
            "covariance": None,
            "hesse_ok": False,
        }

    # Precompute GL geometry once (vectorized χ²)
    gl_pack = model.prepare_cell_gl(obs)
    y_data = np.array([b.content for b in obs], dtype=float)
    e_data = np.array([b.error for b in obs], dtype=float)
    inv_e2 = 1.0 / (e_data * e_data)

    inits = [float(v) for v in model.initial_params]
    inits[0] = max(float(y_data.sum()), 1.0)
    inits = _reseed_N(model, obs, inits, gl_pack=gl_pack)

    def chi2(par):
        try:
            mu = model.expected_cells_batch(par, gl_pack)
        except Exception:
            return 1e30
        if not np.all(np.isfinite(mu)) or np.any(mu < 0) or np.any(mu > 1e29):
            return 1e30
        d = y_data - mu
        return float(np.sum(d * d * inv_e2))

    mini = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
    mini.SetMaxFunctionCalls(int(max_calls))
    mini.SetMaxIterations(int(max_calls))
    mini.SetTolerance(1e-3)
    mini.SetStrategy(int(strategy))
    mini.SetPrintLevel(0)
    mini.SetErrorDef(1.0)
    functor = ROOT.Math.Functor(chi2, npar)
    mini.SetFunction(functor)
    for i, (name, v0) in enumerate(zip(model.param_names, inits)):
        step = max(abs(v0) * 0.05, 0.05 if i == 0 else 0.01)
        mini.SetVariable(i, name, float(v0), step)
    try:
        mini.SetVariableLimits(0, 0.0, max(inits[0] * 50.0, 1e6))
    except Exception:
        pass

    t0 = ROOT.TStopwatch()
    t0.Start()
    migrad_ok = bool(mini.Minimize())
    if not migrad_ok or mini.Status() not in (0, 1):
        mini.SetStrategy(2)
        migrad_ok = bool(mini.Minimize()) or migrad_ok

    cov = None
    hesse_ok = False
    if extract_covariance:
        for strat in (max(int(strategy), 2), 2, 1):
            try:
                mini.SetStrategy(int(strat))
                xs0 = mini.X()
                for i in range(npar):
                    v = float(xs0[i])
                    mini.SetVariableStepSize(i, max(abs(v) * 0.05, 0.01))
                mini.Hesse()
                cov = _read_cov(mini, npar)
                if cov is not None:
                    hesse_ok = True
                    break
            except Exception:
                cov = None
                hesse_ok = False

    wall = t0.RealTime()
    xs = mini.X()
    params = [float(xs[i]) for i in range(npar)]
    errors = (
        [float(mini.Errors()[i]) for i in range(npar)] if hesse_ok else [0.0] * npar
    )

    chi2val = float(chi2(params))
    status = mini.Status()
    ndata = len(obs)
    ndf = ndata - npar
    chi2ndf = chi2val / ndf if ndf > 0 else float("inf")
    pval = float(ROOT.TMath.Prob(chi2val, ndf)) if ndf > 0 and chi2val < 1e20 else 0.0
    soft_ok = (
        status in (0, 1)
        or (status in (2, 3, 4, 5) and chi2ndf < 5.0)
        or (math.isfinite(chi2ndf) and chi2ndf < 1.5 and math.isfinite(chi2val))
    )
    # Predicted cell yields at the minimum (same GL as the fit objective)
    try:
        mu_hat = model.expected_cells_batch(params, gl_pack)
        n_expected_cells = float(np.sum(mu_hat))
    except Exception:
        n_expected_cells = None
    _keepalive = (functor, mini, gl_pack)

    return {
        "key": key,
        "label": model.label,
        "npar": npar,
        "param_names": list(model.param_names),
        "parameters": params,
        "errors": errors,
        "covariance": cov,
        "hesse_ok": hesse_ok,
        "chi2": chi2val,
        "ndf": ndf,
        "chi2ndf": chi2ndf,
        "p_value": pval,
        "converged": bool(soft_ok and math.isfinite(chi2val)),
        "minuit_status": status,
        "migrad_ok": migrad_ok,
        "edm": float(mini.Edm()),
        "ncalls": int(mini.NCalls()),
        "wall_time": float(wall),
        "ndata": ndata,
        "n_expected_cells": n_expected_cells,
        "fit_mode": "chi2_norm",
        "_keepalive": _keepalive,
    }


def json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, (np.floating, np.integer)):
            return json_safe(float(obj))
    except ImportError:
        pass
    return obj


def fit_quality_ok(result):
    if not result.get("converged"):
        return False, "fit not converged (minuit/status)"
    if not result.get("covariance"):
        return False, "covariance not defined (Hesse failed or null matrix)"
    p = result.get("p_value")
    if p is None or not math.isfinite(float(p)):
        return False, "p-value not defined"
    if float(p) < P_MIN:
        return False, "p-value %.4g < %.2f" % (float(p), P_MIN)
    return True, "ok"


def _thr_grid(thr_max, thr_step=0.01, thr_start=0.0):
    """Ascending thr values in thr_step steps, always including thr_max."""
    thr_max = max(float(thr_max), 0.0)
    thr_step = max(float(thr_step), 1e-6)
    t0 = max(0.0, float(thr_start))
    n_max = int(round(thr_max / thr_step))
    n0 = int(round(t0 / thr_step))
    grid = [min(n * thr_step, thr_max) for n in range(n0, n_max + 1)]
    if not grid or grid[-1] < thr_max - 0.5 * thr_step:
        grid.append(thr_max)
    return sorted(set(round(x / thr_step) * thr_step for x in grid))


def find_min_thr(
    model,
    obs,
    thr_max=0.1,
    thr_step=0.01,
    thr_start=0.0,
    max_calls=50000,
    strategy=2,
    extract_covariance=True,
):
    """Smallest thr on the 1% grid with p≥P_MIN + quality.

    Returns (thr_min, fit_dict).  If none pass, thr_max fit (quality FAIL).
    """
    thr_grid = _thr_grid(thr_max, thr_step=thr_step, thr_start=thr_start)
    last_fit = None
    for i, thr in enumerate(thr_grid):
        apply_acceptable_thr(obs, thr)
        fit = fit_chi2_norm_cells(
            model,
            obs,
            max_calls=max_calls,
            strategy=strategy,
            extract_covariance=extract_covariance,
        )
        fit.pop("_keepalive", None)
        fit["acceptable_unc_thr"] = float(thr)
        ok, reason = fit_quality_ok(fit)
        fit["quality_ok"] = ok
        fit["quality_reason"] = reason
        last_fit = fit
        if ok:
            fit["thr_min"] = float(thr)
            fit["thr_search"] = "step_%.4g_n=%d" % (thr_step, i + 1)
            apply_acceptable_thr(obs, thr)
            return float(thr), fit
    last_fit["thr_min"] = float(thr_grid[-1])
    last_fit["thr_search"] = "step_%.4g_fail_n=%d" % (thr_step, len(thr_grid))
    apply_acceptable_thr(obs, thr_grid[-1])
    return float(thr_grid[-1]), last_fit


def select_degree_and_thr(
    xmin,
    xmax,
    ymin,
    ymax,
    obs,
    *,
    degree=None,
    max_degree=8,
    thr=0.1,
    thr_step=0.01,
    auto_thr=False,
    n_quad=4,
    max_calls=50000,
    strategy=2,
):
    """Pick ExpPolyLogY2D degree and thr.

    * ``degree is None`` (auto): for d=1..max_degree, scan thr ∈ [0, thr]
      in ``thr_step`` steps; return first PASS (minimal d, then minimal thr).
    * ``degree`` fixed, ``auto_thr=False``: fit once at that thr.
    * ``degree`` fixed, ``auto_thr=True``: thr scan at that degree only.
    """
    integral = max(sum(b.content for b in obs), 1.0)
    if degree is not None:
        degrees = [int(degree)]
        auto_degree = False
    else:
        degrees = list(range(1, int(max_degree) + 1))
        auto_degree = True
        auto_thr = True  # degree auto always scans thr up to thr max

    last = None
    for d in degrees:
        model = ExpPolyLogY2D(xmin, xmax, ymin, ymax, degree=d, n_quad=n_quad)
        model.initial_params[0] = integral
        print(
            "[fit_one] try degree=%d  npar=%d  thr_mode=%s thr_max=%g"
            % (
                d,
                model.npar,
                "scan" if auto_thr else "fixed",
                thr,
            ),
            flush=True,
        )
        if auto_thr:
            thr_min, fit = find_min_thr(
                model,
                obs,
                thr_max=thr,
                thr_step=thr_step,
                thr_start=0.0,
                max_calls=max_calls,
                strategy=strategy,
            )
        else:
            apply_acceptable_thr(obs, thr)
            fit = fit_chi2_norm_cells(
                model,
                obs,
                max_calls=max_calls,
                strategy=strategy,
                extract_covariance=True,
            )
            fit.pop("_keepalive", None)
            thr_min = float(thr)
            fit["acceptable_unc_thr"] = thr_min
            fit["thr_min"] = thr_min
            fit["thr_search"] = "fixed"
            ok, reason = fit_quality_ok(fit)
            fit["quality_ok"] = ok
            fit["quality_reason"] = reason

        fit["degree"] = d
        fit["auto_degree"] = auto_degree
        last = (model, fit, thr_min)
        print(
            "[fit_one]   d=%d thr=%.4g  chi2/ndf=%s  p=%s  %s"
            % (
                d,
                thr_min,
                fit.get("chi2ndf"),
                fit.get("p_value"),
                "PASS" if fit.get("quality_ok") else "FAIL",
            ),
            flush=True,
        )
        if fit.get("quality_ok"):
            return model, fit, thr_min

    model, fit, thr_min = last
    return model, fit, thr_min


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--grid", required=True, help="Grid JSON from grid_merge.py")
    p.add_argument("--output", required=True, help="Output fit JSON path")
    p.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Poly degree (omit → auto-scan degree 1..--max-degree)",
    )
    p.add_argument(
        "--max-degree",
        type=int,
        default=8,
        help="Upper degree for auto scan (default 8)",
    )
    p.add_argument(
        "--thr",
        "--acceptable-unc-thr",
        dest="thr",
        type=float,
        default=0.1,
        help=(
            "Accepted relative unc. (default 0.1).  With auto degree: max thr "
            "scanned in 1%% steps from 0.  With fixed --degree: fixed thr "
            "unless --auto-thr."
        ),
    )
    p.add_argument(
        "--auto-thr",
        action="store_true",
        help="With fixed --degree, still scan thr from 0 to --thr in 1%% steps",
    )
    p.add_argument(
        "--thr-step",
        type=float,
        default=0.01,
        help="thr grid step (default 0.01 = 1%%)",
    )
    p.add_argument(
        "--n-quad",
        type=int,
        default=4,
        help="Gauss–Legendre order per axis (default 4)",
    )
    p.add_argument("--dnn-min", type=float, default=None)
    p.add_argument("--dnn-max", type=float, default=None)
    p.add_argument("--hme-min", type=float, default=None)
    p.add_argument("--hme-max", type=float, default=None)
    p.add_argument("--max-calls", type=int, default=50000)
    p.add_argument("--strategy", type=int, default=2, help="Minuit2 strategy")
    p.add_argument("--mx", type=int, default=None, help="Mass label override")
    p.add_argument(
        "--no-empty-meta",
        action="store_true",
        help="One χ² term per empty cell (legacy)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.grid):
        sys.exit("[ERROR] grid not found: %s" % args.grid)

    grid = load_grid_json(args.grid)
    cells = grid.get("cells") or []
    if not cells:
        sys.exit("[ERROR] grid has no cells: %s" % args.grid)

    n_neg = sum(1 for c in cells if float(c.get("content", 0)) < 0)
    if n_neg:
        sys.exit(
            "[ERROR] grid has %d negative cells — re-run grid_merge (strict)" % n_neg
        )

    auto_xmin, auto_xmax, auto_ymin, auto_ymax = fit_range_from_positive_cells(cells)
    dnn_min = auto_xmin if args.dnn_min is None else float(args.dnn_min)
    dnn_max = auto_xmax if args.dnn_max is None else float(args.dnn_max)
    hme_min = auto_ymin if args.hme_min is None else float(args.hme_min)
    hme_max = auto_ymax if args.hme_max is None else float(args.hme_max)
    empty_meta = not args.no_empty_meta
    thr_max = float(args.thr)
    auto_degree = args.degree is None

    print(
        "[fit_one] grid=%s  cells=%d  window DNN=[%.4g,%.4g] HME=[%.4g,%.4g]"
        % (args.grid, len(cells), dnn_min, dnn_max, hme_min, hme_max),
        flush=True,
    )
    print(
        "[fit_one] ExpPolyLogY2D  degree=%s  thr_max=%g  thr_step=%g  "
        "empty_meta=%s  n_quad=%d"
        % (
            "auto(1..%d)" % args.max_degree if auto_degree else args.degree,
            thr_max,
            args.thr_step,
            empty_meta,
            args.n_quad,
        ),
        flush=True,
    )

    try:
        obs, obs_info = cells_to_observations(
            grid,
            dnn_min,
            dnn_max,
            hme_min,
            hme_max,
            empty_meta=empty_meta,
        )
    except ValueError as exc:
        sys.exit("[ERROR] %s" % exc)

    if len(obs) <= 5:
        sys.exit("[ERROR] too few cells in fit window (%d)" % len(obs))

    print(
        "[fit_one] n_obs=%d  (pos=%d empty_cells=%d empty_rects=%s empty_obs=%d)  "
        "integral=%.6g"
        % (
            obs_info["n_obs"],
            obs_info["n_pos_obs"],
            obs_info["n_empty_cells"],
            obs_info.get("n_empty_rects"),
            obs_info["n_empty_obs"],
            sum(b.content for b in obs),
        ),
        flush=True,
    )

    model, fit, thr_min = select_degree_and_thr(
        dnn_min,
        dnn_max,
        hme_min,
        hme_max,
        obs,
        degree=args.degree,
        max_degree=args.max_degree,
        thr=thr_max,
        thr_step=float(args.thr_step),
        auto_thr=bool(args.auto_thr) or auto_degree,
        n_quad=args.n_quad,
        max_calls=args.max_calls,
        strategy=args.strategy,
    )
    degree = int(fit.get("degree", model.degree))

    mx = args.mx if args.mx is not None else grid.get("mx")
    func_spec = model.to_spec()
    func_spec["fit_mode"] = "chi2_norm"

    result = json_safe(
        {
            "key": model.key,
            "function_name": model.key,
            "model": "exppoly_logy2d",
            "fit_mode": "chi2_norm",
            "function": func_spec,
            "param_names": fit.get("param_names") or list(model.param_names),
            "parameters": fit.get("parameters") or [],
            "errors": fit.get("errors") or [],
            "covariance": fit.get("covariance"),
            "hesse_ok": fit.get("hesse_ok"),
            "npar": model.npar,
            "chi2": fit.get("chi2"),
            "ndf": fit.get("ndf"),
            "chi2ndf": fit.get("chi2ndf"),
            "p_value": fit.get("p_value"),
            "converged": fit.get("converged"),
            "minuit_status": fit.get("minuit_status"),
            "edm": fit.get("edm"),
            "ncalls": fit.get("ncalls"),
            "wall_time": fit.get("wall_time"),
            "ndata": fit.get("ndata"),
            "degree": degree,
            "n_quad": args.n_quad,
            "acceptable_unc_thr": float(fit.get("acceptable_unc_thr", thr_min)),
            "thr_min": float(fit.get("thr_min", thr_min)),
            "thr_max": thr_max,
            "thr_step": float(args.thr_step),
            "thr_search": fit.get("thr_search"),
            "auto_degree": auto_degree,
            "auto_thr": bool(args.auto_thr) or auto_degree,
            "empty_meta": obs_info["empty_meta"],
            "n_pos_obs": obs_info["n_pos_obs"],
            "n_empty_cells": obs_info["n_empty_cells"],
            "n_empty_obs": obs_info["n_empty_obs"],
            "mx": mx,
            "category": grid.get("category"),
            "process": grid.get("process"),
            "grid_file": os.path.abspath(args.grid),
            "source_file": grid.get("source_file"),
            "hist": grid.get("hist"),
            "fit_range": {
                "dnn": [dnn_min, dnn_max],
                "hme": [hme_min, hme_max],
            },
            "fit_range_auto": {
                "dnn": [auto_xmin, auto_xmax],
                "hme": [auto_ymin, auto_ymax],
                "from": "bbox of cells with content > 0",
            },
            "n_grid_cells": len(cells),
            "n_obs_cells": len(obs),
            "integral_window": float(sum(b.content for b in obs)),
            "n_expected_cells": fit.get("n_expected_cells"),
            "grid_meta": {
                k: grid.get(k)
                for k in (
                    "algorithm",
                    "n_neg_before",
                    "n_neg_after",
                    "n_merged",
                    "n_singleton",
                    "nx_fine",
                    "ny_fine",
                )
                if k in grid
            },
        }
    )

    out_path = args.output
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print("[fit_one] wrote %s" % out_path, flush=True)

    ok, reason = fit_quality_ok(result)
    print(
        "[fit_one] %s  degree=%s  thr_min=%s  chi2/ndf=%s  p=%s  "
        "npar=%s  hesse_ok=%s  quality=%s  wall=%.1fs"
        % (
            result.get("key"),
            result.get("degree"),
            result.get("thr_min"),
            result.get("chi2ndf"),
            result.get("p_value"),
            result.get("npar"),
            result.get("hesse_ok"),
            "PASS" if ok else ("FAIL: %s" % reason),
            result.get("wall_time") or 0.0,
        ),
        flush=True,
    )
    if not ok:
        print("[fit_one] ERROR: %s" % reason, flush=True)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
