"""Shared helpers used by more than one script in this directory.

  load_hist_2d  - load TH2 (native bins; no rebin)
  ContinuousDensity2D family (N × pdf, Gauss–Legendre cell integrals):
    ExpPoly2D, ExpPolyCheb2D, ExpPolyLogY2D, LogBern2D, MixtureExpPoly2D
  eval / integrate for templates and plots
  eigen_shape_shifts
"""

import math
import os

import numpy as np
import ROOT
from numpy.polynomial.legendre import leggauss

ROOT.gROOT.SetBatch(True)


# --------------------------------------------------------------------------- data


class AxisRange(object):
    def __init__(self, xmin, xmax, bin_width):
        self.xmin = xmin
        self.xmax = xmax
        self.bin_width = bin_width


class Hist2DData(object):
    """Numpy snapshot of a TH2 + fit windows (native binning)."""

    def __init__(
        self,
        mx,
        category,
        process,
        x_centers,
        y_centers,
        x_edges,
        y_edges,
        content,
        error,
        dnn_range,
        hme_range,
        integral,
        n_entries,
        source_file,
    ):
        self.mx = mx
        self.category = category
        self.process = process
        self.x_centers = x_centers
        self.y_centers = y_centers
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.content = content
        self.error = error
        self.dnn_range = dnn_range
        self.hme_range = hme_range
        self.integral = integral
        self.n_entries = n_entries
        self.source_file = source_file

    @property
    def nx(self):
        return len(self.x_centers)

    @property
    def ny(self):
        return len(self.y_centers)

    @property
    def dx(self):
        return float(self.x_edges[1] - self.x_edges[0])

    @property
    def dy(self):
        return float(self.y_edges[1] - self.y_edges[0])


def _hadd_path(mx, category, data_dir):
    return os.path.join(data_dir, "hadd_m%d_%s.root" % (mx, category))


def _load_th2(path, hist_path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise FileNotFoundError("Cannot open %s" % path)
    h = f.Get(hist_path)
    if not h:
        f.Close()
        raise KeyError("Histogram '%s' not found in %s" % (hist_path, path))
    h.SetDirectory(0)
    f.Close()
    return h


def _th2_to_data(h, mx, category, process, dnn_range, hme_range, source_file):
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    content = np.zeros((nx, ny), dtype=float)
    error = np.zeros((nx, ny), dtype=float)
    x_c = np.zeros(nx)
    y_c = np.zeros(ny)
    x_edges = np.zeros(nx + 1)
    y_edges = np.zeros(ny + 1)
    for ix in range(1, nx + 1):
        x_c[ix - 1] = h.GetXaxis().GetBinCenter(ix)
        x_edges[ix - 1] = h.GetXaxis().GetBinLowEdge(ix)
    x_edges[nx] = h.GetXaxis().GetBinUpEdge(nx)
    for iy in range(1, ny + 1):
        y_c[iy - 1] = h.GetYaxis().GetBinCenter(iy)
        y_edges[iy - 1] = h.GetYaxis().GetBinLowEdge(iy)
    y_edges[ny] = h.GetYaxis().GetBinUpEdge(ny)
    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            content[ix - 1, iy - 1] = h.GetBinContent(ix, iy)
            error[ix - 1, iy - 1] = h.GetBinError(ix, iy)
    return Hist2DData(
        mx=mx,
        category=category,
        process=process,
        x_centers=x_c,
        y_centers=y_c,
        x_edges=x_edges,
        y_edges=y_edges,
        content=content,
        error=error,
        dnn_range=dnn_range,
        hme_range=hme_range,
        integral=float(content.sum()),
        n_entries=float(h.GetEntries()),
        source_file=source_file,
    )


def load_hist_2d(
    mx,
    data_dir,
    category="res2b",
    process="DY",
    dnn_min=-7.5,
    dnn_max=6.5,
    hme_min=240.0,
    hme_max=1200.0,
    hist_name=None,
):
    """Load plots_2d/{process}_2d (or hist_name) with native binning. No rebin."""
    path = _hadd_path(mx, category, data_dir)
    hname = hist_name or ("plots_2d/%s_2d" % process)
    h = _load_th2(path, hname)
    return _th2_to_data(
        h,
        mx=mx,
        category=category,
        process=process,
        dnn_range=AxisRange(dnn_min, dnn_max, h.GetXaxis().GetBinWidth(1)),
        hme_range=AxisRange(hme_min, hme_max, h.GetYaxis().GetBinWidth(1)),
        source_file=path,
    )


# --------------------------------------------------------------------------- hist quality (negatives / empties)


def _window_mask(data, dnn_min=None, dnn_max=None, hme_min=None, hme_max=None):
    """Boolean (nx, ny) mask for bin centres inside the fit window."""
    xr = data.dnn_range
    yr = data.hme_range
    xlo = xr.xmin if dnn_min is None else float(dnn_min)
    xhi = xr.xmax if dnn_max is None else float(dnn_max)
    ylo = yr.xmin if hme_min is None else float(hme_min)
    yhi = yr.xmax if hme_max is None else float(hme_max)
    mx = (data.x_centers >= xlo) & (data.x_centers <= xhi)
    my = (data.y_centers >= ylo) & (data.y_centers <= yhi)
    return mx[:, None] & my[None, :]


def count_negative_bins(data, window_only=True):
    """Number of bins with content < 0 (optionally restricted to fit window)."""
    c = data.content
    if window_only:
        m = _window_mask(data)
        return int(np.sum((c < 0) & m))
    return int(np.sum(c < 0))


def count_empty_bins(data, window_only=True):
    """Number of bins with content == 0 (optionally restricted to fit window)."""
    c = data.content
    if window_only:
        m = _window_mask(data)
        return int(np.sum((c == 0.0) & m))
    return int(np.sum(c == 0.0))


def hist_quality_issues(data, window_only=True, allow_empty=True):
    """Return list of human-readable problems (empty list if clean).

    By default empty bins are not reported (they are acceptable for fits;
    Garwood σ is used).  Negatives always are.
    """
    issues = []
    n_neg = count_negative_bins(data, window_only=window_only)
    scope = "fit-window" if window_only else "full histogram"
    if n_neg:
        issues.append(
            "%d bins with negative yields in %s "
            "(run grid_merge.py to build a non-negative cell grid)"
            % (n_neg, scope)
        )
    if not allow_empty:
        n_empty = count_empty_bins(data, window_only=window_only)
        if n_empty:
            issues.append(
                "%d empty (zero-yield) bins in %s" % (n_empty, scope)
            )
    return issues


def assert_hist_quality(data, window_only=True, allow_empty=True):
    """Hard-fail if the histogram has negative yields.

    Empty bins are allowed by default (``allow_empty=True``).  Negatives
    must be cleared (via ``grid_merge.py``) before fitting.
    """
    issues = hist_quality_issues(
        data, window_only=window_only, allow_empty=allow_empty
    )
    if issues:
        raise ValueError("; ".join(issues))


def th2_bin_stats(h, xmin=None, xmax=None, ymin=None, ymax=None):
    """Count negative / empty / positive bins of a TH2 (optional centre window)."""
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    n_neg = n_empty = n_pos = n_tot = 0
    for ix in range(1, nx + 1):
        cx = h.GetXaxis().GetBinCenter(ix)
        if xmin is not None and cx < xmin:
            continue
        if xmax is not None and cx > xmax:
            continue
        for iy in range(1, ny + 1):
            cy = h.GetYaxis().GetBinCenter(iy)
            if ymin is not None and cy < ymin:
                continue
            if ymax is not None and cy > ymax:
                continue
            c = float(h.GetBinContent(ix, iy))
            n_tot += 1
            if c < 0:
                n_neg += 1
            elif c == 0.0:
                n_empty += 1
            else:
                n_pos += 1
    return {
        "n_tot": n_tot,
        "n_neg": n_neg,
        "n_empty": n_empty,
        "n_pos": n_pos,
        "nx": nx,
        "ny": ny,
    }


# --------------------------------------------------------------------------- Continuous 2D densities (N × pdf)


class ContinuousDensity2D(object):
    """Base: extended yield N times unit pdf on the fit rectangle.

        s(x,y)  = positive unnormalised shape  (subclass)
        pdf     = s / ∬_window s
        rho     = N * pdf
        mu_cell = N · ∬_cell s / ∬_window s   (Gauss–Legendre)

    Subclasses implement ``_shape_at_arr(x, y, shape_par)`` where
    ``shape_par = par[1:]`` (everything except N).
    """

    def __init__(self, xmin, xmax, ymin, ymax, n_quad=4, key=None):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.n_quad = max(int(n_quad), 2)
        self._nodes, self._weights = leggauss(self.n_quad)
        self._Is_cache_key = None
        self._Is_cache_val = None
        self._x0 = 0.5 * (self.xmin + self.xmax)
        self._y0 = 0.5 * (self.ymin + self.ymax)
        self._xhalf = max(0.5 * (self.xmax - self.xmin), 1e-9)
        self._yhalf = max(0.5 * (self.ymax - self.ymin), 1e-9)
        self.key = key or "ContinuousDensity2D"
        self.label = self.key
        # subclasses set param_names / initial_params / npar
        self.param_names = ["N"]
        self.initial_params = [1.0]
        self.npar = 1

    def _shape_at_arr(self, x, y, shape_par):
        raise NotImplementedError

    def _shape_at(self, x, y, shape_par):
        return float(
            np.asarray(
                self._shape_at_arr(np.array([x]), np.array([y]), shape_par)
            ).ravel()[0]
        )

    def _integral_s_rect(self, shape_par, xlo, xhi, ylo, yhi):
        xmid = 0.5 * (xlo + xhi)
        xhalf = 0.5 * (xhi - xlo)
        ymid = 0.5 * (ylo + yhi)
        yhalf = 0.5 * (yhi - ylo)
        if xhalf <= 0 or yhalf <= 0:
            return 0.0
        acc = 0.0
        for i, wi in enumerate(self._weights):
            x = xmid + xhalf * float(self._nodes[i])
            for j, wj in enumerate(self._weights):
                y = ymid + yhalf * float(self._nodes[j])
                acc += wi * wj * self._shape_at(x, y, shape_par)
        return acc * xhalf * yhalf

    def _integral_s_window(self, shape_par):
        key = tuple(float(c) for c in shape_par)
        if key == self._Is_cache_key and self._Is_cache_val is not None:
            return self._Is_cache_val
        val = self._integral_s_rect(
            shape_par, self.xmin, self.xmax, self.ymin, self.ymax
        )
        self._Is_cache_key = key
        self._Is_cache_val = val
        return val

    def continuous_density(self, x, y, par):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        Is = self._integral_s_window(sp)
        if Is <= 0.0:
            return 0.0
        return N * self._shape_at(x, y, sp) / Is

    def expected_bin(self, par, xlo, xhi, ylo, yhi):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        Is = self._integral_s_window(sp)
        if Is <= 0.0:
            return 0.0
        return N * self._integral_s_rect(sp, xlo, xhi, ylo, yhi) / Is

    @staticmethod
    def _cell_subrects(c):
        rects = getattr(c, "rects", None)
        if rects:
            for r in rects:
                yield (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            return
        if hasattr(c, "xlo"):
            yield (float(c.xlo), float(c.xhi), float(c.ylo), float(c.yhi))
            return
        yield (
            float(c.get("xlo", c.get("xmin"))),
            float(c.get("xhi", c.get("xmax"))),
            float(c.get("ylo", c.get("ymin"))),
            float(c.get("yhi", c.get("ymax"))),
        )

    def prepare_cell_gl(self, cells):
        """Precompute GL stencils; multi-rect meta-clusters expand to subrects."""
        nq = self.n_quad
        nodes = self._nodes
        weights = self._weights
        n_cells = len(cells)
        sub_xlo, sub_xhi, sub_ylo, sub_yhi = [], [], [], []
        cell_of_sub = []
        for k, c in enumerate(cells):
            for xlo, xhi, ylo, yhi in self._cell_subrects(c):
                if not (xhi > xlo and yhi > ylo):
                    continue
                sub_xlo.append(xlo)
                sub_xhi.append(xhi)
                sub_ylo.append(ylo)
                sub_yhi.append(yhi)
                cell_of_sub.append(k)
        n_sub = len(sub_xlo)
        xs = np.zeros((n_sub, nq, nq), dtype=float)
        ys = np.zeros((n_sub, nq, nq), dtype=float)
        fac = np.zeros((n_sub, nq, nq), dtype=float)
        for s in range(n_sub):
            xlo, xhi = sub_xlo[s], sub_xhi[s]
            ylo, yhi = sub_ylo[s], sub_yhi[s]
            xmid = 0.5 * (xlo + xhi)
            xhalf = 0.5 * (xhi - xlo)
            ymid = 0.5 * (ylo + yhi)
            yhalf = 0.5 * (yhi - ylo)
            for i in range(nq):
                x = xmid + xhalf * float(nodes[i])
                for j in range(nq):
                    y = ymid + yhalf * float(nodes[j])
                    xs[s, i, j] = x
                    ys[s, i, j] = y
                    fac[s, i, j] = (
                        float(weights[i]) * float(weights[j]) * xhalf * yhalf
                    )
        xmid_w = 0.5 * (self.xmin + self.xmax)
        xhalf_w = 0.5 * (self.xmax - self.xmin)
        ymid_w = 0.5 * (self.ymin + self.ymax)
        yhalf_w = 0.5 * (self.ymax - self.ymin)
        wxs = np.zeros((nq, nq), dtype=float)
        wys = np.zeros((nq, nq), dtype=float)
        wfac = np.zeros((nq, nq), dtype=float)
        for i in range(nq):
            x = xmid_w + xhalf_w * float(nodes[i])
            for j in range(nq):
                y = ymid_w + yhalf_w * float(nodes[j])
                wxs[i, j] = x
                wys[i, j] = y
                wfac[i, j] = (
                    float(weights[i]) * float(weights[j]) * xhalf_w * yhalf_w
                )
        return {
            "xs": xs,
            "ys": ys,
            "fac": fac,
            "wxs": wxs,
            "wys": wys,
            "wfac": wfac,
            "n_cells": n_cells,
            "n_sub": n_sub,
            "cell_of_sub": np.asarray(cell_of_sub, dtype=np.intp),
        }

    def expected_cells_batch(self, par, gl_pack):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        n_cells = gl_pack["n_cells"]
        Is = float(
            np.sum(
                gl_pack["wfac"]
                * self._shape_at_arr(gl_pack["wxs"], gl_pack["wys"], sp)
            )
        )
        if not (Is > 0.0 and math.isfinite(Is)):
            return np.zeros(n_cells, dtype=float)
        s_pts = self._shape_at_arr(gl_pack["xs"], gl_pack["ys"], sp)
        I_sub = np.sum(gl_pack["fac"] * s_pts, axis=(1, 2))
        cell_of_sub = gl_pack.get("cell_of_sub")
        if cell_of_sub is None and I_sub.shape[0] == n_cells:
            return N * I_sub / Is
        I_cell = np.zeros(n_cells, dtype=float)
        np.add.at(I_cell, cell_of_sub, I_sub)
        return N * I_cell / Is

    def __call__(self, x, y, par):
        return self.continuous_density(x, y, par)

    def integrate_bins(self, par, x_edges, y_edges, n_quad=None):
        old_nq = self.n_quad
        if n_quad is not None and int(n_quad) != self.n_quad:
            self.n_quad = max(int(n_quad), 2)
            self._nodes, self._weights = leggauss(self.n_quad)
            self._Is_cache_key = None
        try:
            N = float(par[0])
            sp = [float(par[i]) for i in range(1, self.npar)]
            Is = self._integral_s_window(sp)
            nx = len(x_edges) - 1
            ny = len(y_edges) - 1
            Z = np.zeros((nx, ny), dtype=float)
            if Is <= 0.0:
                return Z
            for ix in range(nx):
                xlo = float(x_edges[ix])
                xhi = float(x_edges[ix + 1])
                for iy in range(ny):
                    ylo = float(y_edges[iy])
                    yhi = float(y_edges[iy + 1])
                    Z[ix, iy] = (
                        N * self._integral_s_rect(sp, xlo, xhi, ylo, yhi) / Is
                    )
            return Z
        finally:
            if n_quad is not None and int(n_quad) != old_nq:
                self.n_quad = old_nq
                self._nodes, self._weights = leggauss(self.n_quad)
                self._Is_cache_key = None

    def eval_on_grid(self, par, x_centers, y_centers):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        Is = self._integral_s_window(sp)
        Z = np.zeros((len(x_centers), len(y_centers)), dtype=float)
        if Is <= 0.0:
            return Z
        for ix, x in enumerate(x_centers):
            for iy, y in enumerate(y_centers):
                Z[ix, iy] = N * self._shape_at(x, y, sp) / Is
        return Z

    def to_spec(self):
        return {
            "model": "continuous",
            "norm": "continuous",
            "param_names": list(self.param_names),
            "npar": self.npar,
            "n_quad": self.n_quad,
            "key": self.key,
        }


def _total_degree_terms(degree):
    terms = []
    for s in range(1, int(degree) + 1):
        for i in range(s + 1):
            terms.append((i, s - i))
    return terms


def _chebyshev_T_powers(z, max_n):
    """Return list T_0..T_max_n evaluated on array z ∈ [-1,1]."""
    z = np.asarray(z, dtype=float)
    out = [np.ones_like(z), z.copy()]
    for n in range(2, max_n + 1):
        out.append(2.0 * z * out[-1] - out[-2])
    return out[: max_n + 1]


def _bernstein_basis(t, n):
    """Bernstein basis B_0^n .. B_n^n on t ∈ [0,1]. Returns list of arrays."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    om = 1.0 - t
    return [
        float(math.comb(n, i)) * (t**i) * (om ** (n - i)) for i in range(n + 1)
    ]


class ExpPoly2D(ContinuousDensity2D):
    """exp(polynomial) on normalised (xn,yn) ∈ [-1,1] (monomial basis).

        s = exp( sum_{1≤i+j≤deg} a_ij · xn^i · yn^j )
    """

    def __init__(self, xmin, xmax, ymin, ymax, degree=4, key=None, n_quad=4):
        super(ExpPoly2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        self.param_names = ["N"] + ["a_%d_%d" % (i, j) for i, j in self._terms]
        self.npar = len(self.param_names)
        self.key = key or ("ExpPoly2D-%d" % self.degree)
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            inits.append(-1.0 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits

    def _shape_at_arr(self, x, y, coeffs):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * (xn**i) * (yn**j)
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "exppoly2d",
            "norm": "continuous",
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class ExpPolyCheb2D(ContinuousDensity2D):
    """Same total-degree exp-poly, Chebyshev T_i(xn) T_j(yn) basis.

    Better-conditioned at high degree than monomials (same npar as ExpPoly2D).
    """

    def __init__(self, xmin, xmax, ymin, ymax, degree=4, key=None, n_quad=4):
        super(ExpPolyCheb2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        self.param_names = ["N"] + [
            "c_%d_%d" % (i, j) for i, j in self._terms
        ]
        self.npar = len(self.param_names)
        self.key = key or ("ExpPolyCheb2D-%d" % self.degree)
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            # mild falling HME tilt via T_1(yn)=yn
            inits.append(-0.5 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits

    def _shape_at_arr(self, x, y, coeffs):
        xn = np.clip(
            (np.asarray(x, dtype=float) - self._x0) / self._xhalf, -1.0, 1.0
        )
        yn = np.clip(
            (np.asarray(y, dtype=float) - self._y0) / self._yhalf, -1.0, 1.0
        )
        max_n = self.degree
        Tx = _chebyshev_T_powers(xn, max_n)
        Ty = _chebyshev_T_powers(yn, max_n)
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * Tx[i] * Ty[j]
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "exppoly_cheb2d",
            "norm": "continuous",
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class ExpPolyLogY2D(ContinuousDensity2D):
    """ExpPoly on (xn, log-mapped yn): better for steeply falling HME.

    yn_log maps HME ∈ [ymin,ymax] → [-1,1] via
      t = log( (y - ymin + eps) / (ymax - ymin + eps) ) renormalised.
    """

    def __init__(self, xmin, xmax, ymin, ymax, degree=4, key=None, n_quad=4):
        super(ExpPolyLogY2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        self.param_names = ["N"] + ["a_%d_%d" % (i, j) for i, j in self._terms]
        self.npar = len(self.param_names)
        self.key = key or ("ExpPolyLogY2D-%d" % self.degree)
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            inits.append(-1.0 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits
        # log map: u = log1p((y-ymin)/span) / log1p(1) → [0,1] then → [-1,1]
        self._y_eps = 1e-3
        self._log_span = math.log1p(
            (self.ymax - self.ymin + self._y_eps) / self._y_eps
        )

    def _yn_log(self, y):
        y = np.asarray(y, dtype=float)
        u = np.log1p((y - self.ymin + self._y_eps) / self._y_eps) / max(
            self._log_span, 1e-12
        )
        return np.clip(2.0 * u - 1.0, -1.0, 1.0)

    def _shape_at_arr(self, x, y, coeffs):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = self._yn_log(y)
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * (xn**i) * (yn**j)
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "exppoly_logy2d",
            "norm": "continuous",
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class LogBern2D(ContinuousDensity2D):
    """log-Bernstein density: s = exp( sum c_ij B_i^{nx}(u) B_j^{ny}(v) ).

    u,v map the fit window to [0,1].  Constant term c_00 dropped (absorbed in N).
    More stable than high-degree monomials on a compact domain.
    """

    def __init__(
        self, xmin, xmax, ymin, ymax, nx=4, ny=4, key=None, n_quad=4
    ):
        super(LogBern2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.nx = int(nx)
        self.ny = int(ny)
        self._terms = []
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                if i == 0 and j == 0:
                    continue
                self._terms.append((i, j))
        self.param_names = ["N"] + [
            "b_%d_%d" % (i, j) for i, j in self._terms
        ]
        self.npar = len(self.param_names)
        self.key = key or ("LogBern2D-%d-%d" % (self.nx, self.ny))
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            # mild fall in HME: higher j → smaller v → prefer negative b_0_1
            inits.append(-0.5 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits

    def _uv(self, x, y):
        u = (np.asarray(x, dtype=float) - self.xmin) / max(
            self.xmax - self.xmin, 1e-12
        )
        v = (np.asarray(y, dtype=float) - self.ymin) / max(
            self.ymax - self.ymin, 1e-12
        )
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    def _shape_at_arr(self, x, y, coeffs):
        u, v = self._uv(x, y)
        Bu = _bernstein_basis(u, self.nx)
        Bv = _bernstein_basis(v, self.ny)
        arg = np.zeros(np.broadcast(u, v).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * Bu[i] * Bv[j]
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "logbern2d",
            "norm": "continuous",
            "nx": self.nx,
            "ny": self.ny,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class MixtureExpPoly2D(ContinuousDensity2D):
    """Mixture of K ExpPoly2D components (softmax weights).

        s = sum_k w_k · exp(poly_k),   w = softmax(logits)

    Captures multi-region structure (e.g. low-HME peak vs high-HME tail)
    with moderate per-component degree.
    """

    def __init__(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        n_comp=2,
        degree=3,
        key=None,
        n_quad=4,
    ):
        super(MixtureExpPoly2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.n_comp = max(int(n_comp), 2)
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        n_shape = len(self._terms)
        # params: N, logit_1..logit_{K-1}, then K blocks of shape coeffs
        names = ["N"]
        for k in range(1, self.n_comp):
            names.append("logit_%d" % k)
        for k in range(self.n_comp):
            for i, j in self._terms:
                names.append("a%d_%d_%d" % (k, i, j))
        self.param_names = names
        self.npar = len(names)
        self.key = key or (
            "MixExpPoly2D-k%d-d%d" % (self.n_comp, self.degree)
        )
        self.label = self.key
        inits = [1.0]
        for _k in range(1, self.n_comp):
            inits.append(0.0)  # equal weights
        for k in range(self.n_comp):
            for i, j in self._terms:
                # diversify components slightly
                tilt = -1.0 - 0.3 * k
                inits.append(tilt if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits
        self._n_shape = n_shape

    def _weights_and_blocks(self, shape_par):
        K = self.n_comp
        logits = [0.0]  # component 0 reference
        for k in range(1, K):
            logits.append(float(shape_par[k - 1]))
        m = max(logits)
        ex = [math.exp(min(l - m, 50.0)) for l in logits]
        s = sum(ex)
        w = [e / s for e in ex]
        blocks = []
        off = K - 1
        for k in range(K):
            blocks.append(
                [float(shape_par[off + k * self._n_shape + t]) for t in range(self._n_shape)]
            )
        return w, blocks

    def _shape_at_arr(self, x, y, shape_par):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        w, blocks = self._weights_and_blocks(shape_par)
        acc = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for wk, coeffs in zip(w, blocks):
            arg = np.zeros_like(acc)
            for (i, j), a in zip(self._terms, coeffs):
                arg = arg + float(a) * (xn**i) * (yn**j)
            np.clip(arg, -50.0, 50.0, out=arg)
            acc = acc + wk * np.exp(arg)
        return acc

    def to_spec(self):
        return {
            "model": "mixture_exppoly2d",
            "norm": "continuous",
            "n_comp": self.n_comp,
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class ExpPolySep2D(ContinuousDensity2D):
    """Separable exp-poly: s = exp(P_dx(xn) + Q_dy(yn)) (no cross terms).

    Cheap baseline; underfits when DNN–HME correlation is strong.
    """

    def __init__(
        self, xmin, xmax, ymin, ymax, dx=4, dy=4, key=None, n_quad=4
    ):
        super(ExpPolySep2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.dx = int(dx)
        self.dy = int(dy)
        names = ["N"]
        for i in range(1, self.dx + 1):
            names.append("px_%d" % i)
        for j in range(1, self.dy + 1):
            names.append("qy_%d" % j)
        self.param_names = names
        self.npar = len(names)
        self.key = key or ("ExpPolySep2D-%d-%d" % (self.dx, self.dy))
        self.label = self.key
        inits = [1.0] + [0.0] * self.dx + [-1.0] + [0.0] * (self.dy - 1)
        self.initial_params = inits

    def _shape_at_arr(self, x, y, coeffs):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        off = 0
        for i in range(1, self.dx + 1):
            arg = arg + float(coeffs[off]) * (xn**i)
            off += 1
        for j in range(1, self.dy + 1):
            arg = arg + float(coeffs[off]) * (yn**j)
            off += 1
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "exppoly_sep2d",
            "norm": "continuous",
            "dx": self.dx,
            "dy": self.dy,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "n_quad": self.n_quad,
        }


def make_density_2d(
    name,
    xmin,
    xmax,
    ymin,
    ymax,
    degree=4,
    n_quad=4,
    nx=None,
    ny=None,
    n_comp=2,
    dx=None,
    dy=None,
    key=None,
):
    """Factory for continuous 2D density models.

    Names (case-insensitive):
      exppoly | expploy2d | poly
      cheb | exppoly_cheb
      logy | exppoly_logy
      logbern | bern
      mix | mixture
      sep | separable
    """
    n = (name or "exppoly").strip().lower().replace("-", "_")
    if n in ("exppoly", "exppoly2d", "poly", "exp_poly"):
        return ExpPoly2D(
            xmin, xmax, ymin, ymax, degree=degree, n_quad=n_quad, key=key
        )
    if n in ("cheb", "exppoly_cheb", "exppolycheb2d", "chebyshev"):
        return ExpPolyCheb2D(
            xmin, xmax, ymin, ymax, degree=degree, n_quad=n_quad, key=key
        )
    if n in ("logy", "exppoly_logy", "exppolylogy2d", "loghme"):
        return ExpPolyLogY2D(
            xmin, xmax, ymin, ymax, degree=degree, n_quad=n_quad, key=key
        )
    if n in ("logbern", "bern", "logbern2d", "bernstein"):
        return LogBern2D(
            xmin,
            xmax,
            ymin,
            ymax,
            nx=int(nx if nx is not None else degree),
            ny=int(ny if ny is not None else degree),
            n_quad=n_quad,
            key=key,
        )
    if n in ("mix", "mixture", "mixexppoly", "mixture_exppoly2d"):
        return MixtureExpPoly2D(
            xmin,
            xmax,
            ymin,
            ymax,
            n_comp=n_comp,
            degree=degree,
            n_quad=n_quad,
            key=key,
        )
    if n in ("sep", "separable", "exppoly_sep", "exppolysep2d"):
        return ExpPolySep2D(
            xmin,
            xmax,
            ymin,
            ymax,
            dx=int(dx if dx is not None else degree),
            dy=int(dy if dy is not None else degree),
            n_quad=n_quad,
            key=key,
        )
    raise ValueError("unknown density model %r" % name)


def rebuild_exppoly2d(result):
    """Rebuild continuous density from fit JSON (any ContinuousDensity2D)."""
    return rebuild_density_2d(result)


def rebuild_density_2d(result):
    """Rebuild continuous density from fit JSON."""
    spec = result.get("function") or {}
    model_name = (
        spec.get("model")
        or result.get("model")
        or "exppoly2d"
    )
    n_quad = int(spec.get("n_quad") or result.get("n_quad") or 4)
    deg = int(spec.get("degree") or result.get("degree") or 4)
    edges = result.get("fit_range_edges") or {}
    fr = result.get("fit_range") or {}
    dr = edges.get("dnn") or fr.get("dnn") or [-7.5, 6.5]
    hr = edges.get("hme") or fr.get("hme") or [240.0, 1200.0]
    xmin, xmax = float(dr[0]), float(dr[1])
    ymin, ymax = float(hr[0]), float(hr[1])
    key = result.get("key")
    m = str(model_name).lower()
    if m in ("exppoly2d", "exppoly", "poly"):
        return ExpPoly2D(
            xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad
        )
    if m in ("exppoly_cheb2d", "cheb", "exppolycheb2d"):
        return ExpPolyCheb2D(
            xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad
        )
    if m in ("exppoly_logy2d", "logy", "exppolylogy2d"):
        return ExpPolyLogY2D(
            xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad
        )
    if m in ("logbern2d", "logbern", "bernstein"):
        return LogBern2D(
            xmin,
            xmax,
            ymin,
            ymax,
            nx=int(spec.get("nx") or deg),
            ny=int(spec.get("ny") or deg),
            key=key,
            n_quad=n_quad,
        )
    if m in ("mixture_exppoly2d", "mix", "mixture"):
        return MixtureExpPoly2D(
            xmin,
            xmax,
            ymin,
            ymax,
            n_comp=int(spec.get("n_comp") or 2),
            degree=deg,
            key=key,
            n_quad=n_quad,
        )
    if m in ("exppoly_sep2d", "sep", "separable"):
        return ExpPolySep2D(
            xmin,
            xmax,
            ymin,
            ymax,
            dx=int(spec.get("dx") or deg),
            dy=int(spec.get("dy") or deg),
            key=key,
            n_quad=n_quad,
        )
    # default fallback
    return ExpPoly2D(
        xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad
    )


def eval_model_grid(result, x_c, y_c, params=None):
    """Pointwise density N*pdf (not bin counts). Prefer integrate_model_bins."""
    model = rebuild_density_2d(result)
    par = list(params if params is not None else result["parameters"])
    return model.eval_on_grid(par, list(x_c), list(y_c))


def integrate_model_bins(result, x_edges, y_edges, params=None, n_quad=8):
    """Integrate continuous N*pdf over each (x,y) bin rectangle."""
    model = rebuild_density_2d(result)
    if n_quad is not None:
        model.n_quad = max(int(n_quad), 2)
        model._nodes, model._weights = leggauss(model.n_quad)
        model._Is_cache_key = None
    par = list(params if params is not None else result["parameters"])
    return model.integrate_bins(par, x_edges, y_edges)


# --------------------------------------------------------------------------- cov eigenmodes


def eigen_shape_shifts(
    covariance,
    skip_indices=None,
    max_modes=0,
    min_eigenvalue=1e-12,
):
    """Return [(rank, sqrt(lambda), unit eigenvector in full param space), ...]."""
    V = np.array(covariance, dtype=float)
    n = V.shape[0]
    skip = set(skip_indices or [])
    keep = [i for i in range(n) if i not in skip]
    if not keep:
        return []
    Vsub = 0.5 * (V[np.ix_(keep, keep)] + V[np.ix_(keep, keep)].T)
    try:
        evals, evecs = np.linalg.eigh(Vsub)
    except np.linalg.LinAlgError:
        return []
    modes = []
    for rank, idx in enumerate(np.argsort(evals)[::-1]):
        lam = float(evals[idx])
        if lam <= min_eigenvalue:
            continue
        if max_modes and len(modes) >= max_modes:
            break
        u_full = np.zeros(n)
        for j, gi in enumerate(keep):
            u_full[gi] = float(evecs[j, idx])
        norm = float(np.linalg.norm(u_full))
        if norm > 0:
            u_full /= norm
        modes.append((rank, math.sqrt(lam), u_full))
    return modes
