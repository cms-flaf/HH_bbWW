# DY 2D adaptive-grid fit

Fit DY in (DNN, HME) with a continuous **ExpPolyLogY2D** density on an adaptive
non-uniform cell grid built by `grid_merge.py`.

## Pipeline

| File | Role |
|------|------|
| `grid_merge.py` | Adaptive non-uniform cells (mega-free residual merge) → JSON + PDF |
| `fit_one.py` | Minuit2 chi2_norm, **ExpPolyLogY2D only**, auto degree×thr |
| `core.py` | TH2 I/O, continuous density models, template helpers |
| `make_hist_from_fit.py` | Fit JSON → TH2 templates (+ eigen up/down) |
| `plot_fit_slices.py` | DY vs fit HME slices (PDF) |

## Quick start

```bash
cd /path/to/this/directory
DATA=/path/to/hadd_files

python3 -u grid_merge.py \
    --input "$DATA/hadd_m500_res2b.root" --hist plots_2d/DY_2d --mx 500 \
    --output-json grid_m500.json --output-pdf grid_m500.pdf

# Auto: degree 1…8, thr = 0…10% in 1% steps → first p≥0.05 + covariance
python3 -u fit_one.py --grid grid_m500.json --output m500_fit.json

# Fixed degree, fixed thr = 10%
python3 -u fit_one.py --grid grid_m500.json --output m500_fit.json --degree 4 --thr 0.1

# Fixed degree, scan thr up to 10%
python3 -u fit_one.py --grid grid_m500.json --output m500_fit.json --degree 4 --auto-thr
```

Fit JSON includes `parameters`, `degree`, `thr_min`, `chi2`, `p_value`, covariance.

Templates from fit (`make_hist_from_fit.py`) use `thr_min` for a bin-by-bin
accepted-unc systematic: `δ_i = thr·μ_i` → `nominal` bin errors, plus
`nominal_acceptedUnc`, `nominal_acceptedUncUp`, `nominal_acceptedUncDown`.

## Model

```
s(x,y) = exp( Σ a_ij · xn^i · yn_log^j )   (1 ≤ i+j ≤ degree)
pdf    = s / ∬_window s
μ_cell = N · ∬_cell pdf     (Gauss–Legendre)
```

- `xn` maps DNN to [-1,1]; `yn_log` is a log map of HME to [-1,1].
- Empty cells: one meta-cluster term, Garwood σ for n=0.
- Optional accepted unc.: `σ² = σ_stat² + (thr · |content|)²` (default thr max 0.1).

## Data layout

```
{data-dir}/hadd_m{MX}_{category}.root
  plots_2d/DY_2d      TH2  x=DNN, y=HME
  plots_2d/signal_2d  TH2  (optional, equal-signal DNN slicing)
```
