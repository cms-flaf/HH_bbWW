# v2610 parametric DNN: studies for the analysis-note appendix

Record of the studies behind the dilepton parametric DNN (pDNN) trained on the v2610 production:
what was tested, how, the result, and the decision taken. Figures are collected in
`/eos/user/d/daebi/HH_bbWW/DNNTraining/pytorch/appendix/` (file names quoted below); every trained
model is in `/eos/user/d/daebi/HH_bbWW/DNNTraining/pytorch/<model>/` with its code snapshot,
logs, limits and checks. Code: `Studies/DNN_pytorch`, `Studies/DNN_Dataset` (branch
`pdnn-v2610-training`).

## 1. Training dataset

- **Inputs.** v2610 HistTuples (`histTuple_flavor: DNN_training`) of the four 2022–2023 eras,
  read from two sites: Run3_2022/2022EE on LPC (`/store/user/lpcflaf/HH_bbWW/v2610_training`),
  Run3_2023/2023BPix on CERN EOS (`/eos/user/a/abolshov/HH_bbWW/v2610_training`). Every
  HistTuple was checked against its anaTuple input (all present, readable, same 178 branches).
- **Selection.** `SR`, `res2b || recovery || boosted`, dilepton channels (ee, eμ, μμ).
- **Samples and classes** (`class_value`): Radion / BulkGraviton signals in 2L (0 / −3), 1L
  (−1 / −4) and bbττ (−2 / −5) at 30 masses, 250–2000 GeV; backgrounds TT 1, DY 2, ST 3,
  single H 4, VV 5, W 6, ttV 7, VVV 8 (all analysis background processes, one class each). 250
  datasets, every one with inputs in all four eras.
- **Folds.** `event % 4` defines four parity files, about 6.2 M events each, 180 branches.
- **Validation of the dataset maker.** The new `create_dataset.py` reproduces the July25 dataset
  used by the previous pDNN (same branches, weight tree and per-parity yields, identical event
  sets).
- Location: `/eos/user/d/daebi/HH_bbWW/DNNDatasets/v2610_training/Dataset`.

## 2. Input-variable study (before training)

- **Method.** On parity 0, each of the 128 numeric HistTuple variables: weighted single-variable
  separation (2·|AUC−0.5|) of signal (2L + bbττ, both spins, flat in mass) vs class-balanced
  background in three mass bands, background-pair separations (TT/DY/Other), correlation with
  DeepHME_mass, fraction of default values. Table: `01_input_variable_study.txt`.
- **Findings.** Low mass (250–400 GeV) has no strong single variable (best: b-tag scores ~0.4);
  the strongest variables at higher mass (`MX_reduced`, `Pzeta`, `bbllmet_mass`, `HT_total`,
  MT2) correlate 0.8–0.9 with DeepHME_mass; about 35 semileptonic variables are constant in
  dilepton events; several exact duplicates.
- **Decision.** First feature set = previous production (v6) inputs without DeepHME_mass, plus
  `MX_reduced`, `Pzeta`, `MT2_bb`, `MT2_ll` and `DeepHME_mass_error_rel`
  (= error / DeepHME_mass, uncorrelated with the mass). The fat-jet HbbvsQCD score is not used:
  only its 0.92 working point is calibrated.

## 3. Training setup (model v1)

- **Signal.** Radion + BulkGraviton, 2L + bbττ; 1L signal events are not trained on.
- **Outputs.** Signal / TT / DY / Other (softmax); the signal log-odds is the analysis score.
- **Parametric in mass.** Input `X_mass`; signal at its own mass, backgrounds drawn uniformly from
  the 30 masses each epoch; `ll_dR` and `bb_dR` enter as `*_scaled = dR · X_mass`.
- **Weights.** From `weight_Central`: negative weights set to 0, clipped at mean + 3σ; signal
  flattened to equal total weight per mass point (except masses with < 1000 signal events, i.e.
  boosted 250–400 GeV, which keep their own weight); each class scaled to the signal total, per
  regime.
- **Networks.** Separate resolved and boosted networks; MLP 512-256-128-64, SiLU, batch norm,
  dropout 0.35; AdamW (lr 2e-3, weight decay 0.01), batch 8192; LR halved after 6 epochs without
  improvement; early stopping after 10, at most 150 epochs.
- **Folds.** Cycle i trains on parity i, early-stops on parity i+1, is validated on parity i+2
  and applied to parity i+3. Every event is scored once by a model that never saw it.
- **Hardware.** lxplus-gpu (Tesla T4), ~10 s per resolved epoch; one model is 4 jobs of
  ~45 minutes. Loss curves: `02_loss_curve_*.png`. Typical behaviour: the resolved network
  improves slowly for 90–150 epochs with test loss at or below training loss; the boosted
  network stops at 19–40 epochs as its test loss turns up.

## 4. Limit metric

`limit_pdnn.py`: for each mass and spin hypothesis, templates of the signal log-odds in res2b,
recovery and boosted from the validation scores of all four cycles (full statistics, physical
weights), binned in equal-signal quantiles merged until each bin has ≥ 1 background event and
≥ 10 background MC events; stat-only expected limits on r (`combine -M AsymptoticLimits -t -1`,
`--rAbsAcc 1e-6 --rRelAcc 1e-3`), per category and combined. Signal hypothesis: 2L + bbττ of
one spin (spin 0 Radion, spin 2 Graviton) at that mass; backgrounds TT, DY, Other.
Result for v1: `02_limits_v1_combined.png`.

## 5. Overtraining

- **Method.** `overtraining_check.py`: each cycle's model on its training parity vs its
  validation parity, at 5 masses (and both spins for the spin model), weighted KS test per class
  and signal-vs-background AUC.
- **Result** (all four cycles):

  | Model | Regime | AUC(train) − AUC(val): mean | range |
  |---|---|---|---|
  | v1 | resolved | +0.0002 | −0.0008 … +0.0022 |
  | v1 | boosted | +0.0026 | −0.088 … +0.067 |
  | v1_spin | resolved | +0.0003 | −0.0009 … +0.0038 |
  | v1_spin | boosted | +0.0020 | −0.088 … +0.085 |

  KS p < 0.01 in 4 of 400 tests. No significant overtraining. The boosted spread comes from its
  small samples (~17k background events per parity), and early stopping limits the slight
  fitting of the training events. Figures: `03_overtraining_*.pdf`.

## 6. Spin: parametric vs inclusive

- **Test.** v1 (spin-inclusive) vs v1_spin (same model with the spin hypothesis as an input:
  signal carries its spin, backgrounds a random one).
- **Result.** Limit ratio v1_spin / v1, combined categories: spin 0 average 0.997 (1.03 below
  450 GeV, 1.00 at 500–1000, 0.96 above 1 TeV); spin 2 average 0.991. Point-to-point
  differences are within the training noise (section 7).
  Figures: `04_limits_spin_inclusive_vs_parametric_*.png`.
- **Decision.** The production model is **spin-inclusive**.

## 7. Training-to-training noise

- **Test.** v1 retrained with another random seed (v1_seed2).
- **Result.** Ratio of expected limits (combined): RMS deviation ~3% at 250–400 GeV, 2–3% at
  450–900 GeV, 4–6% above 1 TeV (largest single points 6%, 7%, 12%); res2b alone 1–3% at
  intermediate mass; boosted alone up to 7.5% RMS. Figure: `05_limits_seed_noise_combined.png`.
- **Use.** A model change is significant only beyond this band.

## 8. Feature importance (v1)

- **Method.** `feature_importance.py`: on each cycle's validation parity, at 10 masses, each input
  is shuffled (mass-scaled inputs recomputed), as is each group of inputs correlated with
  |r| > 0.8 in background. Measured: relative increase of the AUC deficit,
  (AUC − AUC_shuffled)/(1 − AUC), and of the class-balanced loss; mean and spread over cycles.
  Tables and figures: `06_importance_v1_{resolved,boosted}.{csv,png}`,
  `06_importance_vs_mass_*.png`.
- **Resolved.** The mass-like group (`MX_reduced`, `bbllmet_mass`, `MT2_blbl1/2`, `bb_pt`)
  dominates (AUC deficit ×62 when shuffled together, ×179 above 1 TeV), then `HT_total`+`Pzeta`;
  b-tag scores and `ll_met_MT_WithMass` are the most important at low mass. Negligible:
  `ll_dphi`, `met_ll_dphi`, `ll_bb_pt_ratio`, `DeepHME_mass_error_rel` (≤ 0.02).
- **Boosted.** Most important: the mass-like group, the transverse-mass group, `ll_bb_dR`,
  `ll_dR_scaled`, fat-jet mass. Negligible: `met_ll_dphi`, `fatbjet_tau4`, `lep1_pt`,
  `DeepHME_mass_error_rel`, `ll_dphi`, `MT`, `fatbjet_tau3`.
- `DeepHME_mass_error_rel` is not used by either network and is dropped.

## 9. Pruned models

- **Models.** v2a (moderate, 31 resolved / 27 boosted inputs) and v2b (aggressive, 26 / 21),
  from section 8; the removed inputs are listed in `model_configs/v2610_pDNN_v2{a,b}.yaml`.
- **Result.** Mean ratio of expected limits to the geometric mean of v1 and v1_seed2, in the
  bands 250–400 | 450–900 | 1000–2000 GeV:

  | Category | v1_seed2 (noise) | v2a | v2b |
  |---|---|---|---|
  | combined, spin 0 | 1.010 / 0.994 / 0.987 | 1.001 / 0.996 / 0.971 | 1.013 / 0.963 / 0.954 |
  | combined, spin 2 | 1.009 / 0.988 / 0.995 | 1.003 / 0.994 / 0.989 | 1.020 / 0.952 / 0.954 |
  | res2b, spin 0 | 1.011 / 0.999 / 1.011 | 1.000 / 1.020 / 0.974 | 1.014 / 1.012 / 1.005 |
  | boosted, spin 0 | 0.989 / 0.990 / 0.985 | 0.990 / 0.922 / 0.970 | 0.920 / 0.876 / 0.949 |
  | boosted, spin 2 | 0.979 / 0.984 / 0.995 | 0.941 / 0.926 / 0.988 | 0.951 / 0.867 / 0.949 |

  Removing the inputs costs no sensitivity. v2b is better by 5–13% in boosted, beyond the seed
  noise, consistent with less overfitting of the small boosted background sample, and 4–5% better
  combined above 450 GeV. Figures: `07_limits_pruned_models_*.png`.
- **Decision (proposed).** v2b is the optimized input set.

## 10. DeepHME_mass: DNN input vs 2D fit (to do)

- Planned comparison, all with the same limit chain:
  1. 1D fit of the optimized DNN without DeepHME_mass;
  2. 2D fit of that DNN score × DeepHME_mass;
  3. 1D fit of the same DNN with DeepHME_mass as an input.

## Commits

`da45ca3` dataset maker; `b09ee5c` training, limits, overtraining; `5e4fcf5` seed; `2dad9b1`,
`786328a` feature importance; `6f4e26b` pruned configs.
