# Running the analysis

The pipeline — anaTuples, observables, histograms, plots — is the **standard FLAF chain**; follow
the **[FLAF full-workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)** for
the commands. This page collects what is **specific to HH→bb̄WW**.

```sh
ERA=Run3_2022
VER=dev

law run FLAF.Analysis.tasks.HistPlotTask --era $ERA --version $VER --workflow local --test 1000
```

## The b-tag-shape cache runs first

HH→bb̄WW computes **b-tag shape** weights in a dedicated caching stage,
`AnalysisCacheTask` (aggregated by `AnalysisCacheAggregationTask`), which LAW runs automatically
before histogramming — see
[FLAF → Task reference](https://cms-flaf.github.io/FLAF/reference/tasks/#analysiscachetask).

!!! warning "Budget time for the cache on a cold start"
    On a cold cache `AnalysisCacheTask` is **slow** (roughly an hour per branch), even for simple
    variables. When iterating, reuse an existing cache instead of recomputing it, using a
    [per-task version override](https://cms-flaf.github.io/FLAF/workflow/arguments/#per-task-version-overrides),
    e.g. `--AnalysisCacheTask-version <existing> --AnalysisCacheAggregationTask-version <existing>`.

## Mass reconstruction: DeepHME

The HH mass is reconstructed with **DeepHME** (the bb̄WW counterpart to SVfit in bb̄ττ). It is part
of the observable computation and requires no special command — it runs as part of the standard
producer chain.

DeepHME and the DNN producers run their ONNX models with one thread per session
(`FLAF.Common.OnnxSession`), so their outputs do not depend on the core count of the batch node.
They still differ in the last digits between `x86_64-v3` and `x86_64-v4` nodes: compare them with a
tolerance ([NN outputs and the CPU of the node](https://cms-flaf.github.io/FLAF/concepts/environment/#nn-outputs-and-the-cpu-of-the-node)).

## Stored gen-level t̄t information

Processes with `genInfo: [ TT ]` store, next to the usual observables:

| Column | Content | Read by |
|---|---|---|
| `TTInfo_nLeptonicW` | number of leptonically decaying W bosons (0, 1 or 2; τ counts as leptonic) | — (t̄t is not stitched: `TT` uses only the three decay-channel samples) |
| `genTop_{pt,eta,phi,mass}` | last-copy top and anti-top, in this order | top-p<sub>T</sub> reweighting at `AnaTupleFileTask` (`genTop_pt`; configured for Run3_2022–2023BPix) |
| `genTop_b_{pt,eta,phi}`, `genTop_lep_{pt,eta,phi,mass}` | the b quark and the W's charged lepton of each top | — |
| `genTop_lep_gen_kind` | `GenLepton::Kind` of that lepton, -1 for a hadronic W | — |

`TTInfo_*` columns must stay scalars: the anaTuple stores all columns that share a prefix as one
collection, so an array under `TTInfo_` would turn the count into an array as well.

## Corrections in Run3_2024 and later

`config/Run3_2024/global.yaml` (and 2025, 2026) inherits the analysis-wide `corrections:` block
through the anchor `*corrections_default` and overrides only `btag` (UParTAK4, no shape
calibration yet) and `dy_hhbbtautau`. A correction added to `config/global.yaml` — the shape
weights `pu`, `parton_shower`, `top_pt`, `pdf` and `qcd_scale` included — therefore applies to
those eras too; an entry that has to differ there is overridden as a whole.

## Resonant mass grid

The bb̄WW signal is `GluGluTo{Radion,BulkGraviton}` at 40 mass points from 250 GeV to 5 TeV in each
of `Run3_2022` through `Run3_2023BPix`, with the single-lepton (`2B2JLNu`) and dilepton (`2B2L2Nu`)
final states as separate datasets. In 2023 and 2023BPix the whole grid comes from the custom
production (DSProd, read from private storage with `fs_nanoAOD:` + `dirName:`); in 2022 and 2022EE
18 points per final state do and the other 22 are central `Run3Summer22` datasets. The central
X→YH samples are not selected. The bb̄ττ signal (`2B2Tau`, processes
`GluGluTo{Radion,BulkGraviton}_bbTauTau`) has the same 40 masses: in 2023 and 2023BPix 29 points
per spin come from the custom production and 11 are central, in 2022 and 2022EE 7 per spin come
from the custom production and 33 are central. `Run3_Model` carries every hypothesis through
anaTuple and histogram production; a fit picks one later, in the datacard. See the
[physics model](setup.md#production-model).

Two networks consume that grid, and they do not cover the same part of it:

- the **dilepton DNN** is parametric — one model per fold, evaluated at whatever masses
  `DNN.columns` in `config/global.yaml` asks for. It is configured over the produced points up to
  1.5 TeV, and the datacard `param_values` in `config/Datacards/DL/dl_spin{0,2}.yaml` follow
  the same 25 points. The four lowest of them, 250 to 280 GeV, are below the lowest
  `signal_mass_points` of the models currently in `config/DNN/DoubleLepton_Parametric_*_v1`, so
  those scores extrapolate and stay provisional until the network is retrained over this grid.
- the **single-lepton two-stage DNN** has a binary stage that is parametric too, evaluated at the
  `TwoStageDNN.masses` in `config/global.yaml`: the 30 produced points from 250 GeV to 2 TeV. Its
  multiclass stage does not depend on the mass, so neither do the SL signal and control regions
  (see [below](#single-lepton-two-stage-dnn)). The SL cards (`config/Datacards/SL/`)
  fit `SR_SL` alone, in the HME window of each mass (`SR_SL_HMESub_m<MX>`), with the data-driven
  Fakes as a background ([Single-lepton fakes](#single-lepton-fakes)) and the DL preprocess step; no
  TT or W+jets control region is fitted, as they have no fake estimate. Expected limits only; they are
  not in `StatInference.cards` (they read the SL production, see
  [Statistical inference](stat_inference.md));
  the earlier single-file SL card, `config/Datacards/x_hh_bbww_SL_run3.yaml`, is kept for
  reference only and still reads the per-mass regions `SR_SL_M${MX}` that no longer exist.

Every mass point named in a `variables:` list also needs its binning in `config/plot/histograms.yaml`
— a variable with no matching entry there stops `HistProducerFromNTuple` with a `KeyError`.

## Single-lepton two-stage DNN

The `TwoStageDNN` payload producer (`TwoStageDNNProducer` in `Analysis/DNN_Application.py`,
configured under `payload_producers` in `config/global.yaml`) applies two networks per category
(boosted, resolved) and fold. The models and their feature lists are in
`config/DNN/SingleLepton_AllMassMulticlass_ParametricBinary/`: `<category>/multiclass_nParity<p>.onnx`
and `<category>/parametric_binary_nParity<p>.onnx`, next to `{multiclass,binary}_<category>_dnn_config.yaml`.
Model `p` is applied to the events with `event % 4 == (p + 3) % 4`.

1. **Multiclass**, one model for all masses. Its 7 outputs, in the order of `class_names` (Signal,
   TT, ST, WJets, DY, H, VV), are stored as `TwoStageDNN_<class>`. In `Analysis/hh_bbww.py` the
   class with the highest score selects the SL region: `SR_SL` for Signal, `CR_SL_<class>` for the
   others. The regions are built only when one of them is in the active flavour's regions (see
   [Regions and sub-regions](#regions-and-sub-regions)).
2. **Binary**, parametrized by the resonance mass, which is the **last** input column. The producer
   evaluates it at each of `TwoStageDNN.masses`, filling that column with the hypothesis mass in
   GeV (the input normalization is part of the ONNX graph), and stores `TwoStageDNN_M<mass>_binary`.
   The last entry of the binary feature list must be `mass_feature` (`X_mass`), or the producer
   stops; the `X_mass` branch of the event is not read.

What the models return and what is stored is configured per network in `multiclass_output` and
`binary_output`:

| Key | Meaning |
|---|---|
| `num_nodes` | number of model outputs, checked when the model runs (7 multiclass, 2 binary) |
| `type` | what the model returns: `logit` or `probability` |
| `activation` | what turns its logits into probabilities: `softmax` or `sigmoid` |
| `save_as` | what is stored: `logit` (raw), `probability`, or `recomputed_logit` |
| `signal_node` | binary model with more than one output: the signal output (1; 0 is background) |

The current models return raw logits. As configured (`config/global.yaml`), the binary network
stores the raw logit z<sub>1</sub> of its signal node (`save_as: logit`) and the multiclass network
the softmax probability of each class (`save_as: probability`). `recomputed_logit` would store
log(p / (1 − p)) of the softmax probability of each output, with p clipped to
[`logit_clip_eps`, 1 − `logit_clip_eps`] (±23.03 for `1.0e-10`); for the 2-output binary model that
is the logit difference z<sub>1</sub> − z<sub>0</sub>. The raw signal logit z<sub>1</sub> has no
absolute scale (softmax does not change when the same constant is added to both outputs) and no
bound: the binary DNN entries of `config/plot/histograms.yaml` use [−23.5, 23.5] (94 bins in 1D, 47
on the DNN axis of the plots against DeepHME), and scores outside it are in the under- and overflow,
which plots drop. The multiclass entries use the same axis, so their probabilities fill its bins
[0, 0.5), [0.5, 1) and, for p = 1, [1, 1.5). Storing raw logits cannot be configured for a model that returns
probabilities.

An event that no fold scores stops the producer. A non-finite model output only prints a warning,
with the event numbers, and is stored as it is.

## Columns taken from the central tree

`config/global.yaml` lists in `anaTuple_shift_invariant_columns` the anaTuple columns that no
systematic shift changes: event numbers and dataset metadata, generator weights, the pileup, parton-
shower, PDF, scale and top-p<sub>T</sub> weights, pileup truth, the LHE record and the generator truth
of the signal (`gen*`) and of t̄t (`TTInfo_*`, `genTop_*`). FLAF stores them in the central tree only
and fills them in for events that only a shift selected, after checking that every variation agrees.

- Only event-level generator quantities belong there. Generator information attached to a
  reconstructed object — `lepN_gen_*`, `centralJet_matchedGenJet_*`, the jet flavour labels — follows
  the selected object and changes under JES, JER and the lepton scales; the fuse step stops with
  `Column '…' is declared shift-invariant but differs in …` if one is listed.
- `HLT_*` is left out: `Analysis/hh_bbww.py` defines a trigger column that `GetColumnNames()` does not
  list, and a column taken from the central tree is not listed there.
- Changing the list changes the anaTuple layout, so it goes with a new anaTuple production.

## Categories

HH→bb̄WW is analysed in a **boosted** category (a large-radius jet, high-p_T HH topologies) and two
**resolved** ones, `res2b` (both b-jet candidates b-tagged) and `recovery` (only the candidate with the
higher b-tag score b-tagged); `inclusive` holds every boosted and resolved event, resolved events
without a b-tagged candidate included (`Analysis/hh_bbww.py`, `categories` in `config/global.yaml`).
Category and channel selection is driven by `config/global.yaml`; narrow or extend it there or via
your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/).

## Regions and sub-regions

The regions come from the **active histTuple flavour** (`histTuple_flavor`), not from a list you
set. The analysis' booking plugin (`Analysis/hist_booking.py`, named by `hist_booking_module` in
`config/global.yaml`; FLAF's hook is described in its
[Booking plugin reference](https://cms-flaf.github.io/FLAF/reference/tasks/#booking-plugin)) fills
`QCDRegions`, the list `custom_regions` names, from the flavour's `regions:` and the base regions of
its `sub_regions:`. Everything that reads `QCDRegions` (`createKeyFilterDict`, `histTuple_selectors`,
the plots) gets them from there.

- **To change the regions,** select another flavour or edit the flavour's `regions:` in
  `config/global.yaml`. Setting `QCDRegions` in a `user_custom.yaml` has no effect: the plugin
  overwrites it.
- **Sub-regions** (e.g. `HMESub` of `Signal_Extraction_DL_window`): a base region restricted by a
  cut per model point, read from a cuts file (`config/sub_regions/hme_DL_v2610.yaml`). They are
  filled in HistFromNtupleProducer, as keys of their own (`<channel>/SR_HMESub_m<MX>/<category>`),
  with the variables of their model point (`signal_params` in `config/plot/histograms.yaml`;
  variables without `signal_params` go in every point). The plugin's docstring describes the cut
  forms. An entry's `name` is letters and digits only: `Analysis/EqualSignalBinning.py` maps
  `<base>_<name>_m<MX>` back to its base region by the last two underscores.
- **Channels:** an entry with `channels: [e, mu]` books its sub-region keys in these channels only.
- **Variables:** an entry with `variables: [ "TwoStageDNN_M{mass}_binary" ]` books only these
  variables in its sub-regions (`{<parameter>}` replaced by the model point's value); each must be a
  variable of the flavour that belongs to the point. The other variables stay in the base regions.
- **Sidebands:** an entry with `sidebands:` books, instead of its cuts file's window, the sidebands
  of that window per model point, category and channel: up to `count` per side, nearest first, each
  ending at the nearest bin edge at least `min_width` beyond the previous one, starting at the window
  edge or `gap` beyond it. The first sideband that would leave [`low_limit`, `high_limit`] ends the
  side, so a side keeps the sidebands that fit, possibly none. They are named after the entry with `L<i>` below and `H<i>`
  above the window, `i = 1` the nearest (`SR_SL_HMESBL1_m300`). A sideband's cut is the window's cut
  with the window interval replaced, so other parts of the cut (e.g. a DNN cut) apply to it too; as
  any bin-form interval, a sideband ending at the last bin edge takes the overflow.
- **Single lepton** (`Signal_Extraction_SL_window`): the binary DNN of mass MX
  (`TwoStageDNN_M<MX>_binary`, booked at its own mass only) in e and μ, in the SL HME window of MX
  (`config/sub_regions/hme_SL_v2610.yaml`, the [16, 84] % signal window per category,
  `<REG>_HMESub_m<MX>`) of `SR_SL`, the six `CR_SL_*` and the anti-isolated `SR_SL_Fakeable`, and in up
  to two sidebands of ≥ 50 GeV on each side of the window (`<REG>_HMESB{L2,L1,H1,H2}_m<MX>`) of `SR_SL`
  and `SR_SL_Fakeable`, as 1D histograms instead of the 2D `binary_DNN_m<MX>_vs_HME`. The single-lepton
  fit reads the windows of `SR_SL`; the fake-lepton estimate reads the sidebands of `SR_SL` and the
  window and sidebands of `SR_SL_Fakeable`; the `CR_SL_*` windows are booked but feed neither. A side keeps the sidebands within
  [240, 2000] GeV, 240 GeV being the DeepHME turn-on: at the lowest masses the lower side has one
  sideband or none. `DeepHME_mass` is booked in the base regions only. No DNN cut is applied in these
  sub-regions. The sidebands follow the windows of the cuts file per category: a new window file
  changes them with it. The CI flavour books the same sub-regions at MX = 300 from a window file of its
  own (`config/sub_regions/ci_sl_hme.yaml`, whose windows leave room for no, one and two lower
  sidebands), and books `SR_SL_Fakeable`: every base region of a flavour's sub-regions must be in its `regions:`,
  or the plots of that region find no histograms.

## Single-lepton fakes

Events with a non-prompt or misidentified lepton in the single-lepton signal region are estimated from
data: the process `Fakes` ("Non-prompt leptons", `config/processes.yaml`, `Analysis/SL_fakes.py`), a
[data-driven process](https://cms-flaf.github.io/FLAF/configuration/processes-and-models/#data-driven-processes)
of FLAF listed among the backgrounds of `Run3_Model`, `Run3_Model_DYSplit` and `TestModel`. It has no
datasets: `HistMergerTask` computes it after every process is merged, in the merged file of each
`TwoStageDNN_M<MX>_binary` of `Signal_Extraction_SL_window` (the HME sub-regions of
[Regions and sub-regions](#regions-and-sub-regions)), when data was merged and the merge is not
restricted with `--process` or `--dataset`. Other variables, flavours and binary DNNs without SL
windows (1600–2000 GeV) get none.

In the SL window `SR_SL_HMESub_m<MX>`, per channel (e, μ) and category (res2b, recovery, boosted),
`Fakes = max(f, 0) × TT` bin by bin, with f the excess of data over the simulation in units of TT,

    f = (D − O) / T − 1      D data, O the simulated backgrounds other than TT, T TT,

measured without the data of the isolated window, separately below and from the mass's `split` of the
DNN (two parts, each
applied to the TT bins of its part). Two schemes (`scheme` per mass):

- **H:** f from the isolated HME sidebands, `SR_SL_HMESB{L2,L1}_m<MX>` pooled and
  `SR_SL_HMESB{H1,H2}_m<MX>` pooled, interpolated linearly in HME to the window at the middle of each
  region's HME interval;
- **V:** f(isolated sidebands) / f(anti-isolated sidebands) × f(anti-isolated window), the
  anti-isolated regions being `SR_SL_Fakeable_HMESB*_m<MX>` and `SR_SL_Fakeable_HMESub_m<MX>`; f = 0
  where the isolated sidebands or the anti-isolated window show no excess.

**Configuration** (the analysts' choice): V up to 400 GeV, where the lower sidebands are missing or sit
on the DeepHME turn-on, H above; the `split` per mass (the cuts of an earlier study, moved to the
nearest 0.5 edge of the DNN axis; −3.5 at 1200, 1300 and 1500 GeV, as at 1400 GeV, where the study's
cuts left the low part of the sidebands without TT). Every mass of `config/sub_regions/hme_SL_v2610.yaml` needs a `split` (a bin
edge of the binary DNN) and a `scheme`, and every physics model listing `Fakes` a `tt_process`
(`test/test_sl_fakes.py` checks them, and that the scheme's regions exist in every cell). A scheme
whose regions a cell lacks is a configuration error, and so is a `split` that leaves one part with too
little TT (`min_tt`) in a region group of every cell of a mass while the other part is determined in
some cell: the merge stops rather than give no Fakes to the whole mass.

**Uncertainties**, each in its part of the DNN: `FakesStatLow/High` (f shifted by its statistical
error, propagated to first order from the yields of the regions; Up and Down are
max(max(f, 0) ± σ<sub>f</sub>, 0) × TT; per channel, category and era) and `FakesMethodLow/High`
(max(f, 0) ± |max(f<sub>other</sub>, 0) − max(f, 0)| with the other scheme, per channel and category,
correlated across eras; no shift where the other scheme's regions are missing or it is not
determined). The members of the theory uncertainties (QCD scale, PDF) get no Fakes variation: FLAF
estimates data-driven processes for the other sources only, so the sum of eras and the datacards take
the Fakes nominal there. The nominal decides whether f is determined (at least `min_tt` weighted TT
events in each DNN part of each region group and, for V, an excess in the anti-isolated sidebands):
where it is not, Fakes = 0 in that cell (both parts, with zero variations and nuisances, so that a
sum of eras and the datacards find the process everywhere), with a message; that era then adds nothing
to the cell in a sum of eras. Where a variation leaves f undetermined in a part (no TT in a
region group where the nominal had enough or, for V, no excess in the anti-isolated sidebands), that
part keeps the nominal f, applied to the varied TT, with a message naming the cell and the variation.
The variations of the simulation enter through D − O and T; data stay nominal.

**Blinding:** the estimator never reads the data of the isolated window. The merged file holds them,
as FLAF merges data in every region; the sub-regions are not plotted. The merge prints, per cell, the
scheme used or that the cell is not estimated, and no number; in V, "not determined" with enough TT
means that the anti-isolated sidebands show no excess in a part. Fakes appear on no control plot yet;
the SL datacards include them (`config/Datacards/SL/`, with the nuisances of
`config/Datacards/SL/systematics/fakes.yaml`).

## Choosing which variables to histogram

As for any FLAF analysis, the variable set is controlled by the `variables:` list in
`user_custom.yaml` (or `--variables`). A short list keeps test runs fast:

```yaml
variables:
  - lep1_pt
  - ggF_DNN_HH
```

## Control plots

`HistPlotTask` draws the stacked control plots with the HH→bb̄WW settings below; the keys are
FLAF's, described in its [HistPlotTask reference](https://cms-flaf.github.io/FLAF/reference/tasks/#histplottask).

- **Backgrounds** are grouped into one stack entry each (`plot_process_groups` in
  `config/global.yaml`), from the bottom: Diboson + Triboson, Single Higgs, W→ℓν + Jets, DY (or,
  with `Run3_Model_DYSplit`, DY m_ℓℓ ∈ [10, 50] GeV and DY m_ℓℓ > 50 GeV), Single top,
  t̄t + t̄tV.
- **Signal:** only the spin-0 (Radion) X→HH signal is drawn, at M_X = 400, 700 and 1000 GeV, in
  every channel of the eras with these samples (Run3_2022 to Run3_2023BPix, and Run3_22_23): per
  mass, one curve summing its bb̄WW dilepton (which includes HH→bb̄ZZ→2ℓ2ν),
  bb̄WW single-lepton and bb̄ττ samples. Each sample is normalised to σ(X→HH) = 1 pb times the
  branching fraction of its own final state, and the final states do not overlap. The merged
  histograms keep the samples apart (see [Production model](setup.md#production-model)).
  Signals are normalised to the total background (`signal_plot_scale: bkg`), as the legend says.
- **Control regions** (`plot_control_regions: [ TT_CR, DY_CR ]`): no signal is drawn, and the
  data is shown even for variables blinded in the signal regions (`blind: true` in
  `config/plot/histograms.yaml`).
- **Binary DNNs of the SL windows** (`TwoStageDNN_M<MX>_binary`, 250–1500 GeV): data hidden above a
  threshold per mass, region, channel and category (`blind_ranges`, resolved per plot as FLAF resolves
  `x_rebin`). Each mass m′ has a reference per spin and
  channel: 0.2 × its largest s/√b (b = TT) over the fit bins of the three categories of that channel in
  its own HME window (the SR binning of the SL datacards, the score flows in the first and last bin as
  there). In each signal-like region of the score of MX — its HME window `SR_SL_HMESub_m<MX>`, the HME
  side bins of `SR_SL` (`SR_SL_HMESB*_m<MX>`) and `SR_SL` itself — per spin, channel and category, in
  the fit bins of the window of MX, the data are shown from the lowest score up to the first bin where
  s/√b of any of the 25 masses reaches that mass's reference (same spin and channel; a bin without TT
  holding any signal reaches every reference), and hidden from that bin upwards; per channel and
  category the lower of the two spins' edges. A plot where no bin reaches a reference is not blinded.
  `SR_SL`, which holds the window, is hidden at least from the window's threshold of the same channel and
  category. The inclusive category takes the lowest threshold of the channel's three categories (it
  also holds the resolved events whose leading b jet is not tagged, which none of them holds); a
  channel without a key of its own (the dilepton channels, whose SL regions are empty) the lowest of
  the region. Neighbouring masses
  often set the threshold: at low mass a wide bin with thousands of TT events reaches the reference of a
  neighbour although s/b is below 0.1 %, and the side bins of a mass lie where the windows of its
  neighbours are. The signals of the other masses come from the study model `SL_CrossMassSignals`
  ([Setup](setup.md)). `SR_SL_Fakeable` and the `CR_SL_*` are never blinded. A region without a key of
  its own (those of other flavours, e.g. CI) takes the lowest threshold of the window; a CI region with
  an SL name takes that name's key. `test/test_plot_blinding_config.py` checks that
  the thresholds cover every channel, category and region the scores are plotted in. The binary DNNs of
  other masses stay blinded whole (`blind: true`).
- **Labels:** "CMS Preliminary" ("CMS Simulation Preliminary" where the data are blinded); the
  luminosity of each era is the one the simulation is normalised to; axis titles carry their
  units.
- **Tests:** `test/test_plot_config.py` (CI: Unit Tests, through FLAF's reusable workflow) checks,
  with FLAF's own plot rules, in every era, the Run3_22_23 meta-era and both DY models: the
  control regions (exactly TT_CR and DY_CR) with the data they show, the signal normalisation,
  the grouping and stack order of the backgrounds, and the X→HH entries. It does not render the
  plots.

## Statistical interpretation

Continue to [Statistical inference](stat_inference.md) for datacards, limits and diagnostics.
