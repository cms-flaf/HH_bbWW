# Statistical inference

The final step turns the merged histograms into **datacards** and runs limits with
[Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/), via the `StatInference` and
`inference` submodules. See
[FLAF → walkthrough, stage 5](https://cms-flaf.github.io/FLAF/workflow/walkthrough/#stage-5-statistical-inference)
for where this sits in the pipeline.

These commands run inside CMSSW/Combine, so prefix them with `cmsEnv` (or open one subshell):

```sh
cmsEnv /bin/zsh        # a CMSSW+Combine subshell
```

## 1. Make datacards and run limits

One law chain takes the merged histograms all the way to the overlay limit plots. The
datacard step itself consumes 1D shapes and does no rebinning of its own; the shapes this
analysis feeds it are cut from the 2D DNN-vs-HME plane by a configurable preprocessing
step the chain runs first — see [Where the binning is decided](#where-the-binning-is-decided):

```sh
law run PlotResonantLimitsTask \
  --version dev \
  --hists-version VERSION_OF_THE_MERGED_HISTS \
  --period Run3_2022 \
  --workflow local
```

It runs, in order:

| Task | Does |
| --- | --- |
| `PreprocessShapesTask` | runs the configuration's `preprocess:` step, if it declares one |
| `CreateDatacardsTask` | builds the datacards from the resulting shapes |
| `ResonantLimitsTask` | runs combine and combines the per-era cards per mass point |
| `PlotResonantLimitsTask` | draws the plots declared in the configuration's `limit_plots` |

Each `limit_plots` entry becomes one overlay of its datacard globs. An entry that also
sets `bands: true` gets, in addition, the standard single-curve plot with the ±1σ/±2σ
bands for each of its curves — the overlay draws expected lines only. Note the dhi task
of almost the same name below (`PlotResonantLimits`, no `Task`): that is the one this
task shells out to for the band plots.

`--version` names what the chain *writes*; `--hists-version` names the `Hists_merged`
tree it *reads*, so a re-binning or a re-fit does not require the input histograms to be
reproduced under a new name.

The analysis configuration is [`config/Datacards/x_hh_bbww_DL_run3.yaml`](https://github.com/cms-flaf/HH_bbWW/blob/main/config/Datacards/x_hh_bbww_DL_run3.yaml),
selected by `StatInference.config` in `config/global.yaml`. It declares the eras,
channels, categories, mass points, processes and uncertainties — the chain reads them
from there, not from `global.yaml`'s own variable lists.

!!! warning "`--period` does not choose the era"
    Which eras get datacards and limits comes from the configuration's `eras:` and
    `era_groups:` blocks. A real era listed inside an `era_groups:` entry is covered by
    that meta-era and is not built standalone, so with the Run 3 configuration the chain
    always builds the `Run3_Early` combination of all four eras regardless of `--period`.
    `--period` is only used to construct a valid FLAF `Setup`.

### Where the binning is decided

By a **preprocessing step the datacard configuration declares**, not by anything in the
chain's own logic. `PreprocessShapesTask` runs whatever `preprocess:` names, supplying
`--input`, `--output`, `--era` and `--config`; it knows nothing about what the step does.
A configuration with no `preprocess:` block skips the task entirely and the datacards are
built from the merged histograms unchanged, so an analysis that needs no preprocessing is
unaffected.

This analysis plugs in `StatInference/bin_opt_2d/rebin_2d.py`, which derives the DNN slice
boundaries and the HME mass-bin edges from the shapes themselves. Each base category
`SR/res2b` becomes the datacard bins `SR/res2b_dnn0…dnn3`.

```yaml
preprocess:
  script: StatInference/bin_opt_2d/rebin_2d.py
  args:
    - --binning-config
    - config/Datacards/binning_2d.yaml
```

The two halves live in different places on purpose. The **knobs** — slice count, bin
budget, and the minimum signal/background yields and effective-entry floors a bin must
satisfy — are analysis configuration, versioned with the card in
[`config/Datacards/binning_2d.yaml`](https://github.com/cms-flaf/HH_bbWW/blob/main/config/Datacards/binning_2d.yaml).
The **derived** `binning.json` is a product, not configuration, and is written into the
task's output on EOS beside the shapes it produced.

Every era in `eras:` is binned on its own statistics and gets its own limit; a group era
from `era_groups:` is binned on its members' summed statistics, and its output keeps the
sub-eras separate (`Run3_Early/{Run3_2022,Run3_2022EE,…}`) so the per-era uncertainties
survive to be combined in the datacard maker.

`rebin_2d.py` is also runnable by hand, which is the quickest way to inspect a binning
without going through the chain; `StatInference/bin_opt_2d/call_rebin_2d.sh` is a worked
example.

The datacard configuration then lists the sliced names in `categories:` and repeats the
`category_pattern` used to write them, which is how the per-category limits group the
slices of one base category back together.

There is also `StatInference/bin_opt/`, an offline combine-driven search over candidate
binnings feeding the `hist_bins` option. This analysis does not use it, and leaves
`hist_bins` unset.

### Running on the 1D DNN shapes instead

[`config/Datacards/x_hh_bbww_DL_run3_1D.yaml`](https://github.com/cms-flaf/HH_bbWW/blob/main/config/Datacards/x_hh_bbww_DL_run3_1D.yaml)
is the same analysis reading the per-mass 1D DNN score variables
(`Hists_merged/<era>/DNN_M<mass>_Signal/`) straight from the merged tree, with no rebin
step in front of it at all. Its datacard bins are the categories exactly as listed
(`SR/res2b`, not `SR/res2b_dnn0`), coarsened by its own `hist_bins:` edge list.

```sh
law run PlotResonantLimitsTask \
  --version limits_1D \
  --hists-version VERSION_OF_THE_MERGED_HISTS \
  --period Run3_2022 \
  --user-custom config/user_custom_1D.yaml
```

!!! warning "Give the 1D run its own `--version`"
    The datacard and limit output paths do not encode which configuration produced them,
    so reusing a 2D run's version overwrites its cards.

### Datacards on their own

To build cards outside the chain — a quick check on shapes that are already rebinned:

```sh
cmsEnv python3 StatInference/dc_make/create_datacards.py \
  --input  PATH_TO_REBINNED_SHAPES \
  --output PATH_TO_CARDS \
  --config config/Datacards/x_hh_bbww_DL_run3.yaml
```

## 2. Run limits on existing datacards

For cards you already have on disk, the dhi task can be called directly:

```sh
law run PlotResonantLimits --version dev --datacards 'PATH_TO_CARDS/*.txt' --xsec fb --y-log
```

Hints:

- add `--workflow htcondor` to submit to the batch system (local by default);
- add `--remove-output 4,a,y` to clear previous outputs;
- add `--print-status 0` to get the workflow status and the output file name;
- options and background: the [cms-hh inference documentation](https://cms-hh.web.cern.ch/tools/inference/).

## 3. Pulls & impacts

```sh
PlotPullsAndImpacts --version dev --datacards "PATH_TO_CARDS/<one_card>.txt" \
  --hh-model NO_STR --parameter-values r=1 --parameter-ranges r,-100,100 \
  --method robust --PlotPullsAndImpacts-order-by-impact True --mc-stats True \
  --PullsAndImpacts-custom-args="--expectSignal=1"
```

!!! warning "One mass point at a time"
    Run pulls & impacts on a **single** datacard, not a glob. Use `--print-status 0` to find the
    output file and `--remove-output 4,a,y` to clear previous outputs.
