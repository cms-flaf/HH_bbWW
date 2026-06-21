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

## 1. Create datacards

```sh
cmsEnv python3 StatInference/dc_make/create_datacards.py \
  --input  PATH_TO_SHAPES \
  --output PATH_TO_CARDS \
  --config StatInference/config/x_hh_bbww_run3.yaml
```

The HH→bb̄WW Run 3 configuration is
[`x_hh_bbww_run3.yaml`](https://github.com/cms-flaf/StatInference/blob/main/config/x_hh_bbww_run3.yaml).

## 2. Run limits

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
