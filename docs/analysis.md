# Running the analysis

The pipeline — anaTuples, observables, histograms, plots — is the **standard FLAF chain**; follow
the **[FLAF full-workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)** for
the commands. This page collects what is **specific to HH→bb̄WW**.

```sh
ERA=Run3_2022
VER=dev

law run FLAF.Analysis.tasks.HistPlotTask --period $ERA --version $VER --workflow local --test 1000
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

## Categories

HH→bb̄WW is analysed in **resolved** and **boosted** categories (low- and high-p_T HH topologies).
Category and channel selection is driven by `config/global.yaml`; narrow or extend it there or via
your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/).

## Choosing which variables to histogram

As for any FLAF analysis, the variable set is controlled by the `variables:` list in
`user_custom.yaml` (or `--variables`). A short list keeps test runs fast:

```yaml
variables:
  - lep1_pt
  - ggF_DNN_HH
```

## Statistical interpretation

Continue to [Statistical inference](stat_inference.md) for datacards, limits and diagnostics.
