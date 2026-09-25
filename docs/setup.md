# Setup

HH→bb̄WW is installed and run like any FLAF analysis. The general procedure — prerequisites (CERN
account, CVMFS, grid certificate, SSH keys) and what the first `source env.sh` builds — is in the
**[FLAF installation guide](https://cms-flaf.github.io/FLAF/getting-started/installation/)**. This
page lists the HH→bb̄WW specifics.

## Clone (with git LFS)

```sh
git clone --recursive git@github.com:cms-flaf/HH_bbWW.git
cd HH_bbWW
git lfs pull
source env.sh
```

!!! danger "Two easy-to-miss steps"
    - `--recursive` — pulls the submodules (without it, imports fail on empty directories).
    - `git lfs pull` — fetches the large files HH→bb̄WW tracks with git LFS. Skipping it leaves
      pointer placeholders instead of real files.

    To run a specific central production, clone its tag, e.g. `git clone -b <version> --recursive …`.

## Analysis-specific submodules

| Submodule | Role |
|---|---|
| `DeepHME` | Neural reconstruction of the HH mass for bb̄WW. |
| `SyncTool` | Synchronisation/validation tooling. |
| `StatInference`, `inference` | Datacards and combine-based limits/fits (shared). |

These build automatically as part of `source env.sh`; you do not set them up by hand.

## Production model

The default [physics model](https://cms-flaf.github.io/FLAF/configuration/processes-and-models/)
for HH→bb̄WW is `Run3_Model_Radion` (set in `config/global.yaml`) — the spin-0 hypothesis. The
spin-2 one is `Run3_Model_BulkGraviton`: Radion and BulkGraviton are separate models because a
single model carrying both would be fitted as one combined signal. Both take their bb̄WW signal from
the privately produced X→HH samples, and neither selects the central X→YH samples. bb̄ττ
(`XtoHHto2Tau2B`) is commented out in the Radion model and in the dilepton datacard until the
private X→HH→bb̄ττ samples exist. The samples that do exist cover Run3_2022 through Run3_2023BPix;
later eras have no resonant sample produced yet, so both models load there with no signal in them.
An anaTuple production is not a fit, so it uses `Production_Model` (set in `config/user_custom.yaml`,
see the repository README), which carries both spins' X→HH bb̄WW signals and the same commented
bb̄ττ line. For fast local tests, use `TestModel`
in your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/) instead.
It holds two backgrounds — `custom_CI_Background_TT`, one t̄t dataset, and
`custom_CI_Background_DY`, one DY dataset carrying the same DY stitcher the era configures —
plus one signal and one data process. Each CI background mirrors the `processors:` of the real
process it stands for, so the [stitching](https://cms-flaf.github.io/FLAF/concepts/stitching/)
is exercised over the whole anaTuple → merge chain; keep them in step when you change the real
ones.

## Next

- [Running the analysis](analysis.md) — the bb̄WW-specific run notes (DeepHME, b-tag-shape cache,
  categories).
- [FLAF → Full workflow](https://cms-flaf.github.io/FLAF/workflow/walkthrough/) — the common
  pipeline, stage by stage.
- [FLAF → HTCondor](https://cms-flaf.github.io/FLAF/workflow/htcondor/) /
  [CRAB](https://cms-flaf.github.io/FLAF/workflow/crab/) — CERN batch and full WLCG submission.
