# Parametric DNN (PyTorch)

Training and validation of the parametric multiclass DNN used in the dilepton channel. The
production models in `config/DNN/DoubleLepton_Parametric_{Resolved,Boosted}_v1/` were trained
here with `model_configs/looseBTag_multiclass_Radion_pDNN_v6.yaml`; their `dnn_config.yaml` is
that file with the feature list flattened (see [Folds](#folds)).

One network is trained per regime (`resolved`, `boosted`), with the signal mass hypothesis
`X_mass` as an input. Mass-dependent features (`ll_dR_scaled`, `bb_dR_scaled`, …) are recomputed
for each hypothesis; background events are assigned a hypothesis drawn from the signal mass
points.

## Inputs

The training datasets are the `nParity{0..3}_Merged.root` / `nParity{0..3}_Merged_weight.root`
files written by `Studies/DNN/create_dataset.py`:

```sh
cd Studies/DNN
python3 create_dataset.py --config config/dataset_setup_doubleLep_merged_nocuts.yaml \
    --output-folder /eos/user/<u>/<you>/HH_bbWW/DNNDatasets/<tag>
```

`dataset_setup_doubleLep_merged_nocuts.yaml` is the setup of the `July25` dataset used by the v7
config (identical to the copy stored with that dataset). The setup of the `July24` dataset used
by v6 was not kept.

## Running

Run from this directory (the scripts import `src.model_helper` relative to it):

```sh
python3 train_pdnn.py --config model_configs/looseBTag_multiclass_Radion_pDNN_v6.yaml
python3 apply_pdnn.py --config model_configs/looseBTag_multiclass_Radion_pDNN_v6.yaml
python3 shape.py
```

| Script | Does | Writes (under `output_folder`) |
|---|---|---|
| `train_pdnn.py` | Trains one model per regime and parity fold, exports ONNX, evaluates on the validation fold | `nParity{f}_validation/pdnn_model_{regime}_nparity{f}.onnx`, `validation_applied.root`, training plots |
| `apply_pdnn.py` | Re-evaluates the exported ONNX models on the validation folds | `nParity{f}_validation/validation_applied.root` |
| `shape.py` | Combines the folds, bins DeepHME mass in pDNN slices, writes templates and stack plots | `combined_validation/shape_{cat}.root`, `pdnn_mass_stackplots/` |

`shape.py` takes no arguments: the input folder and binning are the constants at the top of the
file.

## Configs

| Config | Classes | Inputs | Dataset |
|---|---|---|---|
| `looseBTag_multiclass_Radion_pDNN_v6.yaml` | Signal, TT, Other | 21 common + regime features | `../Datasets/July24` |
| `looseBTag_multiclass_Radion_pDNN_v7.yaml` | Signal, TT, DY, ST, Other | v6 + `DeepHME_mass` | `July25` on EOS |

v6 is the production model. v7 is not deployed, and does not train five classes as written: the
network gets `len(class_names)` outputs, but `load_parametric_fold` assigns only three targets
(0 signal, 1 TT, 2 everything else), so the DY and ST outputs are never a target and output 2,
labelled DY, holds all non-TT background.

## Folds

`create_dataset.py` writes the events with `event % nParity == k` to bucket file
`nParity{k}_Merged.root`. Each split in the config has an integer `offset`, and model `i` uses
bucket `(i + offset) % nParity` for it: with the offsets 0/1/2/3 it trains on bucket `i`, uses
bucket `i+1` for the LR schedule, early stopping and the choice of the saved epoch, is validated
on `i+2`, and is meant for `i+3`. It is exported as `nparity{i}` (`model_name` with `{fold}`).

The deployed `config/DNN/*/dnn_config.yaml` uses the same schema, and
`Analysis/DNN_Application.py` inverts the app split: an event in bucket `b` is scored by model
`(b - app offset) % nParity`. A model copied into `config/DNN/` keeps its offsets unchanged.
